"""A short Windows-scheduled tick, independent of the controller's lifetime."""
import argparse
from datetime import datetime,timezone
import os
from pathlib import Path
import shutil
import subprocess
import sys
import psutil
from scripts.common import ROOT,read_json,write_json,utc,sha256
from scripts.policy_versions import digest

TASK_NAME='LubanLM-Teaching-10B-Supervisor'
MODULES=('scripts.run_general_10b','scripts.prepare_general_10b','scripts.prepare_sft_100k','training.train')

def alive(pid,created,modules):
    try:
        process=psutil.Process(int(pid))
        if created is None or abs(process.create_time()-float(created))>.1:return False
        return process.is_running() and any(module in process.cmdline() for module in modules)
    except (psutil.Error,ValueError,TypeError):return False

def remove_task():
    subprocess.run(['schtasks','/Delete','/TN',TASK_NAME,'/F'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)

def decision(plan,state,previous,available,free,signature,tail=''):
    status=state.get('status','')
    if status in ('completed_pending_semantic_review','cancelled'):return 'complete'
    if free<plan['disk_free_bytes_floor'] or available<plan.get('supervisor_start_memory_bytes',4<<30):return 'resource_wait'
    if status.startswith('awaiting_'):return 'review_wait'
    error=state.get('error','')+' '+tail
    if 'MemoryError' in error:return 'repair_wait'
    if 'RESOURCE_WAIT' in error:return 'start'
    if any(x in error for x in ('DATA_SHORTFALL','Public SFT shortfall','BUDGET_EXHAUSTED','GPU cumulative budget exhausted','ValueError:', 'policy changed','mismatch','integrity failed')):return 'repair_wait'
    if previous.get('signature')==signature and previous.get('failed_without_progress',0)>=3:return 'repair_wait'
    return 'start'

def clear_stale_lock(path,modules):
    if not path.exists():return
    pid=int(path.read_text(encoding='utf-8').strip())
    try:
        if any(m in psutil.Process(pid).cmdline() for m in modules):raise RuntimeError('A matching lock owner is still alive')
    except psutil.NoSuchProcess:pass
    path.unlink()

def tick(plan_path,cancel=False):
    plan_path=Path(plan_path).resolve();plan=read_json(plan_path)
    runs,data=Path(plan['runs_root']),Path(plan['data_root']);reports=runs/'reports';reports.mkdir(parents=True,exist_ok=True)
    lock=reports/'supervisor-tick.lock'
    try:fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:
        try:psutil.Process(int(lock.read_text()));return
        except psutil.NoSuchProcess:lock.unlink();fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:
        controller_path=reports/'controller.json';supervisor_path=reports/'supervisor.json'
        state=read_json(controller_path) if controller_path.exists() else {}
        previous=read_json(supervisor_path) if supervisor_path.exists() else {}
        if cancel or (reports/'cancel-requested.json').exists():
            for pid,created in ((state.get('child_pid'),state.get('child_created')),(state.get('pid'),state.get('process_created'))):
                if alive(pid,created,MODULES):psutil.Process(pid).terminate()
            write_json(supervisor_path,dict(status='cancelled',updated_utc=utc()));remove_task();return
        if alive(state.get('pid'),state.get('process_created'),MODULES[:1]):
            age=(datetime.now(timezone.utc)-datetime.fromisoformat(state.get('updated_utc',state['started_utc']))).total_seconds()
            # The parent emits a heartbeat every 10 seconds even during long
            # downloads/training. A stale parent may be replaced; its live child
            # is adopted below and never duplicated or killed as "hung".
            if age>180:
                process=psutil.Process(state['pid']);process.terminate()
                try:process.wait(timeout=10)
                except psutil.TimeoutExpired:
                    write_json(supervisor_path,dict(previous,status='stale_controller_wait',updated_utc=utc()));return
            else:
                write_json(supervisor_path,dict(previous,status='running',updated_utc=utc(),controller_pid=state['pid'],child_pid=state.get('child_pid'),controller_updated_utc=state.get('updated_utc')));return
        if alive(state.get('child_pid'),state.get('child_created'),MODULES[1:]):
            write_json(supervisor_path,dict(previous,status='orphan_worker_adopted',updated_utc=utc(),child_pid=state['child_pid']));return
        if previous.get('status')=='started' and alive(previous.get('controller_pid'),previous.get('controller_created'),MODULES[:1]):return
        progress=[]
        for name in ('preparation_progress.json','sft_preparation_progress.json'):
            path=reports/name
            if path.exists():
                value=read_json(path);progress.append({k:v for k,v in value.items() if k not in ('updated_utc','elapsed_seconds','pid','rss_bytes','available_memory_bytes')})
        code=['scripts/teaching_supervisor.py','scripts/run_general_10b.py','scripts/prepare_general_10b.py','scripts/sft_disk.py','scripts/prepare_sft_100k.py','scripts/corpus_quality.py']
        configuration=digest([sha256(plan_path),*[(f,sha256(ROOT/f)) for f in code]])
        signature=digest(progress);tail=''
        stderr=previous.get('stderr')
        if stderr and Path(stderr).exists():
            with Path(stderr).open('rb') as stream:stream.seek(max(0,Path(stderr).stat().st_size-12000));tail=stream.read().decode('utf-8',errors='replace')
        if previous.get('configuration')!=configuration:
            previous={};tail=''
            if state.get('status')=='failed':state=dict(state,error='')
        action=decision(plan,state,previous,psutil.virtual_memory().available,shutil.disk_usage(data).free,signature,tail)
        if action=='review_wait':
            kind='sft' if 'sft' in state.get('status','') else 'pretrain'
            if (reports/f'{kind}_content_review.json').exists():
                from scripts.content_review_gate import verify_content_review
                try:verify_content_review(plan_path,kind);action='start'
                except (ValueError,KeyError,FileNotFoundError):pass
        info=dict(previous,status=action,updated_utc=utc(),signature=signature,configuration=configuration,available_memory_bytes=psutil.virtual_memory().available,disk_free_bytes=shutil.disk_usage(data).free)
        if action=='complete':write_json(supervisor_path,info);remove_task();return
        if action!='start':write_json(supervisor_path,info);return
        for process in psutil.process_iter(['pid','cmdline','cwd']):
            try:
                command=process.info['cmdline'] or []
                if any(m in command for m in MODULES) and (str(plan_path) in command or process.info['cwd']==str(ROOT)):
                    info.update(status='unregistered_worker_wait',conflicting_pid=process.pid);write_json(supervisor_path,info);return
            except psutil.Error:continue
        clear_stale_lock(runs/'controller.lock',MODULES)
        clear_stale_lock(reports/'gpu_budget.lock',('training.train','scripts.benchmark_general_pretrain','scripts.benchmark_sft_10b'))
        attempt=int(previous.get('attempt',0))+1
        logs=runs/'supervisor-logs';logs.mkdir(parents=True,exist_ok=True)
        stdout,stderr=logs/f'{attempt:04d}.stdout.log',logs/f'{attempt:04d}.stderr.log'
        env=os.environ.copy();(data/'tmp').mkdir(parents=True,exist_ok=True)
        env.update(PYTHONIOENCODING='utf-8',TEMP=str(data/'tmp'),TMP=str(data/'tmp'),RAYON_NUM_THREADS='4',TOKENIZERS_PARALLELISM='true')
        with stdout.open('ab') as out,stderr.open('ab') as err:
            process=subprocess.Popen([sys.executable,'-B','-u','-m','scripts.run_general_10b','--plan',str(plan_path)],cwd=ROOT,env=env,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
        failures=int(previous.get('failed_without_progress',0))+1 if previous.get('signature')==signature else 0
        info.update(status='started',attempt=attempt,controller_pid=process.pid,controller_created=psutil.Process(process.pid).create_time(),stdout=str(stdout),stderr=str(stderr),failed_without_progress=failures)
        write_json(supervisor_path,info)
    finally:lock.unlink(missing_ok=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',default='configs/50m-general-pretrain-10b-teaching.json');parser.add_argument('--cancel',action='store_true');args=parser.parse_args();tick(args.plan,args.cancel)

if __name__=='__main__':main()
