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

def controller_staleness(previous,signature,age):
    unchanged=previous.get('controller_work_signature')==signature
    suspect=age>180 and unchanged
    return dict(controller_work_signature=signature,controller_stale_observed=suspect),suspect and previous.get('controller_stale_observed',False)

def clear_stale_lock(path,modules):
    from scripts.worker_health import acquire_lock,release_lock
    if not path.exists():return
    owner=acquire_lock(path,modules)
    if owner is None:raise RuntimeError('A matching lock owner is still alive')
    release_lock(path,owner)

def tick(plan_path,cancel=False):
    from scripts.worker_health import (acquire_lock,release_lock,load,durable_progress,
        work_signature,health_check,reconcile_cpu)
    import time
    plan_path=Path(plan_path).resolve();plan=read_json(plan_path)
    runs,data=Path(plan['runs_root']),Path(plan['data_root']);reports=runs/'reports';reports.mkdir(parents=True,exist_ok=True)
    lock=reports/'supervisor-tick.lock';owner=acquire_lock(lock,('scripts.teaching_supervisor',))
    if owner is None:return
    try:
        if (reports/'maintenance.json').exists():return
        controller_path=reports/'controller.json';supervisor_path=reports/'supervisor.json'
        state=load(controller_path);previous=load(supervisor_path)
        code=['scripts/teaching_supervisor.py','scripts/worker_health.py','scripts/run_general_10b.py','scripts/prepare_general_10b.py','scripts/sft_disk.py','scripts/prepare_sft_100k.py','scripts/corpus_quality.py']
        configuration=digest([sha256(plan_path),*[(f,sha256(ROOT/f)) for f in code]])
        signature=digest(durable_progress(data))
        # GPU durable progress is independent of changing heartbeat/stage names.
        summaries=[]
        for f in (runs/'checkpoints').glob('*/summary.json'):
            r=load(f);summaries.append((str(f),r.get('step'),r.get('supervised_tokens')))
        checkpoints=[]
        for f in (runs/'checkpoints').glob('*/last.pt'):
            stat=f.stat();checkpoints.append((str(f),stat.st_size,stat.st_mtime_ns))
        signature=digest([signature,sorted(summaries),sorted(checkpoints)])
        info=dict(previous,version=2,updated_utc=utc(),signature=signature,configuration=configuration,
            available_memory_bytes=psutil.virtual_memory().available,disk_free_bytes=shutil.disk_usage(data).free)
        if previous.get('signature')!=signature:info['failed_without_progress']=0
        if cancel or (reports/'cancel-requested.json').exists():
            for pid,created in ((state.get('child_pid'),state.get('child_created')),(state.get('pid'),state.get('process_created'))):
                if alive(pid,created,MODULES):psutil.Process(pid).terminate()
            info['status']='cancelled';write_json(supervisor_path,info);remove_task();return
        parent_alive=alive(state.get('pid'),state.get('process_created'),MODULES[:1])
        child_alive=alive(state.get('child_pid'),state.get('child_created'),MODULES[1:])
        if child_alive:
            pid=state['child_pid'];created=state['child_created'];phase=state.get('status','')
            if phase in ('preparing_10b_cpu','preparing_sft_cpu'):
                # Worker-owned ledger continues if its controller disappears.
                heart=load(reports/'worker_heartbeat.json')
                valid=heart.get('pid')==pid and heart.get('created')==created
                age=time.time()-datetime.fromisoformat(heart['updated_utc']).timestamp() if valid else time.time()-created
                entries=load(reports/'cpu_preparation_budget.json',[])
                entry=next((r for r in entries if r.get('pid')==pid and r.get('created')==created),None)
                if entry:
                    spent=reconcile_cpu(reports/'cpu_preparation_budget.json',pid,created,phase,entry['started_at'])
                else:spent=sum(r['seconds'] for r in entries)
            else:
                age=0;spent=0 # Training health comes from work position and process activity.
            health_previous=previous if previous.get('watched_child')==[pid,created] else {}
            health,stalled=health_check(health_previous,work_signature(plan,pid),age)
            info.update(health,watched_child=[pid,created],child_pid=pid,child_heartbeat_age=age)
            exhausted=spent>=plan['preparation_hours_cap']*3600
            if stalled or exhausted:
                for target,identity,modules in ((pid,created,MODULES[1:]),(state.get('pid'),state.get('process_created'),MODULES[:1])):
                    if alive(target,identity,modules):
                        p=psutil.Process(target);p.terminate()
                        try:p.wait(timeout=10)
                        except psutil.TimeoutExpired:
                            info['status']='worker_stop_wait';write_json(supervisor_path,info);return
                if phase in ('preparing_10b_cpu','preparing_sft_cpu') and entry:
                    reconcile_cpu(reports/'cpu_preparation_budget.json',pid,created,phase,entry['started_at'],finished=True)
                state=dict(state,status='failed',error='PREPARATION_BUDGET_EXHAUSTED' if exhausted else 'Worker stalled')
                write_json(controller_path,state)
                info.update(status='repair_wait' if exhausted else 'worker_stalled',stall_reason='budget' if exhausted else 'heartbeat_or_work_position')
                write_json(supervisor_path,info);return
            if not parent_alive:
                info['status']='orphan_worker_adopted';write_json(supervisor_path,info);return
        if parent_alive:
            age=(datetime.now(timezone.utc)-datetime.fromisoformat(state.get('updated_utc',state['started_utc']))).total_seconds()
            watched=[state['pid'],state['process_created']]
            prior=previous if previous.get('watched_controller')==watched else {}
            controller_health,stale=controller_staleness(prior,work_signature(plan,state['pid']),age)
            info.update(controller_health,watched_controller=watched)
            if stale:
                process=psutil.Process(state['pid']);process.terminate()
                try:process.wait(timeout=10)
                except psutil.TimeoutExpired:
                    info['status']='stale_controller_wait';write_json(supervisor_path,info);return
                info['status']='orphan_worker_adopted' if child_alive else 'controller_stale';write_json(supervisor_path,info);return
            info.update(status='stall_suspected' if info.get('stall_suspected_at') else 'running',controller_pid=state['pid'],controller_updated_utc=state.get('updated_utc'))
            write_json(supervisor_path,info);return
        if previous.get('status')=='started' and alive(previous.get('controller_pid'),previous.get('controller_created'),MODULES[:1]):
            info['status']='started';write_json(supervisor_path,info);return
        # Close dead CPU leases at last observed heartbeat, not at next login hours later.
        for entry in load(reports/'cpu_preparation_budget.json',[]):
            if entry.get('version')==2 and not entry.get('finished') and not alive(entry['pid'],entry['created'],MODULES[1:]):
                reconcile_cpu(reports/'cpu_preparation_budget.json',entry['pid'],entry['created'],entry['phase'],entry['started_at'],finished=True,ended_at=min(time.time(),entry.get('updated_at',entry['started_at']+entry['seconds'])+10))
        tail='';stderr=previous.get('stderr')
        if stderr and Path(stderr).exists():
            with Path(stderr).open('rb') as stream:stream.seek(max(0,Path(stderr).stat().st_size-12000));tail=stream.read().decode('utf-8',errors='replace')
        if previous.get('configuration')!=configuration:
            tail=''
            if state.get('status')=='failed':state=dict(state,error='')
        action=decision(plan,state,info,info['available_memory_bytes'],info['disk_free_bytes'],signature,tail)
        if state.get('status')=='base_complete_sft_data_shortfall':action='repair_wait'
        if action=='review_wait':
            kind='sft' if 'sft' in state.get('status','') else 'pretrain'
            if (reports/f'{kind}_content_review.json').exists():
                from scripts.content_review_gate import verify_content_review
                try:verify_content_review(plan_path,kind);action='start'
                except (ValueError,KeyError,FileNotFoundError):pass
        info['status']=action
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
        # Count failed attempts, including quick failures before any state file update.
        failed=int(info.get('failed_without_progress',0))+1 if previous.get('attempt') else 0
        for field in ('child_pid','watched_child','child_heartbeat_age','controller_updated_utc','work_signature','work_changed_at','stall_suspected_at'):info.pop(field,None)
        info.update(status='started',attempt=attempt,controller_pid=process.pid,controller_created=psutil.Process(process.pid).create_time(),stdout=str(stdout),stderr=str(stderr),failed_without_progress=failed)
        write_json(supervisor_path,info)
    finally:release_lock(lock,owner)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',default='configs/50m-general-pretrain-10b-teaching.json');parser.add_argument('--cancel',action='store_true');args=parser.parse_args();tick(args.plan,args.cancel)

if __name__=='__main__':main()
