"""持久化 24h 实验控制器。模型能力判定等待逐题评分，绝不自动宣布达标。"""
from __future__ import annotations
from scripts.common import DATA, RUNS, ROOT, write_json, read_json, utc
import subprocess
import sys
import os
import time
from pathlib import Path

STATUS=RUNS/'reports/pipeline_status.json'
LOGS=RUNS/'logs'
BASE=RUNS/'checkpoints/50m-base-v2'
CONFIG=ROOT/'configs/config_50m.json'
PARENT=Path('D:/LubanLM-Runs/checkpoints/50m-base/last.pt')
DEFAULT_PROFILE=dict(execution='legacy',attention_backend='manual',optimizer_backend='loop',pin_memory=False,micro_batch=2,accumulation=4)

def profile_options(profile):
    result=['--micro-batch',profile['micro_batch'],'--accumulation',profile['accumulation'],'--execution',profile['execution'],'--attention-backend',profile['attention_backend'],'--optimizer-backend',profile['optimizer_backend'],'--validation-micro-batch',2]
    if profile.get('pin_memory'):result+=['--pin-memory']
    return result

def used(prefix=None):
    path=RUNS/'reports/gpu_budget.json'
    entries=read_json(path) if path.exists() else []
    return sum(r['seconds'] for r in entries if prefix is None or r['stage'].startswith(prefix))/60

def status(phase,**kwargs):
    write_json(STATUS,dict(phase=phase,pid=os.getpid(),updated_utc=utc(),gpu_minutes=used(),capability_status='not_accepted',**kwargs))

def run(module,args,name):
    LOGS.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,'-u','-m',module]+[str(x) for x in args]
    log=LOGS/(name+'.log')
    with log.open('a',encoding='utf-8') as output:
        output.write('\n'+utc()+' '+repr(command)+'\n');output.flush()
        process=subprocess.Popen(command,cwd=ROOT,stdout=output,stderr=subprocess.STDOUT)
        status(name,child_pid=process.pid,log=str(log),command=command)
        code=process.wait()
    if code:
        raise RuntimeError(f'{name} exited {code}; inspect {log}')

class ReviewPending(Exception):
    """没有评分时正常退出，解除与控制器绑定的保活；不是训练失败。"""

def wait_review(raw):
    review=raw.with_suffix('.review.json')
    status('awaiting_development_review',answers=str(raw),review=str(review))
    if not review.exists():
        raise ReviewPending(str(raw))
    result=read_json(review)
    if result.get('answers_sha256') is None or 'qualified_rate' not in result or not result.get('review_complete'):
        raise ValueError('review must identify raw answers and include complete per-question grading')
    from scripts.common import sha256
    if result['answers_sha256']!=sha256(raw):
        raise ValueError('review hash does not match raw answers')
    return result

def candidate(checkpoint,name,probe=False):
    from data.prepare import load_jsonl,save_jsonl
    folder=RUNS/'checkpoints'/name
    summary=folder/'summary.json'
    if not summary.exists():
        remaining=180-used('sft-')
        if remaining<5:
            return None
        options=['--config',CONFIG,'--stage','sft','--init',checkpoint,'--output',folder,'--lr','5e-5','--epochs',1 if probe else 2,'--minutes',min(25 if probe else remaining,remaining),'--full-sft-validation','--save-every-minutes',30,'--log-every',100]
        profile_path=RUNS/'reports/execution_profile.json'
        profile=read_json(profile_path).get('sft',DEFAULT_PROFILE) if profile_path.exists() and not probe else DEFAULT_PROFILE
        options+=profile_options(profile)
        if probe:
            subset=DATA/'processed/sft-probe-tokens.jsonl'
            if not subset.exists():
                rows=load_jsonl(DATA/'processed/sft-train-tokens.jsonl')
                # 固定子集包含所有基础样本，余量从公共样本固定步长抽取。
                authored=[r for r in rows if r['source'].startswith('authored-')]
                public=[r for r in rows if not r['source'].startswith('authored-')]
                step=max(1,len(public)//max(1,5000-len(authored)))
                save_jsonl(subset,(authored+public[::step])[:5000])
            options+=['--sft-train',subset]
        else:
            improved=DATA/'processed/sft-v3'
            if (improved/'manifest.json').exists():
                options+=['--sft-train',improved/'train-tokens.jsonl','--sft-validation',improved/'validation-tokens.jsonl']
        run('training.train',options,'sft-'+name)
    raw=RUNS/'reports'/(name+'-dev.json')
    if not raw.exists():
        run('eval.chat_suite',['--checkpoint',folder/'last.pt','--split','dev','--output',raw,'--minutes',15],name+'-dev')
    review=wait_review(raw)
    return dict(name=name,base_checkpoint=str(checkpoint),checkpoint=str(folder/'last.pt'),qualified_rate=review['qualified_rate'],review=review)

def main():
    if str(DATA)=='D:\\LubanLM-Data' or RUNS==Path('D:/LubanLM-Runs'):
        raise ValueError('set separate LUBAN_DATA and LUBAN_RUNS for v2')
    RUNS.mkdir(parents=True,exist_ok=True)
    owner=RUNS/'pipeline.lock'
    with owner.open('x') as lock:
        lock.write(str(os.getpid()))
    try:
        pending=RUNS/'reports/controller_reload_pending.json'
        if pending.exists():
            reload_state=read_json(pending)
            if reload_state.get('old_controller_pid')!=os.getpid() and not reload_state.get('handled'):
                reload_state.update(handled=True,new_controller_pid=os.getpid(),handled_utc=utc())
                write_json(pending,reload_state)
        status('starting')
        if not (RUNS/'reports/data-v2.json').exists():
            run('data.prepare_chat_v2',[],'prepare-data')
        checks=read_json(RUNS/'reports/training_checks.json') if (RUNS/'reports/training_checks.json').exists() else None
        if not checks or checks.get('data_root')!=str(DATA):
            run('scripts.verify_training',[],'verify-training')
        candidates=[]
        state_path=RUNS/'reports/candidates.json'
        if state_path.exists():
            candidates=read_json(state_path)
        while used('pretrain-')<1197:
            base_summary=read_json(BASE/'summary.json') if (BASE/'summary.json').exists() else None
            # 在阶段等待审阅时重启控制器，也必须先处理这一阶段，不能跳过检查。
            if base_summary and base_summary['stop_reason']=='checkpoint_review':
                milestone=round(base_summary['supervised_tokens']/1e6)
                if milestone in (100,200) and not any(c['name']==f'50m-probe-{milestone}m' for c in candidates):
                    c=candidate(BASE/f"step-{base_summary['step']:06d}.pt",f'50m-probe-{milestone}m',True)
                    if c:
                        candidates.append(c);write_json(state_path,candidates)
                    if c and c['review'].get('development_ready'):
                        break
            if base_summary and base_summary['stop_reason'] in ('step_target','time_budget'):
                break
            profile_path=RUNS/'reports/execution_profile.json'
            profile_state=read_json(profile_path) if profile_path.exists() else {}
            profile=profile_state.get('pretrain',DEFAULT_PROFILE)
            options=['--config',CONFIG,'--output',BASE,'--tokens',400000000,'--minutes',1200-used('pretrain-'),'--length',256,'--lr','3e-4','--warmup-steps',100,'--validation-tokens',500000,'--validation-every-tokens',25000000,'--checkpoint-tokens',100000000,'--save-every-minutes',30,'--log-every',100,'--pause-after-checkpoint']+profile_options(profile)
            if profile_state.get('migration_required') and not profile_state.get('rollback_used'):
                options+=['--performance-reference-loss',profile_state['reference_validation_loss']]
            if (BASE/'last.pt').exists() or (BASE/'latest.pt').exists():
                from training.train import load_checkpoint
                paths=[p for p in [BASE/'last.pt',BASE/'latest.pt'] if p.exists()]
                resume=max(paths,key=lambda p:p.stat().st_mtime)
                if profile_state.get('rollback_pending'):
                    resume=Path(profile_state['checkpoint'])
                from training.execution import execution_settings
                settings=load_checkpoint(resume)['settings']
                if execution_settings(settings)!=execution_settings(profile) or any(settings[k]!=profile[k] for k in ('micro_batch','accumulation')):
                    options+=['--migrate-execution']
                # 精确恢复同一目标/学习率/数据/采样器，不延长现有实验目标。
                options+=['--resume',resume]
            else:
                options+=['--continue-from',PARENT]
            run('training.train',options,'pretrain-50m-v2')
            summary=read_json(BASE/'summary.json')
            if profile_state.get('rollback_pending') and summary['stop_reason']!='performance_guard':
                profile_state['rollback_pending']=False
                write_json(profile_path,profile_state)
            if summary['stop_reason']=='performance_guard':
                if profile_state.get('rollback_used'):
                    raise RuntimeError('validation guard repeated after rollback')
                import shutil
                failure=RUNS/'checkpoints/performance-switch/failed'
                failure.mkdir(parents=True,exist_ok=True)
                for name in ('last.pt','latest.pt','summary.json','validation_latest.json'):
                    if (BASE/name).exists():shutil.copy2(BASE/name,failure/name)
                profile_state.update(pretrain=DEFAULT_PROFILE,rollback_used=True,rollback_pending=True,rollback_utc=utc(),rollback_reason='fixed validation loss increased by >0.1')
                write_json(profile_path,profile_state)
                continue
            milestone=round(summary['supervised_tokens']/1e6)
            if summary['stop_reason']=='checkpoint_review' and milestone in (100,200):
                name=f'50m-probe-{milestone}m'
                if not any(c['name']==name for c in candidates):
                    # 使用不可被下一阶段 last.pt 覆盖的 milestone 权重。
                    ckpt=BASE/f"step-{summary['step']:06d}.pt"
                    c=candidate(ckpt,name,True)
                    if c:
                        candidates.append(c);write_json(state_path,candidates)
                if candidates and candidates[-1]['review'].get('development_ready'):
                    break
        final_base=BASE/'last.pt'
        # 不在末次阶段下降明显时自动选最新；先运行末次 probe，再按开发集排序。
        if not any(c['name']=='50m-probe-final' for c in candidates):
            c=candidate(final_base,'50m-probe-final',True)
            if c:
                candidates.append(c);write_json(state_path,candidates)
        if not candidates:
            status('not_accepted',reason='No reviewed candidate within budget')
            return
        best=max(candidates,key=lambda c:c['qualified_rate'])
        selected=candidate(Path(best['base_checkpoint']),'50m-chat-v2',False)
        if selected is None:
            status('not_accepted',reason='SFT budget exhausted',best=best)
            return
        if not selected['review'].get('development_ready'):
            status('needs_improvement',reason='Final SFT did not pass development criteria',best=selected)
            return
        for split in ('test','core','sampling'):
            raw=RUNS/'reports'/f'50m-chat-v2-{split}.json'
            if not raw.exists():
                run('eval.chat_suite',['--checkpoint',selected['checkpoint'],'--split',split,'--output',raw,'--minutes',20],f'final-{split}')
        status('awaiting_final_acceptance',checkpoint=selected['checkpoint'],base_checkpoint=best['base_checkpoint'],note='Review all test/core/sampling outputs; export and document only after acceptance.')
    except ReviewPending:
        return
    except BaseException as exc:
        status('error',error=repr(exc))
        raise
    finally:
        owner.unlink(missing_ok=True)

if __name__=='__main__':
    main()
