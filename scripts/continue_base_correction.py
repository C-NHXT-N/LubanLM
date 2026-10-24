"""仅在已审阅清洁SFT仍失败后补充Base；保持原预算，完成后退出等待人工评分。"""
from scripts.common import ROOT,RUNS,read_json,write_json,sha256,utc
from scripts.train_chat_v2 import used,run,status,profile_options,DEFAULT_PROFILE,ReviewPending
from training.train import load_checkpoint
from pathlib import Path
import os
import sys
import subprocess

def main():
    decision=read_json(RUNS/'reports/v6-correction-decision.json')
    if decision.get('next_action')!='controlled_base_extension':
        raise ValueError('reviewed A/B decision does not authorize Base fallback')
    for raw_path in decision['reviewed_answers']:
        raw=Path(raw_path);review=read_json(raw.with_suffix('.review.json'))
        if review.get('answers_sha256')!=sha256(raw) or not review.get('review_complete'):
            raise ValueError('unverified development review')
        if review.get('development_ready'):
            raise ValueError('a reviewed candidate already passed development; evaluate it before Base fallback')
    if (RUNS/'pipeline.lock').exists() or (RUNS/'reports/gpu_budget.lock').exists():
        raise RuntimeError('existing controller/GPU work; retain locks')
    owner=RUNS/'pipeline.lock'
    with owner.open('x') as stream:stream.write(str(os.getpid()))
    awake=subprocess.Popen([sys.executable,'-m','scripts.keep_awake','--pid',str(os.getpid())],cwd=ROOT,
                           stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
                           creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    try:
        source=RUNS/'checkpoints/50m-base-v2/last.pt'
        folder=RUNS/'checkpoints/50m-base-correction-v6'
        plan_path=RUNS/'reports/base-correction-v6-plan.json'
        plan=dict(parent_checkpoint=str(source),parent_sha256=sha256(source),new_tokens=400000000,
                  length=256,lr=1e-4,warmup_steps=1000,cap_minutes=300,
                  probe_data_version=decision.get('data_version','sft-v6'),
                  policy='Same immutable Wikipedia dataset, optimizer carried with --continue-from. Not a promise of capability. Compare two identical <=10min SFT probes after extension.',created_utc=utc())
        if plan_path.exists():
            saved=read_json(plan_path)
            if any(saved[k]!=v for k,v in plan.items() if k!='created_utc'):raise ValueError('immutable Base plan changed')
        else:
            plan['pretrain_minutes_at_start']=used('pretrain-')
            write_json(plan_path,plan)
        plan=read_json(plan_path)
        remaining=min(300-(used('pretrain-')-plan['pretrain_minutes_at_start']),1200-used('pretrain-'),1380-used())
        if remaining<10 or 180-used('sft-')<25:
            status('not_accepted',reason='insufficient original budget for controlled Base/probes',checkpoint=str(source));return
        if not (folder/'summary.json').exists() or read_json(folder/'summary.json')['stop_reason']=='time_budget':
            profile=read_json(RUNS/'reports/execution_profile.json').get('pretrain',DEFAULT_PROFILE)
            args=['--config',ROOT/'configs/config_50m.json','--output',folder,'--tokens',400000000,'--minutes',remaining,
                  '--length',256,'--lr','1e-4','--warmup-steps',1000,'--validation-tokens',500000,
                  '--validation-every-tokens',25000000,'--checkpoint-tokens',100000000,
                  '--save-every-minutes',30,'--log-every',100]+profile_options(profile)
            recovery=[p for p in (folder/'latest.pt',folder/'last.pt') if p.exists()]
            args+=['--resume',max(recovery,key=lambda p:p.stat().st_mtime)] if recovery else ['--continue-from',source]
            run('training.train',args,'pretrain-base-correction-v6')
        suite=ROOT/'eval/suites/chat-v6.json'
        data=__import__('scripts.common',fromlist=['DATA']).DATA/'processed'/plan['probe_data_version']
        raws=[]
        for name,base in [('400m',source),('extended',folder/'last.pt')]:
            probe=RUNS/'checkpoints'/f'50m-v6-base-probe-{name}'
            if not (probe/'summary.json').exists():
                available=min(10,180-used('sft-'),1380-used())
                if available<5:
                    status('not_accepted',reason='SFT comparison budget exhausted',checkpoint=str(base));return
                args=['--config',ROOT/'configs/config_50m.json','--stage','sft','--init',base,'--output',probe,
                      '--steps',2000,'--epochs',20,'--lr',1e-5,'--warmup-steps',100,'--minutes',available,
                      '--full-sft-validation','--source-validation','--validation-every-steps',1000,
                      '--sft-train',data/'train-tokens.jsonl','--sft-validation',data/'validation-tokens.jsonl',
                      '--sft-batching','flat','--sft-sampling','stratified','--sft-objective','task-balanced',
                      '--log-every',100]+profile_options(read_json(RUNS/'reports/execution_profile.json')['sft'])
                run('training.train',args,'sft-v6-base-probe-'+name)
            if read_json(probe/'summary.json')['step'] != 2000:
                raise RuntimeError('Base comparison incomplete: do not compare unequal probe training steps')
            raw=RUNS/'reports'/f'50m-v6-base-probe-{name}-dev.json'
            if not raw.exists():
                run('eval.chat_suite',['--checkpoint',probe/'last.pt','--split','dev','--suite-file',suite,
                                      '--output',raw,'--minutes',min(15,1440-used())],'v6-base-probe-'+name+'-dev')
            raws.append(str(raw))
        status('awaiting_development_review',answers=raws,base_checkpoint=str(folder/'last.pt'),
               note='Two matched Base SFT probes need complete semantic review. No periodic monitoring.')
        write_json(RUNS/'reports/base-correction-v6-comparison-pending.json',dict(answers=raws,created_utc=utc()))
    except BaseException as exc:
        status('error',error=repr(exc));raise
    finally:
        owner.unlink(missing_ok=True)
        if awake.poll() is None:awake.terminate();awake.wait(timeout=10)

if __name__=='__main__':main()
