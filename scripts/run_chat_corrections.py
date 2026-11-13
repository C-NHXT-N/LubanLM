"""运行已批准的v6目标函数A/B；完整语义评分由审阅者提供，绝不自动打分。"""
from scripts.common import RUNS,ROOT,read_json,write_json,sha256,utc
from scripts.train_chat_v2 import used
import subprocess
import sys
import os

def main():
    if (RUNS/'pipeline.lock').exists() or (RUNS/'reports/gpu_budget.lock').exists():
        raise RuntimeError('existing controller/GPU task: do not start competing work')
    if min(180-used('sft-'),1380-used()) < 25:
        raise RuntimeError('insufficient original budget for approved A/B trials')
    perf=read_json(RUNS/'reports/sft-correction-performance.json')
    suite=ROOT/'eval/suites/chat-v6.json'
    init=RUNS/'checkpoints/50m-chat-v5/last.pt'
    plans=[]
    for objective in ('token','task-balanced'):
        name='50m-chat-v6-'+('token' if objective=='token' else 'balanced')
        path=RUNS/'reports'/f'{name}-plan.json'
        plan=dict(name=name,data_version='sft-v6',init_checkpoint=str(init),init_sha256=sha256(init),
                  base_checkpoint=str(RUNS/'checkpoints/50m-base-v2/last.pt'),lr=1e-5,epochs=20,
                  minutes=10,steps=2000,sft_batching='flat',sft_sampling='stratified',
                  sft_objective=objective,prefetch_batches=2 if perf['flat_prefetch_adopted'] else 0,
                  source_validation=True,validation_every_steps=1000,early_stop_patience=0,
                  parent_sft_tokens=5776672,checkpoint_name='best.pt',development_only=True,
                  suite_file=str(suite),suite_sha256=sha256(suite),created_utc=utc(),
                  rationale='Identical initial weights, prefix-stable v6 data, task quotas, updates, learning rate and decoder; only loss objective differs. Final outputs never read by training.')
        if not path.exists():write_json(path,plan)
        else:
            old=read_json(path)
            for key in plan:
                if key!='created_utc' and old.get(key)!=plan[key]:raise ValueError('immutable comparison plan changed')
        plans.append(str(path))
        subprocess.run([sys.executable,'-u','-m','scripts.improve_chat_v2','--plan',str(path)],cwd=ROOT,check=True)
    write_json(RUNS/'reports/v6-comparison-pending.json',dict(plans=plans,created_utc=utc(),phase='awaiting_complete_semantic_reviews',no_automatic_grading=True))
    print('Both A/B candidates await complete semantic reviews.',flush=True)

if __name__=='__main__':
    awake=subprocess.Popen([sys.executable,'-m','scripts.keep_awake','--pid',str(os.getpid())],cwd=ROOT,
                           stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
                           creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    try:main()
    finally:
        # 仅结束本控制器创建的辅助进程；其退出释放临时电源请求。
        if awake.poll() is None:
            awake.terminate();awake.wait(timeout=10)
