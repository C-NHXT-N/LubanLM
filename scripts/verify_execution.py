"""实际磁盘检查点的优化路径恢复；保持训练目标，拒绝静默配置迁移。"""
from scripts.common import DATA, RUNS, ROOT, read_json, write_json
from training.train import load_checkpoint
from scripts.benchmark_training import assert_tree_equal
import subprocess
import sys
import torch

def main():
    config=read_json(ROOT/'configs/config_50m.json')
    config.update(name='execution-resume-check',hidden_size=64,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=16,intermediate_size=128)
    path=RUNS/'reports/execution-resume-config.json';write_json(path,config)
    common=[sys.executable,'-m','training.train','--config',str(path),'--steps','4','--minutes','1','--length','32','--micro-batch','4','--accumulation','1','--execution','optimized','--attention-backend','sdpa','--optimizer-backend','fused','--pin-memory','--validation-micro-batch','2','--warmup-steps','2']
    results={}
    for stage in ('pretrain','sft'):
        continuous=RUNS/f'checkpoints/performance-switch/verify-{stage}-continuous'
        resumed=RUNS/f'checkpoints/performance-switch/verify-{stage}-resumed'
        args=common+['--stage',stage]
        if not (continuous/'last.pt').exists():subprocess.run(args+['--output',str(continuous)],cwd=ROOT,check=True)
        if not (resumed/'last.pt').exists():subprocess.run(args+['--resume',str(continuous/'step-000002.pt'),'--output',str(resumed)],cwd=ROOT,check=True)
        a,b=load_checkpoint(continuous/'last.pt'),load_checkpoint(resumed/'last.pt')
        for key in ('model','optimizer','scaler','data_rng','torch_rng','cuda_rng','sampler','step','input_tokens','supervised_tokens'):
            assert_tree_equal(a[key],b[key])
        results[stage]=dict(weights_optimizer_rng_sampler_exact=True,input_tokens=a['input_tokens'],supervised_tokens=a['supervised_tokens'])
    write_json(RUNS/'reports/execution_resume_checks.json',dict(results=results,passed=True))
    print('optimized disk resume checks passed')

if __name__=='__main__':main()
