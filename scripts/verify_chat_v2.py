"""继续训练与精确恢复回归检查，另验证不完整评分无法通过。"""
from scripts.common import DATA, RUNS, ROOT, read_json, write_json, sha256
from training.train import load_checkpoint
from eval.grade_chat_suite import grade
from eval.chat_suite import suite
from pathlib import Path
import subprocess
import sys
import torch

def main():
    identity=sha256(DATA/'processed/sft-train-tokens.jsonl')[:10]
    parent=RUNS/f'checkpoints/verification-{identity}-continuous/last.pt'
    config=RUNS/'reports/resume-config.json'
    out=RUNS/'checkpoints/verification-continue-v2'
    resumed=RUNS/'checkpoints/verification-continue-v2-resumed'
    common=[sys.executable,'-m','training.train','--config',str(config),'--steps','4','--minutes','5','--length','32','--micro-batch','2','--accumulation','2','--warmup-steps','2']
    if not (out/'last.pt').exists():
        subprocess.run(common+['--continue-from',str(parent),'--output',str(out)],cwd=ROOT,check=True)
    if not (resumed/'last.pt').exists():
        subprocess.run(common+['--resume',str(out/'step-000002.pt'),'--output',str(resumed)],cwd=ROOT,check=True)
    old,initial,a,b=[load_checkpoint(p) for p in (parent,out/'initial.pt',out/'last.pt',resumed/'last.pt')]
    assert all(torch.equal(old['model'][k],initial['model'][k]) for k in old['model'])
    assert initial['parent_tokens']==old['supervised_tokens'] and initial['step']==0
    assert all(torch.equal(a['model'][k],b['model'][k]) for k in a['model'])
    for i,state in old['optimizer']['state'].items():
        for k,v in state.items():
            if isinstance(v,torch.Tensor):
                assert torch.equal(v,initial['optimizer']['state'][i][k])
    for i,state in a['optimizer']['state'].items():
        for k,v in state.items():
            if isinstance(v,torch.Tensor):
                assert torch.equal(v,b['optimizer']['state'][i][k])
    assert a['cumulative_pretrain_tokens']==old['supervised_tokens']+a['supervised_tokens']
    assert torch.equal(a['data_rng'],b['data_rng'])
    raw=RUNS/'reports/verification-incomplete-answers.json'
    marks=RUNS/'reports/verification-incomplete-marks.json'
    write_json(raw,dict(split='test',results=[dict(suite('test')[0],seed=2026,answer='你好',eos=True)]))
    write_json(marks,dict(marks=[dict(id=suite('test')[0]['id'],seed=2026,relevant_readable=True,correct=True,looping=False,reason='验证评分格式')]))
    try:
        grade(raw,marks)
    except ValueError as exc:
        assert 'incomplete' in str(exc)
    else:
        raise AssertionError('incomplete evaluation incorrectly accepted')
    result=dict(continue_preserves_weights=True,continue_preserves_adam=True,continue_resets_stage_step=True,cumulative_tokens_correct=True,continue_resume_weights_optimizer_rng_exact=True,incomplete_acceptance_rejected=True,passed=True)
    write_json(RUNS/'reports/continue_checks.json',result)
    print(result)

if __name__=='__main__':
    main()
