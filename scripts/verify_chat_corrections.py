"""前缀、组批与预取消费游标检查；可选真实GPU磁盘恢复检查。"""
from scripts.common import DATA,RUNS,ROOT,read_json,write_json,utc
from sft.template import encode_pair,prompt,PREFIX_ENCODING
from training.batches import SFTPairs,PrefetchedSFT
from training.execution import validate_resume_settings
from data.prepare import save_jsonl
from tokenizers import Tokenizer
from pathlib import Path
import argparse
import tempfile
import torch
import copy
import subprocess
import sys

def verify_task_objective():
    from scripts.common import GPUBudget,require_gpu
    from model import LubanLM,LubanConfig
    from training.execution import update,configure_model
    from training.batches import TASK_WEIGHTS
    from torch.nn import functional as F
    with GPUBudget('sft-correction-objective-check',1):
        device=require_gpu()
        config=LubanConfig('task-loss-check','verification',64,32,2,4,2,8,64,128,True,False,0,1,2)
        torch.manual_seed(2026)
        model=LubanLM(config).to(device);configure_model(model,'sdpa')
        x=torch.tensor([[1,3,4,5],[1,6,7,0],[1,8,9,10],[1,11,12,0]])
        y=torch.tensor([[-100,4,5,2],[-100,7,2,-100],[-100,-100,10,2],[-100,12,2,-100]])
        tasks=torch.tensor([0,0,1,6]);mask=x!=0
        counts=torch.bincount(tasks,minlength=7)
        active=sum(w for w,n in zip(TASK_WEIGHTS,counts) if int(n)>0)
        weights=torch.tensor([TASK_WEIGHTS[int(t)]/int(counts[int(t)])/active for t in tasks],device=device)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits=model(x.to(device),attention_mask=mask.to(device))
            losses=F.cross_entropy(logits.float().flatten(0,1),y.to(device).flatten(),ignore_index=-100,reduction='none').reshape_as(y)
            reference=((losses.sum(1)/(y.to(device)!=-100).sum(1))*weights).sum()
        reference.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        gradients={n:p.grad.detach().clone() for n,p in model.named_parameters()}
        optimizer=torch.optim.AdamW(model.parameters(),lr=0,foreach=False,fused=True)
        scaler=torch.amp.GradScaler('cuda',enabled=False)
        actual,_,_,_=update(model,optimizer,scaler,[(x,y,mask,tasks)],device,torch.bfloat16,'optimized',objective='task-balanced')
        error=abs(float(actual)-float(reference))
        grad_error=max(float((p.grad-gradients[n]).abs().max()) for n,p in model.named_parameters())
        assert error<1e-6 and grad_error<1e-6
        output=dict(loss_absolute_error=error,clipped_gradient_max_absolute_error=grad_error,passed=True)
        write_json(ROOT/'docs/run-results/chat-v2/task_objective_check.json',output)
        print(output)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(4)
    tok=Tokenizer.from_file(str(DATA/'tokenizers/vocab-32000/tokenizer.json'))
    cases=['《岳阳楼记》是一篇文章。','“欢迎”两个字。','-12。','(一个括号)','1+2=3。','\n第一行。','你好，世界。']
    crossing=0
    for answer in cases:
        prefix=tok.encode(prompt('请给出一句话。'),add_special_tokens=False).ids
        new=encode_pair(tok,'请给出一句话。',answer,encoding=PREFIX_ENCODING)
        old=encode_pair(tok,'请给出一句话。',answer)
        crossing+=old['input_ids'][:len(prefix)]!=prefix
        assert new['input_ids'][:len(prefix)]==prefix
        assert new['labels'][:len(prefix)]==[-100]*len(prefix)
        assert new['labels'][len(prefix):]==new['input_ids'][len(prefix):]
        assert tok.decode(new['input_ids'][len(prefix):-1],skip_special_tokens=False)==answer
    assert crossing>0 and encode_pair(tok,'问题',' ',encoding=PREFIX_ENCODING) is None
    from transformers import AutoTokenizer
    hf=AutoTokenizer.from_pretrained(DATA/'tokenizers/vocab-32000',local_files_only=True)
    assert hf.apply_chat_template([{'role':'user','content':'请给出一句话。'}],tokenize=False,add_generation_prompt=True)==prompt('请给出一句话。')
    result=dict(created_utc=utc(),prefix_edge_cases=len(cases),legacy_crossing_cases=crossing,hf_native_prompt_equal=True)
    with tempfile.TemporaryDirectory(prefix='luban-corrections-') as directory:
        path=Path(directory)/'pairs.jsonl'
        rows=[dict(input_ids=[1,3+i,4,5,2]+[2]*(i%3),labels=[-100,-100,4,5,2]+[2]*(i%3)) for i in range(9)]
        save_jsonl(path,rows)
        def make(mode):return SFTPairs(path,0,torch.Generator().manual_seed(2026),batching=mode)
        legacy,flat=make('legacy'),make('flat')
        for _ in range(12):
            a,b=legacy.batch(4,legacy.generator),flat.batch(4,flat.generator)
            assert all(torch.equal(x,y) for x,y in zip(a,b))
            assert legacy.state()==flat.state() and torch.equal(legacy.generator.get_state(),flat.generator.get_state())
        consumed=make('flat');prefetch=PrefetchedSFT(consumed,4,2)
        reference=make('flat')
        for _ in range(7):
            a,b=reference.batch(4,reference.generator),prefetch.batch(4,consumed.generator)
            assert all(torch.equal(x,y) for x,y in zip(a,b))
        saved=copy.deepcopy(consumed.state());rng=consumed.generator.get_state().clone()
        expected=[reference.batch(4,reference.generator) for _ in range(4)]
        prefetch.close()
        restored=make('flat');restored.restore(saved);restored.generator.set_state(rng)
        restarted=PrefetchedSFT(restored,4,2)
        for batch in expected:
            actual=restarted.batch(4,restored.generator)
            assert all(torch.equal(x,y) for x,y in zip(batch,actual))
        restarted.close()
        assert restored.state()==reference.state() and torch.equal(restored.generator.get_state(),reference.generator.get_state())
        result.update(flat_matches_legacy=True,prefetch_consumed_state_and_epoch_resume_exact=True)
    if args.gpu:
        from training.train import load_checkpoint
        from scripts.benchmark_training import assert_tree_equal,functional_checks
        from scripts.common import GPUBudget,require_gpu
        with GPUBudget('sft-correction-functional-checks',2):
            result['functional']=functional_checks(require_gpu())
        folder=RUNS/'checkpoints/correction-resume-check'
        folder.mkdir(parents=True,exist_ok=True)
        config=read_json(ROOT/'configs/config_50m.json')
        config.update(name='correction-resume-check',vocab_size=64,hidden_size=64,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=16,intermediate_size=128)
        write_json(folder/'config.json',config)
        save_jsonl(folder/'pairs.jsonl',rows[:5])
        common=[sys.executable,'-m','training.train','--config',str(folder/'config.json'),'--stage','sft','--sft-train',str(folder/'pairs.jsonl'),'--sft-validation',str(folder/'pairs.jsonl'),'--steps','6','--epochs','5','--minutes','2','--micro-batch','4','--accumulation','1','--lr','1e-5','--warmup-steps','2','--execution','optimized','--attention-backend','sdpa','--optimizer-backend','fused','--sft-batching','flat','--prefetch-batches','2','--validation-micro-batch','2']
        a,b=folder/'continuous',folder/'resumed'
        subprocess.run(common+['--output',str(a)],cwd=ROOT,check=True)
        subprocess.run(common+['--output',str(b),'--resume',str(a/'step-000003.pt')],cwd=ROOT,check=True)
        first,second=load_checkpoint(a/'last.pt'),load_checkpoint(b/'last.pt')
        for key in ('model','optimizer','scaler','sampler','data_rng','torch_rng','cuda_rng','step','input_tokens','supervised_tokens','effective_input_tokens'):
            assert_tree_equal(first[key],second[key])
        result['gpu_disk_weights_optimizer_rng_sampler_exact']=True
    result['passed']=True
    write_json(ROOT/'docs/run-results/chat-v2/correction_checks.json',result)
    print(result)

if __name__=='__main__':main()
