"""同一检查点、同一采样状态下的限时性能/数值对照，不改正式权重。"""
from scripts.common import DATA, RUNS, ROOT, GPUBudget, require_gpu, write_json, read_json, sha256, utc
from training.train import load_checkpoint, schedule
from training.execution import configure_model, configure_optimizer, BatchTransfer, update, loss_for, validate_resume_settings
from training.batches import PackedTokens, SFTPairs
from model import LubanLM, LubanConfig
import argparse
import copy
import statistics
import subprocess
import threading
import time
from pathlib import Path
import psutil
import torch
from torch.nn import functional as F

def assert_tree_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a.cpu(), b.cpu())
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a: assert_tree_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b): assert_tree_equal(x, y)
    else:
        assert a == b

def functional_checks(device):
    config = LubanConfig('perf-check', 'verification', 64, 32, 2, 4, 2, 8, 64, 128, True, False, 0, 1, 2)
    torch.manual_seed(2026)
    model = LubanLM(config).to(device).eval()
    ids = torch.tensor([[3,4,5,6,7]], device=device)
    results = {}
    for backend in ('manual', 'sdpa'):
        configure_model(model, backend)
        with torch.no_grad():
            full = model(ids)
            changed = ids.clone(); changed[:,3:] = 9
            causal = float((full[:,:3] - model(changed)[:,:3]).abs().max())
            first = model(ids[:,:3], use_cache=True)
            rest = model(ids[:,3:], past_key_values=first.past_key_values, use_cache=True)
            cache = float((full[:,3:] - rest.logits).abs().max())
            padded = torch.tensor([[3,4,5,0,0]], device=device)
            padding = float((model(ids[:,:3]) - model(padded,attention_mask=padded != 0)[:,:3]).abs().max())
            assert causal == 0 and cache < 2e-6 and padding < 2e-6
        results[backend] = dict(causal_max_error=causal,cache_max_error=cache,padding_max_error=padding)
    configure_model(model, 'sdpa'); model.train()
    optimizer = torch.optim.AdamW(model.parameters(),lr=.001,foreach=False,fused=True)
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    batches = [(ids[:,:-1].cpu(),ids[:,1:].cpu(),None)]
    # 包括不等长/ignore 标签的 SFT 权重检查。
    x = torch.tensor([[3,4,5,6],[7,8,0,0]])
    y = torch.tensor([[-100,5,6,2],[-100,2,-100,-100]])
    mask = x != 0
    model.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        logits=model(x.to(device),attention_mask=mask.to(device))
        reference=F.cross_entropy(logits.float().flatten(0,1),y.to(device).flatten())
    reference.backward(); grads={n:p.grad.detach().clone() for n,p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    terms=[]
    for i in range(2):
        loss,_,count=loss_for(model,(x[i:i+1],y[i:i+1],mask[i:i+1]),device,torch.bfloat16)
        (loss*count/4).backward();terms.append(float(loss)*count/4)
    weighted_error=abs(sum(terms)-float(reference))
    gradient_error=sum(float((p.grad-grads[n]).square().sum()) for n,p in model.named_parameters())**.5
    assert weighted_error < .01 and gradient_error < .03
    # 相同设置下：权重、Adam、CPU/GPU RNG 的快照恢复和连续更新完全一致。
    update(model,optimizer,scaler,batches,device,torch.bfloat16,'optimized')
    saved_model=copy.deepcopy(model.state_dict()); saved_opt=copy.deepcopy(optimizer.state_dict())
    cpu_rng=torch.get_rng_state(); gpu_rng=torch.cuda.get_rng_state_all()
    update(model,optimizer,scaler,batches,device,torch.bfloat16,'optimized')
    continuous_model=copy.deepcopy(model.state_dict()); continuous_opt=copy.deepcopy(optimizer.state_dict())
    model.load_state_dict(saved_model);optimizer.load_state_dict(saved_opt)
    torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state_all(gpu_rng)
    update(model,optimizer,scaler,batches,device,torch.bfloat16,'optimized')
    assert_tree_equal(model.state_dict(),continuous_model);assert_tree_equal(optimizer.state_dict(),continuous_opt)
    # 非有限损失不能触发参数更新。
    before=copy.deepcopy(model.state_dict())
    handle=model.lm_head.register_forward_hook(lambda m,a,out:out*float('nan'))
    try:
        try:update(model,optimizer,scaler,batches,device,torch.bfloat16,'optimized')
        except RuntimeError:pass
        else:raise AssertionError('nonfinite update accepted')
    finally:handle.remove()
    assert_tree_equal(before,model.state_dict())
    results.update(weighted_sft_loss_error=weighted_error,weighted_sft_gradient_l2_error=gradient_error,optimized_resume_exact=True,nonfinite_update_rejected=True)
    return results

class Resources:
    def __init__(self):
        self.rows=[];self.stop=threading.Event();self.thread=threading.Thread(target=self.sample,daemon=True)
    def sample(self):
        process=psutil.Process();process.cpu_percent(None)
        while not self.stop.is_set():
            try:
                raw=subprocess.check_output(['nvidia-smi','-i','GPU-c50d3c54-c5d3-f0da-7fa4-a48d195a849f','--query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu','--format=csv,noheader,nounits'],text=True)
                values=[float(x.strip()) for x in raw.strip().split(',')]
                self.rows.append(dict(gpu_percent=values[0],card_mib=values[1],power_w=values[2],temperature_c=values[3],cpu_one_core_percent=process.cpu_percent(None),rss_mib=process.memory_info().rss/2**20))
            except (ValueError,subprocess.SubprocessError,psutil.Error):pass
            self.stop.wait(.5)
    def __enter__(self):self.thread.start();return self
    def __exit__(self,*args):self.stop.set();self.thread.join(2)
    def summary(self):
        return {k:statistics.mean(r[k] for r in self.rows) for k in self.rows[0]} if self.rows else {}

def choose_profile(rows, baseline):
    valid=[row for row in rows if row and row.get('passed')]
    if not valid:return baseline
    fastest=max(row['median_tokens_per_second'] for row in valid)
    close=[row for row in valid if row['median_tokens_per_second']>=fastest/1.05]
    reference=baseline['profile']
    best=min(close,key=lambda row:(sum(row['profile'][key]!=reference[key] for key in reference),row['profile']['micro_batch']))
    return best if best['median_tokens_per_second']>=baseline['median_tokens_per_second']*1.1 else baseline

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True,type=Path)
    parser.add_argument('--minutes',type=float,default=20)
    args=parser.parse_args()
    if not 0 < args.minutes <= 20:parser.error('performance budget must be <=20 minutes')
    device=require_gpu();torch.set_num_threads(4)
    report=RUNS/'reports/performance_comparison.json'
    profile_path=RUNS/'reports/execution_profile.json'
    state=load_checkpoint(args.checkpoint)
    digest=sha256(args.checkpoint)
    saved_settings=state['settings']
    assert saved_settings['stage']=='pretrain' and saved_settings['micro_batch']*saved_settings['accumulation']==8
    train=PackedTokens(DATA/'processed/pretrain-train',saved_settings['length'])
    assert state['dataset']==train.identity
    def batches(generator,micro):
        return [train.batch(micro,generator) for _ in range(8//micro)]
    # 分组变化不改变采样顺序，也不改变 optimizer 更新对应的 8 条样本。
    samples=[];rngs=[]
    for micro in (2,4,8):
        gen=torch.Generator();gen.set_state(state['data_rng'])
        rows=batches(gen,micro);samples.append(torch.cat([b[0] for b in rows]));rngs.append(gen.get_state())
    assert all(torch.equal(samples[0],s) for s in samples) and all(torch.equal(rngs[0],s) for s in rngs)
    result=dict(started_utc=utc(),checkpoint=str(args.checkpoint),checkpoint_sha256=digest,checkpoint_step=state['step'],checkpoint_tokens=state['supervised_tokens'],effective_batch=8,functional_checks=None,tests=[],sampling_sequence_equal=True,selected=None,passed=False)
    write_json(report,result)
    base=dict(execution='legacy',attention_backend='manual',optimizer_backend='loop',pin_memory=False,micro_batch=2,accumulation=4)
    with GPUBudget('pretrain-performance-checks',args.minutes) as budget:
        result['functional_checks']=functional_checks(device);budget.tick(force=True)
        model=LubanLM(LubanConfig.from_dict(state['config'])).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=saved_settings['lr'],weight_decay=.1,betas=(.9,.95),foreach=False)
        scaler=torch.amp.GradScaler('cuda',enabled=False)
        rate=schedule(state['step'],__import__('math').ceil(saved_settings['tokens']/(8*saved_settings['length'])),saved_settings['lr'],saved_settings.get('warmup_steps'))
        sft_rows=None
        sft_initial=None
        def reset(profile):
            model.load_state_dict(state['model']);optimizer.load_state_dict(copy.deepcopy(state['optimizer']))
            configure_model(model,profile['attention_backend']);configure_optimizer(optimizer,profile['optimizer_backend'])
            for group in optimizer.param_groups:group['lr']=rate
            model.train();generator=torch.Generator();generator.set_state(state['data_rng'])
            torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
            if sft_rows is not None:
                train.generator=generator
                train.restore(copy.deepcopy(sft_initial))
            return generator
        def gradients(profile):
            gen=reset(profile);rows=batches(gen,profile['micro_batch']);model.zero_grad(set_to_none=True)
            weighted=0.
            targets=sum(int((b[1]!=-100).sum()) for b in rows)
            for batch in rows:
                loss,_,count=loss_for(model,batch,device,torch.bfloat16)
                (loss*count/targets).backward();weighted+=float(loss)*count/targets
            return weighted,{n:p.grad.detach().cpu().clone() for n,p in model.named_parameters()}
        ref_loss,ref_grad=gradients(base)
        def trial(name,profile):
            if budget.remaining()<50:return None
            print('TRIAL_STARTED',name,utc(),flush=True)
            row=dict(name=name,profile=profile,passed=False,repeats=[])
            try:
                candidate_settings={**saved_settings,**profile,'validation_micro_batch':saved_settings['micro_batch']}
                validate_resume_settings(saved_settings,candidate_settings,True)
                loss,grad=gradients(profile)
                numerator=sum(float((grad[n]-g).square().sum()) for n,g in ref_grad.items())
                denominator=sum(float(g.square().sum()) for g in ref_grad.values())
                row.update(loss_abs_error=abs(loss-ref_loss),gradient_relative_l2=(numerator/max(denominator,1e-30))**.5)
                del grad
                if row['loss_abs_error']>.01 or row['gradient_relative_l2']>.02:
                    row['rejection']='numerical tolerance exceeded'
                    return row
                for repeat in range(3):
                    if budget.remaining()<45:break
                    gen=reset(profile);transfer=BatchTransfer(profile['pin_memory'])
                    for _ in range(20):
                        update(model,optimizer,scaler,transfer.prepare(batches(gen,profile['micro_batch'])),device,torch.bfloat16,profile['execution'],profile['pin_memory'])
                        budget.tick(force=profile['execution']=='legacy')
                    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                    prepare_seconds=0.
                    measured_inputs=measured_targets=0
                    with Resources() as resources:
                        start=time.perf_counter()
                        for _ in range(100):
                            t=time.perf_counter();batch=transfer.prepare(batches(gen,profile['micro_batch']));prepare_seconds+=time.perf_counter()-t
                            _,_,inputs,targets=update(model,optimizer,scaler,batch,device,torch.bfloat16,profile['execution'],profile['pin_memory'])
                            measured_inputs+=sum(int(b[2].sum()) if b[2] is not None else b[0].numel() for b in batch)
                            measured_targets+=targets
                            budget.tick(force=profile['execution']=='legacy')
                            if budget.remaining()<10:raise TimeoutError('performance budget nearly exhausted')
                        torch.cuda.synchronize();seconds=time.perf_counter()-start
                    sample=resources.summary()
                    sample['card_peak_mib']=max((r['card_mib'] for r in resources.rows),default=0)
                    row['repeats'].append(dict(seconds=seconds,tokens_per_second=measured_inputs/seconds,supervised_tokens_per_second=measured_targets/seconds,data_prepare_percent=100*prepare_seconds/seconds,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,resources=sample))
                    if sample.get('card_peak_mib',0)>=12288:raise RuntimeError('card memory limit exceeded')
                if len(row['repeats'])==3:
                    row['median_tokens_per_second']=statistics.median(r['tokens_per_second'] for r in row['repeats']);row['passed']=True
            except (RuntimeError,TimeoutError) as exc:
                row['rejection']=str(exc);optimizer.zero_grad(set_to_none=True);torch.cuda.empty_cache()
            finally:
                result['tests'].append(row);write_json(report,result)
                print('TRIAL_COMPLETED',name,row.get('median_tokens_per_second'),row.get('rejection'),flush=True)
            return row
        baseline=trial('baseline-2x4',base)
        if baseline is None or not baseline['passed']:raise RuntimeError('baseline measurement incomplete')
        variants=[]
        for micro in (2,4,8):
            item=trial(f'optimized-{micro}x{8//micro}',{**base,'execution':'optimized','micro_batch':micro,'accumulation':8//micro})
            if item and item['passed']:variants.append(item)
        best=choose_profile(variants,baseline)
        fused=trial('fused-adamw',{**best['profile'],'optimizer_backend':'fused'})
        if fused and fused['passed'] and fused['median_tokens_per_second']>best['median_tokens_per_second']*1.05:best=fused
        sdpa=trial('sdpa',{**best['profile'],'attention_backend':'sdpa'})
        if sdpa and sdpa['passed'] and sdpa['median_tokens_per_second']>best['median_tokens_per_second']*1.05:best=sdpa
        pinned=trial('pinned-transfer',{**best['profile'],'pin_memory':True})
        if pinned and pinned['passed'] and pinned['median_tokens_per_second']>best['median_tokens_per_second']*1.05:best=pinned
        if best['median_tokens_per_second']<baseline['median_tokens_per_second']*1.1:best=baseline
        # SFT 变长 padding 与 pretrain 不同，不能直接套用预训练微批次。
        sft_best=None
        if budget.remaining()>180:
            original_train=train
            sft_path=DATA/'processed/sft-v3/train-tokens.jsonl'
            if not sft_path.exists():sft_path=DATA/'processed/sft-train-tokens.jsonl'
            train=SFTPairs(sft_path,state['config']['pad_token_id'],torch.Generator().manual_seed(2026))
            sft_rows=train.rows;sft_initial=copy.deepcopy(train.state())
            ref_loss,ref_grad=gradients(base)
            sft_baseline=trial('sft-baseline-2x4',base)
            candidates=[]
            for micro in (2,4,8):
                item=trial(f'sft-optimized-{micro}x{8//micro}',{**best['profile'],'execution':'optimized','micro_batch':micro,'accumulation':8//micro})
                if item and item['passed']:candidates.append(item)
            if sft_baseline and sft_baseline['passed']:
                sft_best=choose_profile(candidates,sft_baseline)
                result['sft_selected']=sft_best['name']
                result['sft_speedup']=sft_best['median_tokens_per_second']/sft_baseline['median_tokens_per_second']
            train=original_train;sft_rows=None
        # CPU/CUDA profiler 不用于选参，只记录运算后端和同步证据。
        gen=reset(best['profile'])
        try:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                update(model,optimizer,scaler,batches(gen,best['profile']['micro_batch']),device,torch.bfloat16,best['profile']['execution'])
                torch.cuda.synchronize()
            keys=[e.key for e in prof.key_averages()]
            result['attention_operators']=[key for key in keys if 'attention' in key or 'softmax' in key]
            (RUNS/'reports/performance_profiler.txt').write_text(prof.key_averages().table(sort_by='self_cpu_time_total',row_limit=25),encoding='utf-8')
        except RuntimeError as exc:result['profiler_error']=str(exc)
        result.update(selected=best['name'],speedup=best['median_tokens_per_second']/baseline['median_tokens_per_second'],completed_utc=utc(),passed=True)
        write_json(report,result)
        write_json(profile_path,dict(pretrain=best['profile'],sft=sft_best['profile'] if sft_best else base,validation_micro_batch=2,benchmark_report=str(report),checkpoint=str(args.checkpoint),checkpoint_sha256=digest,reference_validation_loss=read_json(args.checkpoint.parent/'validation_latest.json')['validation_loss'],speedup=result['speedup'],migration_required=best['profile']!=base,rollback_used=False,created_utc=utc()))
        print('SELECTED',best['name'],'SPEEDUP',result['speedup'],flush=True)

if __name__=='__main__':main()
