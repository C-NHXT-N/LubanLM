"""同一批1024条语义样本的组批/预取/分桶对照，限额10分钟，不改正式权重。"""
from scripts.common import DATA,RUNS,ROOT,read_json,write_json,sha256,GPUBudget,require_gpu,utc
from training.train import load_checkpoint
from training.batches import SFTPairs,PrefetchedSFT
from training.execution import BatchTransfer,update,configure_model
from scripts.benchmark_training import Resources
from data.prepare import load_jsonl,save_jsonl
from model import LubanLM,LubanConfig
from pathlib import Path
import argparse
import statistics
import time
import torch

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    args=parser.parse_args()
    if (RUNS/'reports/gpu_budget.lock').exists():raise RuntimeError('another GPU task is running')
    torch.set_num_threads(4)
    source=DATA/'processed/sft-v6/train-tokens.jsonl'
    all_rows=load_jsonl(source)
    chosen=torch.randperm(len(all_rows),generator=torch.Generator().manual_seed(7126))[:1024].tolist()
    path=RUNS/'reports/sft-correction-profile-subset.jsonl'
    save_jsonl(path,[all_rows[i] for i in chosen]);del all_rows
    result=dict(created_utc=utc(),source_sha256=sha256(source),subset_sha256=sha256(path),checkpoint_sha256=sha256(args.checkpoint),profiles=[])
    with GPUBudget('sft-correction-performance-checks',10) as budget:
        state=load_checkpoint(args.checkpoint);device=require_gpu()
        model=LubanLM(LubanConfig.from_dict(state['config'])).to(device)
        configure_model(model,'sdpa')
        def setup(mode,prefetch_enabled):
            model.load_state_dict(state['model']);model.train()
            optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,betas=(.9,.95),weight_decay=.1,foreach=False,fused=True)
            scaler=torch.amp.GradScaler('cuda',enabled=False)
            gen=torch.Generator().manual_seed(2026)
            data=SFTPairs(path,0,gen,batching=mode)
            feed=PrefetchedSFT(data,8,2) if prefetch_enabled else data
            return optimizer,scaler,gen,data,feed,BatchTransfer(True)
        for mode,prefetch_enabled in [('legacy',False),('flat',True),('bucketed',True)]:
            row=dict(batching=mode,prefetch=prefetch_enabled,repeats=[])
            for repeat in range(3):
                if budget.remaining()<60:raise RuntimeError('profile budget insufficient')
                opt,scaler,gen,data,feed,transfer=setup(mode,prefetch_enabled)
                for _ in range(10):update(model,opt,scaler,transfer.prepare([feed.batch(8,gen)]),device,torch.bfloat16,'optimized',True)
                if prefetch_enabled:feed.close()
                # 预热不改变计时样本；每次覆盖完全相同的1024条数据。
                gen=torch.Generator().manual_seed(2026);data=SFTPairs(path,0,gen,batching=mode)
                feed=PrefetchedSFT(data,8,2) if prefetch_enabled else data
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                valid=targets=padded=0;prepare_seconds=0.
                with Resources() as resources:
                    start=time.perf_counter()
                    for _ in range(128):
                        t=time.perf_counter();batch=feed.batch(8,gen);ready=transfer.prepare([batch]);prepare_seconds+=time.perf_counter()-t
                        valid+=int(batch[2].sum());padded+=batch[0].numel()
                        _,_,_,n=update(model,opt,scaler,ready,device,torch.bfloat16,'optimized',True);targets+=n
                        budget.tick()
                    torch.cuda.synchronize();seconds=time.perf_counter()-start
                if prefetch_enabled:feed.close()
                row['repeats'].append(dict(seconds=seconds,effective_token_s=valid/seconds,supervised_token_s=targets/seconds,padding_fraction=1-valid/padded,prepare_seconds=prepare_seconds,resources=resources.summary(),valid_inputs=valid,targets=targets))
                del opt,scaler;torch.cuda.empty_cache()
            row['median_effective_token_s']=statistics.median(r['effective_token_s'] for r in row['repeats'])
            result['profiles'].append(row)
            print(mode,row['median_effective_token_s'],flush=True)
        # profiler仅定位，不计为正常吞吐。
        opt,scaler,gen,data,feed,transfer=setup('flat',False)
        try:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                for _ in range(3):update(model,opt,scaler,transfer.prepare([feed.batch(8,gen)]),device,torch.bfloat16,'optimized',True)
                torch.cuda.synchronize()
            text=prof.key_averages().table(sort_by='self_cpu_time_total',row_limit=25)
            (RUNS/'reports/sft-correction-profiler.txt').write_text(text,encoding='utf-8')
        except RuntimeError as exc:result['profiler_error']=str(exc)
        reference=result['profiles'][0]['median_effective_token_s']
        for row in result['profiles']:row['speedup']=row['median_effective_token_s']/reference
        # 分桶改变样本顺序，只能用于新阶段；预取相同序列的收益单独判断。
        result.update(flat_prefetch_adopted=result['profiles'][1]['speedup']>=1.15,
                      bucketed_adoptable_new_stage=result['profiles'][2]['speedup']>=1.15,
                      passed=True,completed_utc=utc())
        write_json(RUNS/'reports/sft-correction-performance.json',result)
        write_json(ROOT/'docs/run-results/chat-v2/sft-correction-performance.json',result)
        print(result)

if __name__=='__main__':main()
