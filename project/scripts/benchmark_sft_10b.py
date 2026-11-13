from scripts.common import ROOT,read_json,write_json,GPUBudget,require_gpu,sha256
from scripts.policy_versions import sft_identity
from training.train import load_checkpoint,validation
from training.batches import SFTPairs,PrefetchedSFT
from training.execution import BatchTransfer,update,configure_model,trim_cuda_cache
from model import LubanLM,LubanConfig
from pathlib import Path
import argparse
import time
import torch
import psutil

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--checkpoint',required=True);args=p.parse_args()
    plan=read_json(args.plan);torch.set_num_threads(4)
    rows=[];digest=sha256(args.checkpoint);process=psutil.Process()
    progress=Path(plan['runs_root'])/'reports/sft_benchmark_progress.json'
    with GPUBudget('sft-throughput-benchmark',10) as budget:
        device=require_gpu();state=load_checkpoint(args.checkpoint)
        model=LubanLM(LubanConfig.from_dict(state['config'])).to(device);configure_model(model,'sdpa')
        for size in (8,16):
            torch.cuda.empty_cache();model.load_state_dict(state['model']);model.train()
            optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5,fused=True)
            data=SFTPairs(Path(plan['data_root'])/'processed'/plan['sft'].get('data_version','sft-100k')/'train-tokens.jsonl',0,torch.Generator().manual_seed(2026),batching='bucketed')
            valid=SFTPairs(Path(plan['data_root'])/'processed'/plan['sft'].get('data_version','sft-100k')/'validation-tokens.jsonl',0,torch.Generator().manual_seed(9090),batching='flat')
            transfer=BatchTransfer(True);scaler=torch.amp.GradScaler('cuda',enabled=False)
            try:
                for _ in range(5):update(model,optimizer,scaler,transfer.prepare([data.batch(size,data.generator)]),device,torch.bfloat16,'optimized',True)
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.monotonic();tokens=0
                for index in range(100):
                    if index%5==0:
                        trim_cuda_cache(device)
                        rss=process.memory_info().rss;total=torch.cuda.get_device_properties(0).total_memory
                        write_json(progress,dict(batch=size,step=index,rss_bytes=rss,available_ram=psutil.virtual_memory().available,gpu_allocated=torch.cuda.memory_allocated(),gpu_reserved=torch.cuda.memory_reserved(),gpu_total=total))
                        if rss>plan.get('sft_rss_bytes_cap',3<<30):raise MemoryError('SFT benchmark RSS exceeded before update')
                        if torch.cuda.memory_reserved()>total*.80:raise MemoryError('SFT allocator reserve exceeded safe probe threshold after trim')
                    batch=data.batch(size,data.generator);tokens+=int(batch[2].sum())
                    update(model,optimizer,scaler,transfer.prepare([batch]),device,torch.bfloat16,'optimized',True)
                    if index%5==0 and budget.tick():raise RuntimeError('BUDGET_EXHAUSTED: SFT benchmark')
                torch.cuda.synchronize();seconds=time.monotonic()-start
                validation(model,valid,2,device,torch.bfloat16,batches=8,details=True)
                rss=process.memory_info().rss
                if rss>plan.get('sft_rss_bytes_cap',3<<30) or psutil.virtual_memory().available<plan.get('memory_free_bytes_floor',1<<30):raise MemoryError('SFT training reader/model/validation resource gate failed')
                peak=torch.cuda.max_memory_reserved();total=torch.cuda.get_device_properties(0).total_memory
                rows.append(dict(batch=size,status='ok' if total-peak>=1<<30 else 'insufficient_headroom',effective_tokens_per_second=tokens/seconds,peak_reserved=peak,process_rss_bytes=rss,windows_peak_working_set=process.memory_info().peak_wset))
            except torch.cuda.OutOfMemoryError:rows.append(dict(batch=size,status='oom'))
            data.rows.close();valid.rows.close();del optimizer,data,valid;model.zero_grad(set_to_none=True);budget.tick()
        candidates=[r for r in rows if r['status']=='ok']
        if not candidates:raise RuntimeError('No safe SFT batch')
        best=max(candidates,key=lambda r:r['effective_tokens_per_second'])
        small=next((r for r in candidates if r['batch']==8),None)
        if small and best['effective_tokens_per_second']<=small['effective_tokens_per_second']*1.03:best=small
        if sha256(args.checkpoint)!=digest:raise RuntimeError('Benchmark changed checkpoint')
        folder=Path(plan['data_root'])/'processed'/plan['sft']['data_version']
        write_json(Path(plan['runs_root'])/'reports/sft_benchmark.json',dict(rows=rows,selected_batch=best['batch'],parent_unchanged=True,checkpoint_sha256=digest,execution_sha256=sha256(ROOT/'training/execution.py'),policy_identity=sft_identity(args.plan),data_sha256={s:sha256(folder/f'{s}-tokens.jsonl') for s in ('train','validation')}))

if __name__=='__main__':main()
