"""Bounded unattended prepare -> audit -> full-pass Base -> SFT -> evaluation."""
from scripts.common import ROOT,read_json,write_json,sha256,utc
from scripts.run_general_2b import verify_manifest
from scripts.policy_versions import pretrain_identity,sft_identity
from scripts.content_review_gate import verify_content_review
from pathlib import Path
import argparse
import json
import math
import os
import subprocess
import sys
import time
import psutil

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,default=ROOT/'configs/50m-general-pretrain-10b.json')
    args=p.parse_args();args.plan=args.plan.resolve();plan=read_json(args.plan)
    data,runs=Path(plan['data_root']),Path(plan['runs_root']);runs.mkdir(parents=True,exist_ok=True)
    (data/'tmp').mkdir(parents=True,exist_ok=True)
    lock=runs/'controller.lock'
    with lock.open('x',encoding='utf-8') as stream:stream.write(str(os.getpid()))
    env=os.environ.copy();env.update(LUBAN_DATA=str(data),LUBAN_RUNS=str(runs),LUBAN_GPU_BUDGET_MINUTES=str(plan['fresh_gpu_budget_minutes']),PYTHONIOENCODING='utf-8',HF_ENDPOINT='https://hf-mirror.com',TOKENIZERS_PARALLELISM='true',RAYON_NUM_THREADS='4')
    env.update(TEMP=str(data/'tmp'),TMP=str(data/'tmp'),TORCH_HOME=str(data/'cache/torch'),HF_HOME=str(data/'cache/huggingface'))
    state=dict(pid=os.getpid(),process_created=psutil.Process().create_time(),plan=str(args.plan),plan_sha256=sha256(args.plan),started_utc=utc(),capability_status='not_accepted',additional_target=plan['additional_tokens'],cumulative_target=plan['target_cumulative_tokens'],expected_pretrain_hours=round(plan['additional_tokens']/plan['expected_gpu_tokens_per_hour'],2))
    def status(value,**extra):
        state.update(status=value,updated_utc=utc(),**extra);write_json(runs/'reports/controller.json',state)
        print(json.dumps(state,ensure_ascii=False),flush=True)
    def run(module,arguments,phase):
        status(phase)
        process=subprocess.Popen([sys.executable,'-B','-u','-m',module,*map(str,arguments)],cwd=ROOT,env=env)
        state['child_pid']=process.pid;write_json(runs/'reports/controller.json',state)
        state['child_created']=psutil.Process(process.pid).create_time()
        cpu_path=runs/'reports/cpu_preparation_budget.json'
        cpu_entries=read_json(cpu_path) if cpu_path.exists() else []
        preparing=phase in ('preparing_10b_cpu','preparing_sft_cpu')
        started=time.monotonic()
        if preparing:cpu_entries.append(dict(phase=phase,started_utc=utc(),seconds=0,finished=False))
        while process.poll() is None:
            try:rss=psutil.Process(process.pid).memory_info().rss
            except psutil.NoSuchProcess:break
            state.update(updated_utc=utc(),child_rss_bytes=rss)
            if preparing:
                cpu_entries[-1]['seconds']=time.monotonic()-started;write_json(cpu_path,cpu_entries)
                if sum(r['seconds'] for r in cpu_entries)>=plan['preparation_hours_cap']*3600:
                    process.terminate();process.wait();raise RuntimeError('PREPARATION_BUDGET_EXHAUSTED')
            write_json(runs/'reports/controller.json',state);time.sleep(10)
        code=process.wait();state.pop('child_pid',None);state.pop('child_created',None)
        if preparing:
            cpu_entries[-1].update(seconds=time.monotonic()-started,finished=True);write_json(cpu_path,cpu_entries)
        if code:raise RuntimeError(f'{phase} failed with exit code {code}')
    def used():
        path=runs/'reports/gpu_budget.json'
        return sum(r['seconds'] for r in read_json(path))/60 if path.exists() else 0
    def remaining_stage(kind,cap):
        path=runs/'reports/gpu_budget.json'
        spent=sum(r['seconds'] for r in read_json(path) if r['stage'].startswith(kind+'-'))/60 if path.exists() else 0
        value=min(cap-spent,plan['fresh_gpu_budget_minutes']-used())
        if value<=0:raise RuntimeError('BUDGET_EXHAUSTED: '+kind)
        return value
    awake=None
    try:
        awake=subprocess.Popen([sys.executable,'-B','-m','scripts.keep_awake','--pid',str(os.getpid())],cwd=ROOT,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
        old_ledgers=[Path('D:/LubanLM-Runs/chat-v2/reports/gpu_budget.json'),Path('D:/LubanLM-Runs/general-2b-v1/reports/gpu_budget.json')]
        state['historical_ledger_hashes']={str(p):sha256(p) for p in old_ledgers}
        state['historical_gpu_minutes']=sum(sum(r['seconds'] for r in read_json(p))/60 for p in old_ledgers)
        import torch
        parent=torch.load(plan['parent_checkpoint'],map_location='cpu',weights_only=False)
        if parent['settings']['stage']!='pretrain' or parent['cumulative_pretrain_tokens']!=plan['parent_tokens'] or parent['tokenizer_sha256']!=plan['tokenizer_sha256']:raise ValueError('Parent lineage mismatch')
        state['parent_sha256']=sha256(plan['parent_checkpoint']);del parent
        # Base and SFT gates are independent. A SFT shortfall never grants
        # permission to train unreviewed data, nor silently cancels Base.
        run('scripts.prepare_general_10b',['--plan',args.plan],'preparing_10b_cpu')
        sft_folder=data/'processed'/plan['sft'].get('data_version','sft-100k')
        manifest=sft_folder/'manifest.json'
        # CPU preparation is serialized before training and sample inspection.
        if not manifest.exists():run('scripts.prepare_sft_100k',['--plan',args.plan],'preparing_sft_cpu')
        status('verifying_all_training_shards')
        verified={s:verify_manifest(data/f'processed/pretrain-{s}/manifest.json',plan['tokenizer_sha256'],min_shard_tokens=1) for s in ('train','validation')}
        prepared=read_json(runs/'reports/preparation_complete.json')
        if prepared.get('policy_identity')!=pretrain_identity(args.plan) or verified['train']['tokens']<plan['additional_tokens']+1:raise RuntimeError('10B data audit failed')
        if not (runs/'reports/pretrain_content_review.json').exists():
            status('awaiting_pretrain_content_inspection',samples=str(runs/'reports/pretrain-content-samples.json'));return
        review=verify_content_review(args.plan,'pretrain')
        write_json(runs/'reports/pretraining_data_audit.json',dict(created_utc=utc(),manifests=verified,content_review=review,physical_download_audit=str(runs/'reports/network.jsonl'),parent_overlap=prepared['parent_overlap']))
        if not (runs/'reports/teaching-throughput-v3.json').exists():
            run('scripts.benchmark_general_pretrain',['--plan',args.plan,'--batches',24,32,'--steps',100,'--report-name','teaching-throughput-v3.json'],'benchmarking_pretrain')
        bench=read_json(runs/'reports/teaching-throughput-v3.json')
        measured=next(r['tokens_per_hour'] for r in bench['rows'] if r['micro_batch']==24 and r['status']=='completed')
        write_json(runs/'reports/time_estimate.json',dict(created_utc=utc(),measured_tokens_per_hour=measured,additional_tokens=plan['additional_tokens'],estimated_pretrain_hours=plan['additional_tokens']/measured,preparation_timing_separate=True,sft_hours='pending eligible data and benchmark',includes_full_validation=False))
        base=runs/'checkpoints/50m-general-base-12b';t=plan['training']
        base_summary=base/'summary.json'
        if not base_summary.exists() or read_json(base_summary)['supervised_tokens']<prepared['training_target_tokens']:
            remaining=remaining_stage('pretrain',plan['pretrain_minutes_cap'])
            options=['--config',ROOT/plan['model_config'],'--stage','pretrain','--output',base,'--minutes',remaining,'--tokens',prepared['training_target_tokens'],'--execution','optimized','--attention-backend','sdpa','--optimizer-backend','fused','--pin-memory','--validation-micro-batch',2,'--pretrain-sampling','full-pass','--sampler-seed',plan['seed'],'--prefetch-batches',2,'--keep-step-checkpoints',2,'--checkpoint-bytes-cap',6<<30,'--disk-free-bytes-floor',plan['disk_free_bytes_floor']]
            for k,v in t.items():options+=['--'+k.replace('_','-'),v]
            existing=[p for p in (base/'last.pt',base/'latest.pt',base/'initial.pt') if p.exists()]
            options+=['--resume',max(existing,key=lambda p:p.stat().st_mtime)] if existing else ['--continue-from',plan['parent_checkpoint']]
            write_json(runs/'reports/pretrain_invocation.json',dict(arguments=list(map(str,options)),target_tokens=prepared['training_target_tokens']))
            run('training.train',options,'training_pretrain')
        base_result=read_json(base_summary)
        if base_result['supervised_tokens']<prepared['training_target_tokens']:raise RuntimeError('Base stopped before full corpus coverage; SFT not started')
        # The immutable 2B origin is retained outside this stage. Drop duplicate
        # named recovery copies only after Base has completed, keeping last/best.
        for name in ('initial.pt','latest.pt'):
            duplicate=base/name
            if duplicate.exists():duplicate.unlink()
        chat=runs/'checkpoints/50m-general-chat-12b';chat_summary=chat/'summary.json'
        if not manifest.exists():
            if (runs/'reports/sft_shortfall.json').exists():status('base_complete_sft_data_shortfall',report=str(runs/'reports/sft_shortfall.json'));return
            run('scripts.prepare_sft_100k',['--plan',args.plan],'preparing_sft_cpu')
        sft=read_json(manifest)
        if sft.get('policy_identity')!=sft_identity(args.plan) or sft.get('template_sha256')!=sha256(ROOT/'sft/template.py'):raise RuntimeError('SFT policy/template changed')
        if sft['examples']['train']<plan['sft']['train_minimum']:raise RuntimeError('SFT count gate failed')
        for split in ('train','validation','test'):
            if sha256(sft_folder/f'{split}-tokens.jsonl')!=sft[split+'_sha256']:raise RuntimeError('SFT content changed')
        if not (runs/'reports/sft_content_review.json').exists():status('awaiting_sft_content_inspection');return
        verify_content_review(args.plan,'sft')
        if not chat_summary.exists() or read_json(chat_summary)['step']<read_json(chat_summary)['planned_steps']:
            # Probe on private model copies, choose only a measured safe batch.
            if not (runs/'reports/sft_benchmark.json').exists():run('scripts.benchmark_sft_10b',['--plan',args.plan,'--checkpoint',base/'last.pt'],'benchmarking_sft')
            batch=read_json(runs/'reports/sft_benchmark.json')['selected_batch']
            options=['--config',ROOT/plan['model_config'],'--stage','sft','--output',chat,'--minutes',remaining_stage('sft',plan['sft_minutes_cap']),'--epochs',plan['sft']['epochs'],'--lr',plan['sft']['lr'],'--micro-batch',batch,'--accumulation',1,'--execution','optimized','--attention-backend','sdpa','--optimizer-backend','fused','--pin-memory','--validation-micro-batch',2,'--sft-train',sft_folder/'train-tokens.jsonl','--sft-validation',sft_folder/'validation-tokens.jsonl','--sft-batching','bucketed','--prefetch-batches',2,'--full-sft-validation','--source-validation','--validation-every-steps',math.ceil(sft['examples']['train']/batch),'--save-every-minutes',30,'--log-every',100,'--keep-step-checkpoints',2,'--checkpoint-bytes-cap',6<<30,'--disk-free-bytes-floor',plan['disk_free_bytes_floor']]
            existing=[p for p in (chat/'last.pt',chat/'latest.pt',chat/'initial.pt') if p.exists()]
            options+=['--resume',max(existing,key=lambda p:p.stat().st_mtime)] if existing else ['--init',base/'last.pt']
            run('training.train',options,'training_sft')
        result=read_json(chat_summary)
        if result['step']<result['planned_steps']:raise RuntimeError('SFT stopped before planned epochs')
        selected=chat/'best.pt'
        if not selected.exists():selected=chat/'last.pt'
        run('scripts.evaluate_general_10b',['--checkpoint',plan['parent_checkpoint'],'--output',runs/'reports/parent_chat_responses.json'],'evaluating_original_base')
        run('scripts.evaluate_general_10b',['--checkpoint',selected,'--output',runs/'reports/final_chat_responses.json'],'evaluating_final_chat')
        run('export.prepare',['--checkpoint',base/'last.pt','--output',runs/'export/base'],'exporting_base')
        run('export.prepare',['--checkpoint',selected,'--output',runs/'export/chat'],'exporting_chat')
        if sha256(plan['parent_checkpoint'])!=state['parent_sha256']:raise RuntimeError('Parent checkpoint changed')
        for path,digest in state['historical_ledger_hashes'].items():
            if sha256(path)!=digest:raise RuntimeError('Historical ledger changed')
        status('completed_pending_semantic_review',base_checkpoint=str(base/'last.pt'),chat_checkpoint=str(selected),actual_cumulative_tokens=base_result['cumulative_pretrain_tokens'],new_gpu_minutes=used(),combined_gpu_minutes=state['historical_gpu_minutes']+used())
    except BaseException as exc:
        status('failed',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        if awake is not None:awake.terminate();awake.wait(timeout=10)
        lock.unlink(missing_ok=True)

if __name__=='__main__':main()
