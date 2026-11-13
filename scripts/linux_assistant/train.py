"""Single-GPU BF16 LoRA with token-weighted accumulation and explicit resumable state."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import time
from scripts.common import write_json, sha256, utc
from scripts.linux_assistant.common import config,DEFAULT_CONFIG,jsonl,training_rows,verify_data,verify_model,fingerprint
from scripts.linux_assistant.modeling import load,targets,training_data,Padding

ACTIVE_CONFIG=DEFAULT_CONFIG


def frozen_digest(model):
    import torch
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        if not p.requires_grad:
            h.update(name.encode());h.update(p.detach().reshape(-1).contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()


def trainable_digest(model):
    import torch
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        if p.requires_grad:
            h.update(name.encode());h.update(p.detach().reshape(-1).contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()


def optimizer_options(training):
    """Resolve opt-in optimizer implementation without changing AdamW math."""
    mode=training.get('optimizer_mode','adamw')
    if mode=='adamw':
        return mode,{}
    if mode=='adamw_fused':
        return mode,{'fused':True}
    raise ValueError(f'Unsupported optimizer_mode: {mode}')


def gradients_are_finite(params,mode='aggregated',torch_module=None):
    """Check the same per-parameter finite predicate with one optional host sync."""
    if not params:
        return False
    if torch_module is None:
        import torch as torch_module
    flags=[torch_module.isfinite(p.grad).all() for p in params]
    if mode=='per_parameter':
        return all(bool(flag) for flag in flags)
    if mode=='aggregated':
        return bool(torch_module.stack(flags).all().item())
    raise ValueError(f'Unsupported gradient_check_mode: {mode}')


def bucket_update_chunks(chunks,data,enabled=False):
    """Sort only inside existing optimizer updates; never move records between updates."""
    if not isinstance(enabled,bool):
        raise ValueError('length_bucket_within_update must be a boolean')
    if not enabled:
        return [list(chunk) for chunk in chunks]
    return [sorted(chunk,key=lambda index:len(data[index]['input_ids'])) for chunk in chunks]


def smoke_indices(data,seed,count=16):
    """Select distinct smoke records deterministically and always include the longest sample."""
    if len(data)<count:
        raise ValueError(f'Smoke check requires {count} distinct training records')
    selected=random.Random(seed).sample(range(len(data)),count)
    longest=max(range(len(data)),key=lambda index:len(data[index]['input_ids']))
    if longest not in selected:
        selected[-1]=longest
    return selected*2


def micro_batches(chunk,data,max_examples,token_budget=None):
    """Keep every update's examples, limiting padded tokens in each forward pass."""
    batch=[];width=0
    for index in chunk:
        length=len(data[index]['input_ids'])
        if token_budget is not None and length>token_budget:
            raise ValueError('Individual sample exceeds micro_batch_token_budget; do not truncate')
        proposed=max(width,length)
        if batch and (len(batch)>=max_examples or
                (token_budget is not None and proposed*(len(batch)+1)>token_budget)):
            yield batch
            batch=[];width=0
        batch.append(index);width=max(width,length)
    if batch:yield batch


def seed_everything(seed):
    import torch
    random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)


def make_model(c,adapter=None):
    import torch
    from peft import LoraConfig, get_peft_model, PeftModel
    model,tokenizer=load(c['model'],train=True)
    target_names=targets(model)
    if adapter:
        model=PeftModel.from_pretrained(model,adapter,is_trainable=True)
    else:
        model=get_peft_model(model,LoraConfig(r=c['lora']['r'],lora_alpha=c['lora']['alpha'],
            lora_dropout=c['lora']['dropout'],bias='none',target_modules=target_names,task_type='CAUSAL_LM'))
    if any(p.requires_grad and 'lora_' not in name for name,p in model.named_parameters()):
        raise ValueError('Unexpected trainable base/vision parameter')
    model.config.get_text_config().use_cache=False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads()
    return model,tokenizer,target_names


def save_checkpoint(path,model,tokenizer,optimizer,scheduler,step,identity,history):
    import torch
    if path.exists(): raise FileExistsError(path)
    path.mkdir(parents=True)
    model.save_pretrained(path,safe_serialization=True)
    tokenizer.save_pretrained(path)
    torch.save({'step':step,'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
                'python_rng':random.getstate(),'identity':identity},path/'training-state.pt')
    write_json(path/'run-state.json',{'step':step,'identity':identity,'history':history})


def intermediate_dev_sample(rows):
    """Balanced half-size development sample for checkpoint selection only."""
    task=[r for r in rows if r['kind']=='task']
    general=[r for r in rows if r['kind']=='general']
    sampled=[]
    for category in sorted({r['category'] for r in task}):
        bucket=[r for r in task if r['category']==category]
        sampled.extend(bucket[::2])
    groups=sorted({r['group'] for r in general});target=math.ceil(len(general)/2)
    buckets=[[r for r in general if r['group']==g] for g in groups]
    selected=[];offset=0
    while len(selected)<target:
        for bucket in buckets:
            if len(bucket)>offset and len(selected)<target:selected.append(bucket[offset])
        offset+=1
    sampled.extend(selected)
    if not sampled:raise ValueError('Empty development sample')
    return sampled


def checkpoint_policy(training, total_steps, smoke=False):
    """Resolve immutable save/evaluation steps for comparable checkpoint selection."""
    if smoke:
        # Formal steps such as 63/125/188/250 cannot be applied to a four-update
        # resume check. Keep its two saves separate from formal model selection.
        if total_steps != 4:
            raise ValueError('Smoke/resume check requires four updates (effective batch 8)')
        return {2, 4}, False
    configured=training.get('checkpoint_steps')
    if configured is not None:
        if (not isinstance(configured,list) or not configured or
                any(type(step) is not int or step<1 or step>total_steps for step in configured) or
                configured!=sorted(set(configured)) or configured[-1]!=total_steps):
            raise ValueError('checkpoint_steps must be sorted unique integers ending at total updates')
        save_steps=set(configured)
    else:
        every=training.get('save_steps')
        if type(every) is not int or every<1:
            raise ValueError('save_steps must be a positive integer when checkpoint_steps is absent')
        save_steps={step for step in range(every,total_steps+1,every)}|{total_steps}
    full=training.get('full_development_at_each_checkpoint',False)
    if type(full) is not bool:raise ValueError('full_development_at_each_checkpoint must be boolean')
    return save_steps,full


def execute(c,run,smoke=False,resume=None,stop_after=None,second=False,allow_candidate_data=False):
    import torch
    from transformers import get_linear_schedule_with_warmup
    optimizer_mode,optimizer_kwargs=optimizer_options(c['training'])
    gradient_check_mode=c['training'].get('gradient_check_mode','per_parameter')
    if gradient_check_mode not in ('aggregated','per_parameter'):
        raise ValueError(f'Unsupported gradient_check_mode: {gradient_check_mode}')
    if resume is None and (run/'configuration.json').exists():
        raise FileExistsError('Existing run is immutable; resume its checkpoint or choose a new experiment')
    execution_started=time.monotonic();session_started_utc=utc()
    manifest=verify_data(c);verify_model(c['model'],c['revision'])
    candidate_data_override=False
    content_review=None
    if not c.get('training_review'):
        raise ValueError('Linux assistant training requires a current independent content-review approval, including smoke runs')
    if c.get('training_review'):
        if allow_candidate_data:raise ValueError('Reviewed workflow cannot bypass its data gate')
        from scripts.linux_assistant.review_training_data import validate_training_review
        content_review=validate_training_review(c)
    if c.get('experiment_scope')=='controlled_1000_skill_rebuild':
        # Enforce the same release gate for direct --stage launches as for the
        # unattended wrapper; do this before any CUDA initialization/model load.
        from scripts.linux_assistant.retrain import preflight
        # Direct execution must also require the reviewed clean baseline and
        # completed cleanup, not only the dataset checks used by both paths.
        preflight(c)
        if second:
            from scripts.linux_assistant.selection_bindings import require_first_failure
            require_first_failure(c)
    seed_everything(c['seed']);torch.set_num_threads(4);torch.cuda.reset_peak_memory_stats()
    active_training_rows=training_rows(c)
    active_training_ids={r['id'] for r in active_training_rows}
    source_reviews=jsonl(c['data']/'source-review.jsonl')
    source_review_by_id={r.get('id'):r for r in source_reviews}
    training_source_statuses={}
    for row in active_training_rows:
        review=source_review_by_id.get(row['id'],{})
        status=review.get('review_status','missing')
        training_source_statuses[status]=training_source_statuses.get(status,0)+1
    if not smoke:
        reviews=source_reviews
        if content_review:
            # Explicit per-record review replaces the historical hard-coded 90-scene gate.
            pass
        elif allow_candidate_data:
            reviewed_ids={r.get('id') for r in reviews if r.get('review_status') in ('verified','candidate_author_review') and r.get('evidence',{}).get('reference_sha256')}
            if not active_training_ids.issubset(reviewed_ids):
                raise ValueError('Candidate-data override requires evidence for every training record')
            candidate_data_override=any(source_review_by_id[row_id].get('review_status')!='verified' for row_id in active_training_ids)
        else:
            if len(reviews)!=90 or any(r['review_status']!='verified' or not r['evidence'] for r in reviews):
                raise ValueError('Formal training requires completed per-scenario source review')
            baseline=c['report']/'baseline-reviewed.json'
            if not baseline.exists(): raise ValueError('Formal training requires semantically reviewed dual-model baseline')
    reserve_mib=c['training'].get('gpu_reserve_mib')
    allocator_fraction=None
    if reserve_mib is not None:
        torch.cuda.empty_cache()
        free,total=torch.cuda.mem_get_info()
        available=free-int(reserve_mib)*2**20
        if available<=0:raise RuntimeError('Insufficient free VRAM before training model load')
        allocator_fraction=min(c['training'].get('allocator_max_memory_fraction',0.9),available/total)
        torch.cuda.set_per_process_memory_fraction(allocator_fraction,0)
    model,tokenizer,target_names=make_model(c,resume)
    rows,data=training_data(c,tokenizer)
    if smoke:
        # Exactly sixteen distinct training records, each seen twice over four accumulated updates.
        indices=smoke_indices(data,c['seed'])
    else:
        indices=[]
        for epoch in range(c['training']['epochs']):
            order=list(range(len(data)));random.Random(c['seed']+epoch).shuffle(order);indices.extend(order)
    accumulation=c['training']['gradient_accumulation']
    micro_batch=c['training'].get('micro_batch',1)
    if micro_batch<1 or accumulation<micro_batch:
        raise ValueError('micro_batch must be positive and fit within gradient_accumulation')
    chunks=[indices[i:i+accumulation] for i in range(0,len(indices),accumulation)]
    save_steps,full_development=checkpoint_policy(c['training'],len(chunks),smoke=smoke)
    bucket_enabled=c['training'].get('length_bucket_within_update',False)
    chunks=bucket_update_chunks(chunks,data,bucket_enabled)
    token_budget=c['training'].get('micro_batch_token_budget')
    if token_budget is not None and (not isinstance(token_budget,int) or token_budget<1):
        raise ValueError('micro_batch_token_budget must be a positive integer')
    max_length_index=max(range(len(data)),key=lambda index:len(data[index]['input_ids']))
    lr=c['training']['second_learning_rate'] if second else c['training']['learning_rate']
    identity={'config':fingerprint({k:v for k,v in c.items() if k not in ('report','export','runs','data','model','comparison_model')}),
              'manifest':fingerprint(manifest),'revision':c['revision'],'indices':fingerprint(indices),
              'smoke':smoke,'lr':lr,'targets':target_names,'source_review':sha256(c['data']/'source-review.jsonl'),
              'candidate_data_override':candidate_data_override,'content_review':content_review}
    run.mkdir(parents=True,exist_ok=True)
    session_file=run/'sessions'/('session-'+session_started_utc.replace(':','-')+'.json')
    write_json(session_file,{'started_utc':session_started_utc,'resume':str(resume) if resume else None,
               'stop_policy':'native_assistant_end_or_text_end_v2','completed':False})
    write_json(run/'configuration.json',{'identity':identity,'target_modules':target_names,
        'dataset_manifest_sha256':sha256(c['data']/'manifest.json'),
        'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
        'total_parameters':sum(p.numel() for p in model.parameters()),
        'input_tokens':sum(len(d['input_ids']) for d in data),
        'supervised_tokens':sum(sum(t!=-100 for t in d['labels']) for d in data),
        'steps':len(chunks),'precision':'BF16','packing':False,
        'checkpoint_steps':sorted(save_steps),'full_development_at_each_checkpoint':full_development,
        'effective_examples_per_update':accumulation,'micro_batch':micro_batch,
        'optimizer_mode':optimizer_mode,'gradient_check_mode':gradient_check_mode,
        'length_bucket_within_update':bucket_enabled,
        'micro_batch_token_budget':token_budget,'allocator_fraction':allocator_fraction,
        'training_split_records':len(active_training_rows),
        'training_max_sequence_length':len(data[max_length_index]['input_ids']),
        'training_max_sequence_record_id':rows[max_length_index]['id'],
        'smoke_includes_longest_record':smoke and max_length_index in indices,
        'resolved_configuration':{k:str(v) if isinstance(v,Path) else v for k,v in c.items()},
        'candidate_data_override':candidate_data_override,
        'source_review_statuses':training_source_statuses,
        'code_sha256':{name:sha256(Path(__file__).with_name(name)) for name in ('train.py','modeling.py','evaluate.py','common.py')}})
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=lr,
                               weight_decay=c['training']['weight_decay'],**optimizer_kwargs)
    scheduler=get_linear_schedule_with_warmup(optimizer,math.ceil(len(chunks)*c['training']['warmup_ratio']),len(chunks))
    start_step=0;history=[]
    if resume:
        # Only checkpoints produced in this experiment are accepted; never load untrusted pickle.
        if not resume.resolve().is_relative_to(c['runs'].resolve()): raise ValueError('Resume checkpoint outside experiment')
        state=torch.load(resume/'training-state.pt',map_location='cpu',weights_only=False)
        if state['identity']!=identity: raise ValueError('Resume data/model/config identity mismatch')
        optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        start_step=state['step'];torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng']);random.setstate(state['python_rng'])
        history=json.loads((resume/'run-state.json').read_text(encoding='utf8'))['history']
        if scheduler.last_epoch!=start_step: raise ValueError('Scheduler/step mismatch')
        optimizer_steps={int(s['step'].item() if hasattr(s['step'],'item') else s['step']) for s in optimizer.state.values() if 'step' in s}
        if optimizer_steps!={start_step}: raise ValueError('Optimizer step counters did not restore consistently')
        if optimizer.param_groups[0]['lr']!=scheduler.get_last_lr()[0]: raise ValueError('Optimizer/scheduler learning rates differ after restore')
        write_json(run/'resume-check.json',{'passed':True,'restored_step':start_step,
                  'restored_scheduler_epoch':scheduler.last_epoch,'lr':optimizer.param_groups[0]['lr'],
                  'optimizer_state_entries':len(optimizer.state),'optimizer_steps':sorted(optimizer_steps)})
    pad=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    collate=Padding(pad);base_before=frozen_digest(model) if smoke else None
    elapsed_offset=history[-1]['elapsed_seconds'] if history else 0
    lora_before=trainable_digest(model);started=time.monotonic();model.train()
    for step in range(start_step,len(chunks)):
        chunk=chunks[step];supervised=sum(sum(t!=-100 for t in data[i]['labels'][1:]) for i in chunk)
        if not supervised: raise ValueError('Empty accumulated supervised batch')
        optimizer.zero_grad(set_to_none=True);loss_sum=0.
        for indices_batch in micro_batches(chunk,data,micro_batch,token_budget):
            batch={k:v.to('cuda:0') for k,v in collate([data[i] for i in indices_batch]).items()}
            count=int((batch['labels'][:,1:]!=-100).sum())
            with torch.autocast('cuda',dtype=torch.bfloat16):
                outputs=model(**batch,use_cache=False)
                loss=outputs.loss
            if not torch.isfinite(loss): raise FloatingPointError(f'Nonfinite loss at update {step+1}')
            (loss*count/supervised).backward();loss_sum+=float(loss.detach())*count/supervised
            del outputs,loss,batch
        params=[p for p in model.parameters() if p.requires_grad and p.grad is not None]
        if not gradients_are_finite(params,gradient_check_mode,torch): raise FloatingPointError('Missing or nonfinite adapter gradients')
        if smoke and any(p.grad is not None for p in model.parameters() if not p.requires_grad): raise ValueError('Base gradient detected')
        grad=float(torch.nn.utils.clip_grad_norm_(params,c['training']['max_grad_norm']))
        optimizer.step();scheduler.step()
        record={'step':step+1,'loss':loss_sum,'grad_norm':grad,'lr':scheduler.get_last_lr()[0],
                'supervised_tokens':supervised,'elapsed_seconds':elapsed_offset+time.monotonic()-started,
                'session_elapsed_seconds':time.monotonic()-started,
                'wall_utc':utc(),
                'cuda_peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20}
        history.append(record);write_json(run/'progress.json',{'completed':False,'history':history})
        print(json.dumps(record),flush=True)
        save_now=(step+1) in save_steps or step+1==stop_after
        if save_now:
            cp=run/f'checkpoint-{step+1}'
            save_checkpoint(cp,model,tokenizer,optimizer,scheduler,step+1,identity,history)
            evaluation_steps=c['training'].get('evaluation_steps',c['training'].get('save_steps'))
            evaluate_now=(step+1) in save_steps if full_development else (
                step+1==len(chunks) or (type(evaluation_steps) is int and evaluation_steps>0 and (step+1)%evaluation_steps==0))
            if not smoke and evaluate_now:
                from scripts.linux_assistant.evaluate import evaluate
                full_dev=jsonl(c['data']/'dev.jsonl')+jsonl(c['data']/'general-dev.jsonl')
                final=step+1==len(chunks)
                dev=full_dev if full_development or final else intermediate_dev_sample(full_dev)
                scope=f'full_{len(dev)}' if len(dev)==len(full_dev) else f'stratified_{len(dev)}'
                evaluate(c,model,tokenizer,dev,run/f'dev-{step+1}.jsonl',
                         {'adapter':sha256(cp/'adapter_model.safetensors'),
                          'adapter_config_sha256':sha256(cp/'adapter_config.json'),
                          'step':step+1,'selection_scope':scope})
                model.train()
        if stop_after and step+1>=stop_after: break
    final_step=history[-1]['step']
    report={'completed':final_step==len(chunks),'step':final_step,'history':history,
            'adapter_changed':trainable_digest(model)!=lora_before,
            'frozen_base_unchanged':frozen_digest(model)==base_before if smoke else None,
            'cuda_peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
            'seconds':time.monotonic()-started}
    write_json(run/'progress.json',report)
    write_json(session_file,{'started_utc':session_started_utc,'finished_utc':utc(),
        'resume':str(resume) if resume else None,'start_step':start_step,'end_step':final_step,
        'session_total_seconds':time.monotonic()-execution_started,'loop_seconds':time.monotonic()-started,
        'stop_policy':'native_assistant_end_or_text_end_v2','completed':True})
    if smoke and (not report['adapter_changed'] or not report['frozen_base_unchanged']): raise ValueError('Smoke update/freeze check failed')
    del model,optimizer,scheduler;gc.collect();torch.cuda.empty_cache()
    return report


def reload_smoke(c,run):
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer
    from scripts.linux_assistant.evaluate import evaluate
    model,_=load(c['model'])
    checkpoint=run/'checkpoint-4'
    tokenizer=AutoTokenizer.from_pretrained(checkpoint,local_files_only=True)
    model=PeftModel.from_pretrained(model,checkpoint,is_trainable=False)
    rows=evaluate(c,model,tokenizer,jsonl(c['data']/'dev.jsonl')[:1],run/'reload-generation-native-eos.jsonl',
                  {'step':4,'smoke':True,'adapter_sha256':sha256(checkpoint/'adapter_model.safetensors')})
    if not rows[0]['answer'].strip():raise ValueError('Reloaded smoke adapter generated an empty answer')
    result=json.loads((run/'progress.json').read_text(encoding='utf8'));result['reload_passed']=True
    write_json(run/'progress.json',result)
    del model;gc.collect();torch.cuda.empty_cache()


def main():
    global ACTIVE_CONFIG
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    p.add_argument('--stage',choices=['smoke','formal'],required=True);p.add_argument('--resume',type=Path)
    p.add_argument('--second',action='store_true');p.add_argument('--allow-candidate-data',action='store_true',help='Run with source-reviewed candidate records and record the override in provenance')
    a=p.parse_args();ACTIVE_CONFIG=a.config;c=config(a.config)
    if a.stage=='smoke':
        run=c['runs']/'smoke'
        if a.resume: execute(c,run,smoke=True,resume=a.resume)
        else:
            execute(c,run,smoke=True,stop_after=2)
            execute(c,run,smoke=True,resume=run/'checkpoint-2')
        reload_smoke(c,run)
    else:
        smoke=json.loads((c['runs']/'smoke/progress.json').read_text(encoding='utf8'))
        if not smoke['completed'] or not smoke['frozen_base_unchanged'] or not smoke.get('reload_passed'): raise ValueError('M2 smoke not passed')
        if a.second:
            decision=json.loads((c['report']/'first-run-decision.json').read_text(encoding='utf8'))
            if decision.get('passed') is not False: raise ValueError('Second run only after documented development failure')
        execute(c,c['runs']/('lr-5e-5' if a.second else 'lr-1e-4'),resume=a.resume,second=a.second,allow_candidate_data=a.allow_candidate_data)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        import traceback
        # The default report remains useful even if setup/resume fails before model allocation.
        try:
            c=config(ACTIVE_CONFIG)
            write_json(c['report']/'training-failure.json',{'error':str(error),'type':type(error).__name__,
                       'traceback':traceback.format_exc(),'action':'Stop this stage and diagnose; do not switch model or scale.'})
        finally:raise
