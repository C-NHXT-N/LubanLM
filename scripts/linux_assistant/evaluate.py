"""Resumable development evaluation. Final evaluation requires a frozen candidate lock."""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import time
from scripts.common import write_json, sha256
from scripts.linux_assistant.common import config,DEFAULT_CONFIG,jsonl,write_jsonl,verify_data,verify_model,screen,fingerprint
from scripts.linux_assistant.modeling import load,evaluation_messages,render,generation_eos_ids
from scripts.linux_assistant.baseline import atomic_rows,prepare,resume_rows,generate_batch,autotune


def evaluate(c,model,tokenizer,rows,destination,identity):
    import torch
    torch.set_num_threads(4)
    destination=Path(destination)
    eos_ids=generation_eos_ids(model,tokenizer)
    identity={**identity,'data_hash':fingerprint(rows),'system':c['system_prompt'],
              'context_length':c['context_length'],
              'prompt_policy':'context_through_last_user_exclude_target_assistant',
              'generation':{'greedy':True,'max_new_tokens':c['max_new_tokens'],'thinking':False,
                            'eos_token_ids':eos_ids,'stop_policy':'native_assistant_end_or_text_end_v2'}}
    identity_path=destination.with_suffix('.identity.json')
    if identity_path.exists():
        if json.loads(identity_path.read_text(encoding='utf8'))!=identity:
            raise ValueError('Cannot resume evaluation with different model/data/settings')
    else: write_json(identity_path,identity)
    prepared=prepare(c,tokenizer,rows)
    previous=resume_rows(destination,prepared)
    completed=[previous[p['row']['id']] for p in prepared if p['row']['id'] in previous]
    if [r['id'] for r in completed]!=[r['id'] for r in rows[:len(completed)]]:
        raise ValueError('Evaluation resume requires a contiguous committed prefix')
    model.eval();text_config=model.config.get_text_config()
    old_cache=text_config.use_cache;text_config.use_cache=True
    events=[];batch_size=1;tune=None
    try:
        if (c.get('experiment_scope')=='controlled_1000_skill_rebuild' and
                not identity.get('final_round') and len(completed)<len(prepared)):
            batching=destination.with_name(destination.stem+'.batching')
            def status(phase,**extra):
                write_json(batching/'status.json',{'phase':phase,**extra})
            # Each adapter gets its own measured token-exact parity check.
            # A baseline batch-size choice is never assumed valid for LoRA.
            tune=autotune(c,model,tokenizer,prepared,eos_ids,batching,identity,status)
            batch_size=tune['selected_batch_size']
        while len(completed)<len(prepared):
            items=prepared[len(completed):len(completed)+batch_size]
            try:
                generated,seconds=generate_batch(model,tokenizer,items,c['max_new_tokens'],eos_ids)
            except torch.cuda.OutOfMemoryError:
                smaller=[b for b in tune['eligible_fallback_batches'] if b<batch_size] if tune else []
                if not smaller:raise
                gc.collect();torch.cuda.empty_cache()
                events.append({'event':'oom_reduce_batch','from':batch_size,'to':max(smaller)})
                batch_size=max(smaller)
                write_json(destination.with_suffix('.batch-events.json'),events)
                continue
            if len(generated)!=len(items):
                raise ValueError('Evaluation batch output coverage differs from its inputs')
            for item,ids in zip(items,generated):
                row=item['row'];answer=tokenizer.decode(ids,skip_special_tokens=True)
                result={'id':row['id'],'group':row['group'],'kind':row['kind'],'category':row['category'],
                        'messages':row['messages'],'input_messages':item['context'][1:],
                        'prompt':item['prompt'],'input_ids':item['input_ids'],
                        'answer':answer,'generated_ids':ids,'reference':row['reference'],'rubric':row['rubric'],
                        'seconds':seconds,'batch_size':len(items),'timing_scope':'shared_batch_wall',
                        'stop_reason':'eos' if ids and ids[-1] in eos_ids else 'length_limit',**screen(row,answer)}
                for name in ('grading','evaluation_capability','evaluation_subskill'):
                    if name in row:result[name]=row[name]
                if any(t in eos_ids for t in ids[:-1]):raise ValueError('Generation continued after a native stop token')
                completed.append(result)
            atomic_rows(destination,completed)
            print(f'EVAL {destination.stem} {len(completed)}/{len(rows)} batch={len(items)} {seconds:.1f}s',flush=True)
    finally: text_config.use_cache=old_cache
    write_json(destination.with_suffix('.summary.json'),{
        'completed':len(completed)==len(rows),'samples':len(completed),
        'step':identity.get('step'),'evaluation_scope':identity.get('selection_scope','full'),
        'screen_pass':None if any('grading' in r for r in completed) else sum(r['screen_pass'] for r in completed),
        'format_pass':sum(r.get('format_compliant') is True for r in completed),'semantic_review':'pending',
        'accepted_score':None,'selected_batch_size':batch_size,'oom_events':events,
        'batching_parity':'verified_vs_batch1' if tune else 'single_example_or_already_completed',
        'note':'Necessary-condition screening is not semantic acceptance.'})
    return completed


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    p.add_argument('--model',choices=['qwen35','qwen3','merged'],default='qwen35')
    p.add_argument('--adapter',type=Path)
    p.add_argument('--split',choices=['dev','test'],default='dev')
    p.add_argument('--output',type=Path)
    a=p.parse_args();c=config(a.config);verify_data(c)
    path=c['comparison_model'] if a.model=='qwen3' else c['model']
    if a.model=='merged': path=c['export']/'merged'
    identity={'model':str(path),'adapter':None}
    if a.model!='merged': identity['revision']=verify_model(path,c['comparison_revision'] if a.model=='qwen3' else c['revision'])
    if a.adapter:
        identity['adapter']={str(f.name):sha256(f) for f in a.adapter.glob('adapter*') if f.is_file()}
    out=a.output or c['report']/f'{a.model}-{a.split}.jsonl'
    if a.split=='test':
        lock=c['report']/'candidate-lock.json'
        if not lock.exists(): raise ValueError('Final test is sealed until candidate-lock.json exists')
        # Final orchestration owns one evaluation round; no interactive --test tuning loop.
        raise ValueError('Use the locked final-round orchestrator, not standalone test evaluation')
    model,tokenizer=load(path)
    if a.adapter:
        from peft import PeftModel
        model=PeftModel.from_pretrained(model,a.adapter,is_trainable=False)
    rows=jsonl(c['data']/f'{a.split}.jsonl')+jsonl(c['data']/f'general-{a.split}.jsonl')
    evaluate(c,model,tokenizer,rows,out,identity)


if __name__=='__main__': main()
