"""Unattended, resumable BF16 baseline with measured batching and greedy parity."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import gc
import json
import os
from pathlib import Path
import time
import traceback

from scripts.common import WORKSPACE, write_json, sha256, utc
from scripts.linux_assistant.common import config, jsonl, fingerprint, verify_data, verify_model, screen
from scripts.linux_assistant.modeling import load, render, evaluation_messages, generation_eos_ids

POLICY='context_through_last_user_exclude_target_assistant'


def validate_batch_candidates(c):
    candidates=c['baseline']['batch_candidates']
    if (not isinstance(candidates,list) or not candidates or
            any(type(b) is not int or b<1 for b in candidates) or
            candidates[0]!=1 or candidates!=sorted(set(candidates))):
        raise ValueError('Batch candidates must be sorted positive integers starting at 1')
    return candidates


def atomic_rows(path,rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temporary.open('w',encoding='utf8') as stream:
        for row in rows:
            stream.write(json.dumps(row,ensure_ascii=False)+'\n')
        stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)


@contextmanager
def exclusive_job(path):
    """OS lock is released on process exit; restarting never needs a stale-lock delete."""
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+b') as stream:
        if stream.tell()==0:stream.write(b'0');stream.flush()
        stream.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:
            stream.seek(0)
            if os.name=='nt':msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(stream,fcntl.LOCK_UN)


def trim_generation(ids,eos_ids):
    for i,token in enumerate(ids):
        if token in eos_ids:return ids[:i+1]
    return ids


def prepare(c,tokenizer,rows):
    prepared=[]
    for row in rows:
        context=evaluation_messages(c,row)
        prompt=render(tokenizer,context,True)
        ids=tokenizer(prompt,add_special_tokens=False)['input_ids']
        if len(ids)+c['max_new_tokens']>c['context_length']:
            raise ValueError('Prompt exceeds fixed context budget: '+row['id'])
        prepared.append({'row':row,'context':context,'prompt':prompt,'input_ids':ids})
    return prepared


def generate_batch(model,tokenizer,prepared,max_new_tokens,eos_ids):
    import torch
    from transformers import GenerationConfig
    pad=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_ids[0]
    width=max(len(p['input_ids']) for p in prepared)
    ids=[[pad]*(width-len(p['input_ids']))+p['input_ids'] for p in prepared]
    masks=[[0]*(width-len(p['input_ids']))+[1]*len(p['input_ids']) for p in prepared]
    inputs={'input_ids':torch.tensor(ids,dtype=torch.long,device=model.device),
            'attention_mask':torch.tensor(masks,dtype=torch.long,device=model.device)}
    generation=GenerationConfig(do_sample=False,use_cache=True,max_new_tokens=max_new_tokens,
                               eos_token_id=eos_ids,pad_token_id=pad)
    torch.cuda.synchronize();started=time.perf_counter()
    with torch.inference_mode():output=model.generate(**inputs,generation_config=generation)
    torch.cuda.synchronize();seconds=time.perf_counter()-started
    generated=[trim_generation(r[width:].tolist(),eos_ids) for r in output]
    del inputs,output
    return generated,seconds


def pilot_rows(prepared,n):
    """Cover categories plus both ends of the prompt length range, without test data."""
    selected=[]
    for category in sorted({p['row']['category'] for p in prepared}):
        bucket=[p for p in prepared if p['row']['category']==category]
        selected.append(max(bucket,key=lambda p:len(p['input_ids'])))
    for p in sorted(prepared,key=lambda p:len(p['input_ids'])):
        if len(selected)>=n:break
        if p not in selected:selected.append(p)
    return sorted(selected[:n],key=lambda p:len(p['input_ids']))


def autotune(c,model,tokenizer,prepared,eos_ids,directory,identity,status):
    import torch
    candidates=validate_batch_candidates(c)
    path=directory/'autotune.json'
    if path.exists():
        cached=json.loads(path.read_text(encoding='utf8'))
        if cached['identity']!=identity:raise ValueError('Autotune identity mismatch')
        return cached
    pilot=pilot_rows(prepared,c['baseline']['pilot_samples'])
    trials=[];reference=None
    for batch_size in candidates:
        status('autotuning',candidate_batch=batch_size,pilot_samples=len(pilot))
        outputs=[];seconds=0.
        torch.cuda.reset_peak_memory_stats()
        try:
            # Kernel warmup is excluded from measurements and uses only development input.
            generate_batch(model,tokenizer,pilot[:min(batch_size,len(pilot))],2,eos_ids)
            for offset in range(0,len(pilot),batch_size):
                ids,duration=generate_batch(model,tokenizer,pilot[offset:offset+batch_size],c['max_new_tokens'],eos_ids)
                outputs.extend(ids);seconds+=duration
            if reference is None:reference=outputs
            mismatch=[p['row']['id'] for p,a,b in zip(pilot,reference,outputs) if a!=b]
            trials.append({'batch_size':batch_size,'seconds':seconds,'token_exact_parity':not mismatch,
                           'mismatches':mismatch,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
                           'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20})
        except torch.cuda.OutOfMemoryError:
            gc.collect();torch.cuda.empty_cache()
            trials.append({'batch_size':batch_size,'oom':True,'token_exact_parity':False})
            break
        write_json(directory/'autotune-progress.json',{'trials':trials})
    valid=[t for t in trials if t.get('token_exact_parity')]
    if not valid:raise ValueError('No valid greedy baseline execution mode')
    best=min(valid,key=lambda t:t['seconds'])
    result={'identity':identity,'pilot_ids':[p['row']['id'] for p in pilot],
            'pilot_outputs_batch1':reference,'trials':trials,'selected_batch_size':best['batch_size'],
            'eligible_fallback_batches':sorted(t['batch_size'] for t in valid),
            'policy':'Fastest measured candidate with token-exact greedy parity to batch=1.'}
    write_json(path,result)
    return result


def resume_rows(path,prepared):
    expected={p['row']['id']:p for p in prepared}
    done={}
    for row in jsonl(path) if path.exists() else []:
        key=row['id']
        if key in done or key not in expected:raise ValueError('Unexpected/duplicate resumed result: '+key)
        p=expected[key]
        if (row['input_ids']!=p['input_ids'] or row['prompt']!=p['prompt'] or
                row['messages']!=p['row']['messages'] or row['input_messages']!=p['context'][1:]):
            raise ValueError('Resumed prompt changed: '+key)
        done[key]=row
    return done


def execute(c,directory):
    import torch
    started=time.perf_counter();started_utc=utc()
    directory.mkdir(parents=True,exist_ok=True)
    status_path=directory/'status.json'
    def status(phase,**extra):
        write_json(status_path,{'state':'running','phase':phase,'pid':os.getpid(),
                   'started_utc':started_utc,'updated_utc':utc(),'elapsed_seconds':time.perf_counter()-started,**extra})
    with exclusive_job(directory/'worker.lock'),exclusive_job(WORKSPACE/'runs/linux-assistant-gpu.lock'):
        try:
            status('validating')
            manifest=verify_data(c)
            if (manifest.get('construction')=='structured_semantic_cases' or
                    c.get('experiment_scope')=='controlled_1000_skill_rebuild'):
                if not c.get('training_review'):raise ValueError('Rebuilt data requires current quality approval before baseline generation')
                from scripts.linux_assistant.review_training_data import validate_training_review
                validate_training_review(c)
            revision=verify_model(c['model'],c['revision'])
            if c.get('experiment_scope')=='controlled_1000_skill_rebuild':
                from transformers import AutoTokenizer
                from scripts.linux_assistant.modeling import training_data
                from scripts.linux_assistant.release_contract import validate_release_contract
                contract_tokenizer=AutoTokenizer.from_pretrained(c['model'],local_files_only=True,trust_remote_code=False)
                contract_rows,contract_encoded=training_data(c,contract_tokenizer)
                validate_release_contract(c,contract_rows,contract_encoded)
                del contract_tokenizer,contract_rows,contract_encoded
            validate_batch_candidates(c)
            rows=jsonl(c['data']/'dev.jsonl')+jsonl(c['data']/'general-dev.jsonl')
            if len({r['id'] for r in rows})!=len(rows):raise ValueError('Duplicate development IDs')
            identity={'model':str(c['model']),'revision':revision,'manifest':fingerprint(manifest),
                      'rows':fingerprint(rows),'system_prompt':c['system_prompt'],'prompt_policy':POLICY,
                      'max_new_tokens':c['max_new_tokens'],'context_length':c['context_length'],
                      'greedy':True,'thinking':False,'precision':'BF16','baseline_options':c['baseline'],
                      'code_sha256':{name:sha256(Path(__file__).with_name(name)) for name in ('baseline.py','modeling.py')}}
            identity_path=directory/'identity.json'
            if identity_path.exists() and json.loads(identity_path.read_text(encoding='utf8'))!=identity:
                raise ValueError('Resume model/data/settings/code identity mismatch; choose another output directory')
            write_json(identity_path,identity)
            torch.set_num_threads(c['baseline'].get('cpu_threads',4))
            torch.manual_seed(c['seed']);torch.cuda.manual_seed_all(c['seed'])
            torch.cuda.set_per_process_memory_fraction(c['baseline'].get('memory_fraction',.85))
            status('loading_model')
            model,tokenizer=load(c['model']);eos_ids=generation_eos_ids(model,tokenizer)
            prepared=prepare(c,tokenizer,rows)
            write_json(directory/'prompt-audit.json',{'completed':True,'samples':len(prepared),
                       'policy':POLICY,'all_contexts_end_in_user':all(p['context'][-1]['role']=='user' for p in prepared),
                       'removed_target_turns':sum(r['messages'][-1]['role']=='assistant' for r in rows),
                       'test_split_loaded':False,'max_input_tokens':max(len(p['input_ids']) for p in prepared)})
            tune=autotune(c,model,tokenizer,prepared,eos_ids,directory,identity,status)
            results=directory/'answers.jsonl';done=resume_rows(results,prepared)
            pending=sorted([p for p in prepared if p['row']['id'] not in done],key=lambda p:len(p['input_ids']))
            batch_size=tune['selected_batch_size'];position=0;events=[]
            torch.cuda.reset_peak_memory_stats();evaluation_started=time.perf_counter();initial_done=len(done)
            while position<len(pending):
                batch=pending[position:position+batch_size]
                status('evaluating',completed=len(done),total=len(rows),batch_size=batch_size,
                       current_ids=[p['row']['id'] for p in batch])
                try:generated,seconds=generate_batch(model,tokenizer,batch,c['max_new_tokens'],eos_ids)
                except torch.cuda.OutOfMemoryError:
                    lower=[b for b in tune['eligible_fallback_batches'] if b<batch_size]
                    if not lower:raise
                    gc.collect();torch.cuda.empty_cache()
                    events.append({'event':'oom_reduce_batch','from':batch_size,'to':max(lower),'utc':utc()})
                    batch_size=max(lower);write_json(directory/'events.json',events);continue
                for p,ids in zip(batch,generated):
                    row=p['row'];answer=tokenizer.decode(ids,skip_special_tokens=True)
                    done[row['id']]={'id':row['id'],'group':row['group'],'kind':row['kind'],'category':row['category'],
                        'messages':row['messages'],'input_messages':p['context'][1:],
                        'prompt':p['prompt'],'input_ids':p['input_ids'],
                        'answer':answer,'generated_ids':ids,'reference':row['reference'],'rubric':row['rubric'],
                        'stop_reason':'eos' if ids and ids[-1] in eos_ids else 'length_limit',
                        'batch_size':len(batch),'batch_seconds':seconds,'timing_scope':'shared_batch_wall',
                        'model_revision':revision,'prompt_policy':POLICY,**screen(row,answer)}
                    if 'grading' in row:done[row['id']]['grading']=row['grading']
                position+=len(batch)
                atomic_rows(results,[done[r['id']] for r in rows if r['id'] in done])
                processed=len(done)-initial_done
                eta=(time.perf_counter()-evaluation_started)/processed*(len(rows)-len(done))
                status('evaluating',completed=len(done),total=len(rows),batch_size=batch_size,eta_seconds=eta)
                print(f'BASELINE {len(done)}/{len(rows)} batch={len(batch)} seconds={seconds:.2f}',flush=True)
            ordered=[done[r['id']] for r in rows]
            groups={}
            for kind in ('task','general'):
                selected=[r for r in ordered if r['kind']==kind]
                groups[kind]={'samples':len(selected),
                    'necessary_condition_pass':None if any('grading' in r for r in selected) else sum(r['screen_pass'] for r in selected),
                    'format_pass':sum(r.get('format_compliant') is True for r in selected)}
            summary={'completed':True,'samples':len(ordered),'model_revision':revision,'prompt_policy':POLICY,
                     'selected_batch_size':tune['selected_batch_size'],'groups':groups,
                     'semantic_review':'pending','accepted_score':None,'test_split_loaded':False,
                     'started_utc':started_utc,'finished_utc':utc(),'session_seconds':time.perf_counter()-started,
                     'evaluation_session_seconds':time.perf_counter()-evaluation_started,
                     'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
                     'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,
                     'answers_sha256':sha256(results),'oom_events':events,
                     'note':'Rule screening is not semantic grading; previous target-leaking evaluation is invalid.'}
            write_json(directory/'summary.json',summary)
            atomic_rows(directory/'review-queue.jsonl',[{**r,'accepted':None,'review_reason':''} for r in ordered])
            report=f'''# Qwen3.5-2B原模型统一开发集基线

状态：生成完成，待语义复核。共{len(ordered)}题，使用原模型BF16、greedy、关闭思考、最多{c['max_new_tokens']}个输出Token；仅使用开发集。目标答案未传入模型，历史上下文保留。

自动选择批次{tune['selected_batch_size']}；选择依据是开发样本上与batch=1逐Token相同且耗时最低。耗时约{summary['session_seconds']/60:.1f}分钟（本次进程含加载和批次校验），显存峰值{summary['peak_allocated_mib']/1024:.2f}GiB。

任务{groups['task']['samples']}题、通用{groups['general']['samples']}题。内容与步骤待逐题语义复核，accepted_score尚未生成；明确格式检查通过分别为{groups['task']['format_pass']}和{groups['general']['format_pass']}题，不能把格式检查当作正确率。逐题原始结果见answers.jsonl，复核队列见review-queue.jsonl，完整指标见summary.json。

旧评测把目标助手答案放入输入，因此旧分数已撤回，不能与本次干净基线比较；适配器需要使用修正入口重新评测。
'''
            (directory/'REPORT.md').write_text(report,encoding='utf8')
            write_json(status_path,{'state':'awaiting_semantic_review','phase':'awaiting_semantic_review','generation_completed':True,'pid':os.getpid(),
                       'completed':len(done),'total':len(rows),'finished_utc':utc(),'summary':'summary.json'})
        except BaseException as error:
            write_json(status_path,{'state':'failed','pid':os.getpid(),'updated_utc':utc(),
                       'error':str(error),'type':type(error).__name__,'traceback':traceback.format_exc(),
                       'resume':'Rerun the same launcher; validated completed batches will be reused.'})
            raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    a=parser.parse_args();c=config(a.config)
    directory=(WORKSPACE/c.get('baseline_report',c['report'])).resolve()
    execute(c,directory)


if __name__=='__main__':main()
