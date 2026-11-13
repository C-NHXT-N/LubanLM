"""Independent preparation, gated training, frozen evaluation and comparison."""
import argparse
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import time

class BudgetPause(RuntimeError):
    pass


def model_weight_hash(state):
    """Hash tensors themselves, independent of optimizer and checkpoint metadata."""
    import hashlib
    import torch
    result=hashlib.sha256()
    for name,tensor in sorted(state['model'].items()):
        result.update(json.dumps([name,str(tensor.dtype),list(tensor.shape)],ensure_ascii=False).encode())
        result.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return result.hexdigest()

def write_json(path,value):
    from scripts.shortchat_io import write_json as guarded_write
    return guarded_write(path,value)

def code_identity():
    from scripts.common import ROOT,sha256
    files=('scripts/compare_shortchat.py','scripts/prepare_shortchat.py','scripts/shortchat_preparation.py','scripts/shortchat_common.py','scripts/shortchat_io.py','scripts/shortchat_allocation.py','scripts/shortchat_source_quality.py','training/train.py',
           'training/batches.py','training/execution.py','training/sft_audit.py',
           'sft/template.py','inference/generate.py','model/luban_lm.py','model/blocks.py')
    return {name:sha256(ROOT/name) for name in files}

def preflight_binding(plan):
    from scripts.common import sha256
    return dict(subset_sha256=sha256(Path(plan['data_root'])/'processed/subset-tokens.jsonl'),
                tokenizer_sha256=sha256(plan['tokenizer']),
                parents={b['name']:sha256(b['checkpoint']) for b in plan['bases']},
                code=code_identity(),effective_batch=16,precision='bf16',attention='sdpa',optimizer='fused')

def environment(plan):
    data=Path(plan['data_root']);runs=Path(plan['runs_root'])
    data.mkdir(parents=True,exist_ok=True);runs.mkdir(parents=True,exist_ok=True)
    os.environ.update(LUBAN_DATA=str(data),LUBAN_RUNS=str(runs),LUBAN_GPU_BUDGET_MINUTES=str(plan['gpu_minutes']),
        PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1',TEMP=str(data/'tmp'),TMP=str(data/'tmp'))
    (data/'tmp').mkdir(exist_ok=True)
    folder=data/'tokenizers/vocab-32000';folder.mkdir(parents=True,exist_ok=True)
    target=folder/'tokenizer.json'
    if not target.exists():shutil.copyfile(plan['tokenizer'],target)

def child(module, args, log):
    from scripts.common import utc
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a',encoding='utf-8') as out:
        out.write('\n'+json.dumps(dict(started_utc=utc(),module=module,args=list(map(str,args))),ensure_ascii=False)+'\n');out.flush()
        result=subprocess.run([sys.executable,'-B','-m',module,*map(str,args)],stdout=out,stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),env=os.environ.copy())
        out.write(json.dumps(dict(ended_utc=utc(),returncode=result.returncode))+'\n')
    if result.returncode==75:raise BudgetPause(f'{module} saved progress and paused for GPU budget; see {log}')
    if result.returncode:raise RuntimeError(f'{module} failed ({result.returncode}); see {log}')

def check_data(plan,require_review=True,manifest_path=None,config_path=None):
    from scripts.common import read_json,sha256
    from scripts.shortchat_common import CATEGORIES,SPLITS,SYSTEM,validate_review,digest,norm
    from sft.template import encode_messages
    from tokenizers import Tokenizer
    folder=Path(plan['data_root'])/'processed';manifest=read_json(manifest_path or folder/'manifest.json')
    if set(manifest['artifacts']['hashes'])!={*SPLITS,'subset'}:raise ValueError('Partial exports cannot pass the training gate')
    if sha256(plan['tokenizer'])!=plan['tokenizer_sha256'] or sha256(plan['source_plan'])!=plan['source_plan_sha256']:raise ValueError('Tokenizer or source whitelist changed')
    source_plan=read_json(plan['source_plan']);provenance={};source_manifest={}
    if require_review and plan.get('reuse_candidates_v3'):
        if config_path is None:raise ValueError('Training audit requires the actual preparation configuration')
        from scripts.common import ROOT
        files=('scripts/shortchat_preparation.py','scripts/shortchat_common.py','scripts/shortchat_io.py','scripts/shortchat_allocation.py','sft/template.py',
            'scripts/prepare_sft_100k.py','scripts/public_sft_formats.py','scripts/prepare_general_10b.py','scripts/corpus_quality.py','scripts/direct_mirror.py','scripts/shortchat_source_quality.py','eval/chat_suite.py')
        current=digest(dict(config=sha256(config_path),builder=sha256(ROOT/'scripts/prepare_shortchat.py'),code={f:sha256(ROOT/f) for f in files},
            metadata={s['repo']:sha256(ROOT/s['metadata']) for s in source_plan['sft']['raw_sources']}))
        if current!=manifest['artifacts']['policy']:raise ValueError('Frozen data uses obsolete preparation rules or configuration')
    for source in source_plan['sft']['raw_sources']:
        metadata=read_json(source['metadata'])
        if metadata.get('sha')!=source['revision']:raise ValueError('Metadata does not match pinned revision')
        blobs={r.get('path',r.get('rfilename')):r for r in metadata['siblings']}
        for file in source['files']:
            provenance[source['repo'],source['revision'],file]=blobs[file]['lfs']['sha256']
        source_manifest[source['repo']]=dict(revision=source['revision'],metadata_sha256=sha256(source['metadata']),files={file:provenance[source['repo'],source['revision'],file] for file in source['files']})
    tokenizer=Tokenizer.from_file(plan['tokenizer']);families={};ids={};originals={}
    old_questions=set();old_buckets={};old_collisions=set()
    if plan.get('reuse_candidates_v3'):
        from scripts.common import ROOT
        from scripts.shortchat_common import skeleton
        from scripts.prepare_general_10b import fingerprint
        old_path=ROOT/'docs/run-results/general-10b-20261005/frozen-test.json'
        for row in read_json(old_path).get('questions',[]):
            old_questions.add(norm(row['question']))
            old_questions.update(norm(m['content']) for m in row.get('messages',[]) if m.get('role')=='user')
        for q in old_questions:
            if len(q)<20:continue
            fp=fingerprint(skeleton(q))
            for shift in (0,16,32,48):old_buckets.setdefault((shift,(fp>>shift)&65535),set()).add(fp)
    for split,n in {**SPLITS,'subset':20000}.items():
        p=folder/(split+'-tokens.jsonl')
        if sha256(p)!=manifest['artifacts']['hashes'][split]:raise ValueError('Frozen data changed')
        f=set();i=set();o=set();counts={k:0 for k in CATEGORIES};targets=short=turns=rows=0;sources={};generated=0
        with p.open(encoding='utf-8') as stream:
            for line in stream:
                r=json.loads(line);rows+=1;i.add(r['id']);f.add(r['family_id']);counts[r['category']]+=1
                if old_questions and split!='subset':
                    for user in r['messages'][::2]:
                        q=norm(user['content'])
                        if q in old_questions:old_collisions.add(r['family_id']);continue
                        if len(q)<20:continue
                        fp=fingerprint(skeleton(q))
                        if any((fp^old).bit_count()<=3 for shift in (0,16,32,48) for old in old_buckets.get((shift,(fp>>shift)&65535),())):old_collisions.add(r['family_id'])
                original=digest([r['source'],r['raw_sha256'],r['source_row']]);o.add(original)
                if not r['generated'] and provenance.get((r['source'],r['revision'],r['file']))!=r['raw_sha256']:raise ValueError('Record is outside pinned source files')
                qhash=digest([SYSTEM,*[(m['role'],norm(m['content'])) for m in r['messages'][:-1]]])
                if r['id']!=digest([original,qhash]):raise ValueError('Sample identity changed')
                sources[r['source']]=sources.get(r['source'],0)+1;generated+=r['generated']
                if r['generated'] and r['category'] not in ('boundary','basic'):raise ValueError('Rule data outside approved categories')
                encoded=encode_messages(tokenizer,r['messages'],512,128,system=SYSTEM)
                if not encoded or any(encoded[k]!=r[k] for k in ('input_ids','labels','assistant_tokens','encoding_version')):raise ValueError('SFT encoding or EOS mismatch')
                if r['answer_tokens']!=[len(tokenizer.encode(m['content'],add_special_tokens=False).ids) for m in r['messages'][1::2]]:raise ValueError('Answer length metadata changed')
                targets+=r['assistant_tokens'];short+=sum(v<=64 for v in r['answer_tokens']);turns+=len(r['answer_tokens'])
        if rows!=n or len(i)!=n or len(o)!=n or counts!={k:v*n//100000 for k,v in CATEGORIES.items()} or short/turns<.8:raise ValueError('Count/category/length/conversation audit failed')
        if targets!=manifest['stats'][split]['supervised_tokens']:raise ValueError('Supervision count mismatch')
        if generated>n//10 or any(v>n*(.15 if s=='BelleGroup/train_1M_CN' else .4) for s,v in sources.items() if s!='rule-basics-v1'):raise ValueError('Source caps exceeded')
        families[split]=f;ids[split]=i;originals[split]=o
    splits=list(SPLITS)
    if any(families[a]&families[b] for j,a in enumerate(splits) for b in splits[j+1:]):raise ValueError('Cross-split family leakage')
    if any(originals[a]&originals[b] for j,a in enumerate(splits) for b in splits[j+1:]):raise ValueError('Cross-split original conversation leakage')
    if not ids['subset']<=ids['train']:raise ValueError('Comparison subset is not part of full train')
    if old_collisions:
        write_json(Path(plan['runs_root'])/'reports/old_test_overlap.json',dict(status='failed',families=sorted(old_collisions)))
        raise ValueError('Old test exact/near overlap; remove listed families before publishing a complete manifest')
    if old_questions:write_json(Path(plan['runs_root'])/'reports/old_test_overlap.json',dict(status='passed',families=[],old_questions=len(old_questions)))
    for split in ('dev','test'):
        if sha256(folder/(split+'-suite.json'))!=manifest['artifacts']['suite_hashes'][split]:raise ValueError('Frozen evaluation suite changed')
    if not require_review:
        write_json(Path(plan['runs_root'])/'reports/structural_audit.json',dict(status='structural_only',artifacts=manifest['artifacts'],sources=source_manifest))
        return manifest
    if plan.get('reuse_candidates_v3'):
        from scripts.shortchat_preparation import validate_boundary_review
        validate_boundary_review(folder,manifest['artifacts']['policy'])
    review=validate_review(folder/'review-samples.json',folder/'review-marks.json',manifest['artifacts'])
    samples=read_json(folder/'review-samples.json')
    if len(samples)!=800 or sum(r['split']=='train' for r in samples)!=600 or sum(r['split']=='validation' for r in samples)!=200:
        raise ValueError('Content review requires exactly 600 train and 200 validation samples')
    write_json(Path(plan['runs_root'])/'reports/data_audit.json',dict(status='passed',artifacts=manifest['artifacts'],review=review,sources=source_manifest))
    return manifest

def probe(plan):
    import torch
    from scripts.common import GPUBudget,require_gpu
    from training.train import load_checkpoint
    from training.batches import SFTPairs
    from training.execution import configure_model,update,BatchTransfer
    from model import LubanConfig,LubanLM
    torch.set_num_threads(4);device=require_gpu();results=[]
    with GPUBudget('shortchat-preflight',10):
        state=load_checkpoint(plan['bases'][0]['checkpoint'])
        for micro,accumulation in ((16,1),(8,2)):
            model=LubanLM(LubanConfig.from_dict(state['config'])).to(device);model.load_state_dict(state['model']);configure_model(model,'sdpa')
            data=SFTPairs(Path(plan['data_root'])/'processed/subset-tokens.jsonl',0,torch.Generator().manual_seed(2026),batching='bucketed')
            longest=sorted(range(len(data.rows)),key=lambda i:-data.lengths[i])[:16];data.order=longest+list(range(len(data.rows)));data.cursor=0
            optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=.1,betas=(.9,.95),fused=True)
            scaler=torch.amp.GradScaler('cuda',enabled=False);transfer=BatchTransfer(True)
            ok=False
            try:
                torch.cuda.reset_peak_memory_stats()
                update(model,optimizer,scaler,transfer.prepare([data.batch(micro,data.generator) for _ in range(accumulation)]),device,torch.bfloat16,'optimized',True)
                torch.cuda.synchronize();peak=torch.cuda.max_memory_reserved();free,total=torch.cuda.mem_get_info(device)
                ok=free>=1<<30
                seconds=None
                if ok:
                    started=time.perf_counter()
                    for _ in range(3):
                        update(model,optimizer,scaler,transfer.prepare([data.batch(micro,data.generator) for _ in range(accumulation)]),device,torch.bfloat16,'optimized',True)
                    torch.cuda.synchronize();seconds=(time.perf_counter()-started)/3
                results.append(dict(micro_batch=micro,accumulation=accumulation,ok=ok,peak_reserved_bytes=peak,device_free_bytes=free,seconds_per_update=seconds))
            except torch.cuda.OutOfMemoryError:results.append(dict(micro_batch=micro,accumulation=accumulation,ok=False,error='oom'))
            finally:data.rows.close();del model,optimizer,data;torch.cuda.empty_cache()
            if ok:
                result=dict(status='passed',binding=preflight_binding(plan),micro_batch=micro,accumulation=accumulation,probes=results,
                    estimated_three_base_training_minutes=seconds*7500/60,estimate_excludes='full validation, checkpoint writes, generation and semantic review')
                write_json(Path(plan['runs_root'])/'reports/preflight.json',result);return
        raise RuntimeError('No safe common effective batch 16')

def chat_prompt(messages):
    from scripts.shortchat_common import SYSTEM
    text='<bos>系统：'+SYSTEM+'\n'
    for m in messages:text+=('用户：' if m['role']=='user' else '助手：')+m['content']+('\n' if m['role']=='user' else '<eos>\n')
    return text+'助手：'

def evaluate(plan, checkpoint, split, alias):
    import time
    from scripts.common import read_json,sha256,utc,GPUBudget,require_gpu
    from scripts.shortchat_common import repetition
    from inference.generate import load_model,complete
    from training.execution import configure_model
    folder=Path(plan['data_root'])/'processed';suite_path=folder/(split+'-suite.json');suite=read_json(suite_path)['questions']
    out=Path(plan['runs_root'])/'evaluations'/alias;out.mkdir(parents=True,exist_ok=True)
    if not re.fullmatch(r'[a-f0-9]{12}',alias):raise ValueError('Invalid evaluation alias')
    raw=out/(split+'.jsonl');meta_path=out/(split+'-meta.json')
    binding=dict(checkpoint_sha256=sha256(checkpoint),suite_sha256=sha256(suite_path),code=code_identity(),max_new_tokens=256,temperature=0,repetition_penalty=1)
    previous=[]
    if raw.exists():
        if read_json(meta_path)['binding']!=binding:raise ValueError('Evaluation resume binding changed')
        data=raw.read_bytes()
        if data and not data.endswith(b'\n'):
            raw.write_bytes(data[:data.rfind(b'\n')+1])
        previous=[json.loads(x) for x in raw.read_text(encoding='utf-8').splitlines() if x.strip()]
        if [r['id'] for r in previous]!=[r['id'] for r in suite[:len(previous)]]:raise ValueError('Evaluation prefix mismatch')
    if len(previous)==len(suite):
        write_json(meta_path,dict(binding=binding,status='complete',completed=len(previous),total=len(suite)))
        write_json(out/(split+'-blind.json'),dict(alias=alias,raw_sha256=sha256(raw),rows=[{k:v for k,v in r.items() if k not in ('grading','generated_ids')} for r in previous]))
        return
    start=time.monotonic()
    with GPUBudget('shortchat-evaluate-'+alias+'-'+split,120) as budget:
        model,tokenizer=load_model(checkpoint,require_gpu());configure_model(model,'sdpa')
        write_json(meta_path,dict(binding=binding,status='running',completed=len(previous),total=len(suite)))
        with raw.open('a',encoding='utf-8') as stream:
            for index,r in enumerate(suite[len(previous):],len(previous)):
                if budget.remaining()<30:break
                answer,generated=complete(model,tokenizer,chat_prompt(r['messages']),max_new_tokens=256,temperature=0,repetition_penalty=1)
                row=dict(r,answer=answer,generated_ids=generated,token_count=len(generated),stop_reason='eos' if generated and generated[-1]==2 else 'length_limit',
                    replacement_character='\ufffd' in answer,repeated_flag=repetition(answer),grading=None)
                if r['category']=='basic':
                    from scripts.corpus_quality import arithmetic
                    correct=arithmetic(r['messages'][-1]['content'],answer)
                    if correct is not None:row['grading']=dict(relevant_readable=not row['replacement_character'],correct=correct,looping=row['repeated_flag'],reason='Program verified the arithmetic expression and numeric answer.')
                stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush();os.fsync(stream.fileno());budget.tick()
                completed=index+1;new=completed-len(previous);elapsed=time.monotonic()-start
                write_json(meta_path,dict(binding=binding,status='complete' if completed==len(suite) else 'running',completed=completed,total=len(suite),updated_utc=utc(),estimated_remaining_seconds=elapsed/max(1,new)*(len(suite)-completed)))
        if read_json(meta_path)['completed']!=len(suite):write_json(meta_path,dict(binding=binding,status='budget_paused',completed=read_json(meta_path)['completed'],total=len(suite)))
    if read_json(meta_path)['completed']!=len(suite):raise BudgetPause('Evaluation budget pause; incomplete suite cannot be graded')
    blinded=[{k:v for k,v in r.items() if k not in ('grading','generated_ids')} for r in [json.loads(x) for x in raw.read_text(encoding='utf-8').splitlines()]]
    write_json(out/(split+'-blind.json'),dict(alias=alias,raw_sha256=sha256(raw),rows=blinded))

def graded(plan,alias,split):
    from scripts.common import read_json,sha256
    from scripts.shortchat_common import weighted_score
    out=Path(plan['runs_root'])/'evaluations'/alias;raw=out/(split+'.jsonl')
    meta=read_json(out/(split+'-meta.json'))
    if meta['status']!='complete':raise ValueError('Incomplete evaluation')
    rows=[json.loads(x) for x in raw.read_text(encoding='utf-8').splitlines()]
    suite_path=Path(plan['data_root'])/'processed'/(split+'-suite.json');suite=read_json(suite_path)['questions']
    if meta['binding']['suite_sha256']!=sha256(suite_path) or meta['completed']!=len(suite) or len(rows)!=len(suite):raise ValueError('Evaluation coverage mismatch')
    if [r['id'] for r in rows]!=[r['id'] for r in suite] or len({r['id'] for r in rows})!=len(suite):raise ValueError('Evaluation sample order or uniqueness mismatch')
    for row,reference in zip(rows,suite):
        if any(row[k]!=v for k,v in reference.items()):raise ValueError('Frozen question or reference changed')
    marks_path=out/(split+'-marks.json')
    if not marks_path.exists():return None
    marks=read_json(marks_path)
    if marks['raw_sha256']!=sha256(raw):raise ValueError('Grades refer to different responses')
    decisions={r['id']:r for r in marks['marks']}
    pending={r['id'] for r in rows if r['grading'] is None}
    if len(decisions)!=len(marks['marks']) or set(decisions)!=pending:raise ValueError('Semantic marks incomplete or duplicated')
    for r in rows:
        if r['id'] in decisions:r['grading']={k:v for k,v in decisions[r['id']].items() if k!='id'}
    return weighted_score(rows)

def auto(plan,config):
    from scripts.common import read_json,sha256,utc
    from scripts.shortchat_common import CATEGORIES
    runs=Path(plan['runs_root']);reports=runs/'reports';reports.mkdir(exist_ok=True)
    def status(stage,**kw):write_json(reports/'controller.json',dict(stage=stage,updated_utc=utc(),**kw))
    folder=Path(plan['data_root'])/'processed'
    if not (folder/'manifest.json').exists():
        status('preparing');child('scripts.prepare_shortchat',['--plan',config,'--download'],reports/'prepare.log')
    if not (folder/'manifest.json').exists():status('data_shortfall');return
    if not (folder/'review-marks.json').exists():status('awaiting_data_review',samples=str(folder/'review-samples.json'));return
    manifest=check_data(plan,config_path=config)
    if not (reports/'preflight.json').exists():
        status('preflight');child('scripts.compare_shortchat',['--plan',config,'--phase','probe'],reports/'preflight.log')
    batch=read_json(reports/'preflight.json')
    if batch.get('status')!='passed' or batch.get('binding')!=preflight_binding(plan):raise ValueError('Preflight does not match frozen data, parents and code')
    if batch['micro_batch']*batch['accumulation']!=16:raise ValueError('Invalid effective batch')
    identity={b['name']:sha256(b['checkpoint']) for b in plan['bases']}
    frozen=dict(data=manifest['artifacts'],parents=identity,plan_sha256=sha256(config),code=code_identity())
    binding=reports/'experiment_binding.json'
    if binding.exists() and read_json(binding)!=frozen:raise ValueError('Experiment binding changed; use a new version')
    write_json(binding,frozen);epochs=[];parameters=[]
    from training.train import load_checkpoint
    weight_hashes={b['name']:model_weight_hash(load_checkpoint(b['checkpoint'])) for b in plan['bases']}
    write_json(reports/'initialization.json',dict(checkpoint_sha256=identity,initialization_weight_sha256=weight_hashes,optimizer='fresh AdamW',
        adamw_betas=[.9,.95],weight_decay=.1,gradient_clip=1.0,training_seed=2026,effective_batch=16))
    for base in plan['bases']:
        output=runs/'checkpoints'/base['name'];summary=output/'summary.json'
        if not summary.exists() or read_json(summary)['step']!=2500:
            if summary.exists() and read_json(summary)['stop_reason'].startswith('error:'):
                raise ValueError('A failed optimizer update cannot be silently resumed; inspect and restart that trial')
            status('training',base=base['name'])
            args=['--config',plan['model_config'],'--stage','sft','--output',output,'--minutes',60,'--epochs',2,'--lr',plan['lr'],
                '--warmup-steps',125,'--micro-batch',batch['micro_batch'],'--accumulation',batch['accumulation'],
                '--execution','optimized','--attention-backend','sdpa','--optimizer-backend','fused','--pin-memory',
                '--sft-train',folder/'subset-tokens.jsonl','--sft-validation',folder/'validation-tokens.jsonl',
                '--sft-batching','bucketed','--prefetch-batches',2,'--validation-micro-batch',2,'--full-sft-validation',
                '--validation-every-steps',1250,'--save-epoch-checkpoints','--save-every-minutes',15,'--log-every',100,
                '--disk-free-bytes-floor',20<<30,'--parent-sft-tokens',0]
            existing=[p for p in (output/'last.pt',output/'latest.pt',output/'epoch-01.pt') if p.exists()]
            resume=max(existing,key=lambda p:p.stat().st_mtime) if existing else None
            args+=['--resume',resume] if resume else ['--init',base['checkpoint']]
            child('training.train',args,reports/(base['name']+'-train.log'))
            if read_json(summary)['step']!=2500:status('training_paused',base=base['name']);return
        audit=read_json(output/'epoch_audit.json')
        actual=read_json(output/'run_config.json');settings=actual['settings']
        expected=dict(stage='sft',epochs=2,lr=2e-5,warmup_steps=125,micro_batch=batch['micro_batch'],accumulation=batch['accumulation'],
            execution='optimized',attention_backend='sdpa',optimizer_backend='fused',pin_memory=True,sft_batching='bucketed',
            sft_sampling='random',sft_objective='token',prefetch_batches=2,validation_micro_batch=2,full_sft_validation=True,
            validation_every_steps=1250,save_epoch_checkpoints=True)
        if any(settings.get(k)!=v for k,v in expected.items()) or actual['training_seed']!=2026 or actual['dtype']!='torch.bfloat16':raise ValueError('Trial training parameters differ from the approved experiment')
        if read_json(summary)['parent']['sha256']!=identity[base['name']]:raise ValueError('Trial initialized from a different Base')
        parameters.append(dict(base=base['name'],parameters=expected,training_seed=actual['training_seed'],dtype=actual['dtype']))
        if audit['supervised_tokens']!=manifest['stats']['subset']['supervised_tokens']*2 or len(audit['epochs'])!=2 or not all(e['every_example_once'] for e in audit['epochs']):raise ValueError('Epoch coverage or target count mismatch')
        if epochs and audit['epochs']!=epochs[0]:raise ValueError('Base comparison sampled different orders')
        epochs.append(audit['epochs'])
    write_json(reports/'training_fairness.json',dict(status='passed',bases=identity,initialization_weight_sha256=weight_hashes,epochs=epochs[0],actual_parameters=parameters,effective_batch=16,optimizer='fresh AdamW',training_seed=2026))
    aliases=[]
    for base in plan['bases']:
        for epoch in (1,2):
            ckpt=runs/'checkpoints'/base['name']/f'epoch-{epoch:02d}.pt';alias=sha256(ckpt)[:12]
            aliases.append(dict(base=base['name'],epoch=epoch,checkpoint=str(ckpt),alias=alias))
            if not (runs/'evaluations'/alias/'dev-meta.json').exists() or read_json(runs/'evaluations'/alias/'dev-meta.json')['status']!='complete':
                status('evaluating_dev',alias=alias);child('scripts.compare_shortchat',['--plan',config,'--phase','evaluate','--checkpoint',ckpt,'--split','dev','--alias',alias],reports/(alias+'-dev.log'))
    write_json(reports/'blind_aliases.json',aliases)
    scores={r['alias']:graded(plan,r['alias'],'dev') for r in aliases}
    if any(v is None for v in scores.values()):status('awaiting_dev_review',aliases=[r['alias'] for r in aliases]);return
    selected=[]
    for base in plan['bases']:
        options=[dict(r,dev=scores[r['alias']]) for r in aliases if r['base']==base['name']]
        options.sort(key=lambda r:(-r['dev']['score'],r['dev']['loop_rate'],-r['dev']['eos_rate'],r['epoch']))
        selected.append(options[0])
    write_json(reports/'selected_epochs.json',selected)
    for r in selected:
        meta=runs/'evaluations'/r['alias']/'test-meta.json'
        if not meta.exists() or read_json(meta)['status']!='complete':
            status('evaluating_test',alias=r['alias']);child('scripts.compare_shortchat',['--plan',config,'--phase','evaluate','--checkpoint',r['checkpoint'],'--split','test','--alias',r['alias']],reports/(r['alias']+'-test.log'))
        r['test']=graded(plan,r['alias'],'test')
    if any(r['test'] is None for r in selected):status('awaiting_test_review',aliases=[r['alias'] for r in selected]);return
    for r in selected:
        t=r['test'];c=t['by_category']
        r['accepted']=c['daily_single']>=.9 and c['daily_multi']>=.9 and c['instruction']>=.95 and c['basic']>=.95 and t['eos_rate']>=.98 and t['loop_rate']<=.01 and t['replacement_count']==0
    selected.sort(key=lambda r:-r['test']['score']);tied=selected[0]['test']['score']-selected[1]['test']['score']<.02
    for b in plan['bases']:
        if sha256(b['checkpoint'])!=identity[b['name']]:raise ValueError('Original Base changed')
    write_json(reports/'comparison.json',dict(status='completed',results=selected,tied=tied,recommended=None if tied or not selected[0]['accepted'] else selected[0]['base']))
    lines=['# 三个基座短对话对照','', '| 基座 | 轮次 | 加权通过率 | 自然结束 | 循环 | 达标 |','|---|---:|---:|---:|---:|---|']
    for r in selected:
        t=r['test'];lines.append(f"| {r['base']} | {r['epoch']} | {t['score']:.1%} | {t['eos_rate']:.1%} | {t['loop_rate']:.1%} | {'是' if r['accepted'] else '否'} |")
    lines+=['','差距不足2个百分点，按并列处理。' if tied else '排名基于冻结测试集，不据此保证开放问题全部正确。','本轮未启动10万条正式微调。']
    (reports/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');status('completed',comparison=str(reports/'comparison.md'))

def pending_review_files(plan,state):
    if state['stage']=='awaiting_data_review':return [Path(plan['data_root'])/'processed/review-marks.json']
    if state['stage'] in ('awaiting_dev_review','awaiting_test_review'):
        split='dev' if state['stage']=='awaiting_dev_review' else 'test'
        return [Path(plan['runs_root'])/'evaluations'/alias/(split+'-marks.json') for alias in state['aliases']]
    return []

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',default='configs/50m-shortchat-base-compare.json');p.add_argument('--phase',choices=['auto','prepare','audit','probe','evaluate'],default='auto');p.add_argument('--checkpoint');p.add_argument('--split',choices=['dev','test']);p.add_argument('--alias');p.add_argument('--wait-for-reviews',action='store_true')
    p.add_argument('--download-bind-ip',help='Local IPv4 address assigned to the physical download adapter; VPN fallback is forbidden')
    p.add_argument('--download-interface',help='Physical download adapter name, e.g. WLAN')
    args=p.parse_args()
    if args.download_bind_ip:
        from ipaddress import IPv4Address
        os.environ['LUBAN_DOWNLOAD_BIND_IP']=str(IPv4Address(args.download_bind_ip))
    if args.download_interface:os.environ['LUBAN_DOWNLOAD_INTERFACE']=args.download_interface
    config=Path(args.plan).resolve();plan=json.loads(config.read_text(encoding='utf-8'));environment(plan)
    from scripts.common import utc
    from scripts.worker_health import acquire_lock,release_lock
    lock=Path(plan['runs_root'])/'controller.lock';owner=None
    if args.phase in ('auto','prepare'):
        owner=acquire_lock(lock,('scripts.compare_shortchat',))
        if not owner:raise RuntimeError('Comparison controller already active')
    try:
        if args.phase=='auto':
            from scripts.common import read_json
            while True:
                auto(plan,config)
                required=pending_review_files(plan,read_json(Path(plan['runs_root'])/'reports/controller.json'))
                if not args.wait_for_reviews or not required:break
                while not all(path.exists() for path in required):time.sleep(10)
        elif args.phase=='prepare':
            report=Path(plan['runs_root'])/'reports/controller.json'
            write_json(report,dict(stage='preparing',updated_utc=utc()))
            child('scripts.prepare_shortchat',['--plan',config,'--download'],Path(plan['runs_root'])/'reports/prepare.log')
            write_json(report,dict(stage='awaiting_data_review' if (Path(plan['data_root'])/'processed/manifest.json').exists() else 'data_shortfall',updated_utc=utc()))
        elif args.phase=='audit':check_data(plan,config_path=config)
        elif args.phase=='probe':check_data(plan,config_path=config);probe(plan)
        elif args.phase=='evaluate':evaluate(plan,args.checkpoint,args.split,args.alias)
    except BaseException as exc:
        if isinstance(exc,BudgetPause) or str(exc)=='GPU cumulative budget exhausted':
            if args.phase=='auto':write_json(Path(plan['runs_root'])/'reports/controller.json',dict(stage='budget_paused',error=str(exc),updated_utc=utc()))
            raise SystemExit(75) from exc
        if args.phase in ('auto','prepare'):write_json(Path(plan['runs_root'])/'reports/controller.json',dict(stage='failed',error=f'{type(exc).__name__}: {exc}',updated_utc=utc()))
        raise
    finally:
        if owner:release_lock(lock,owner)

if __name__=='__main__':main()
