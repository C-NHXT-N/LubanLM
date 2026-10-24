"""CPU-only release checks for the unified data; never generates model answers."""
from __future__ import annotations
import argparse
from collections import Counter
from difflib import SequenceMatcher
import hashlib
import json
import re
import shutil
from pathlib import Path
from scripts.common import WORKSPACE,sha256,write_json,utc
from scripts.linux_assistant.common import jsonl,write_jsonl,verify_data,fingerprint
from scripts.linux_assistant.modeling import encode_training,evaluation_messages,render,Padding
from scripts.linux_assistant.semantic_grading import format_check

DATA=WORKSPACE/'datasets/linux-assistant-unified'
REPORT=WORKSPACE/'reports/linux-assistant-data-quality'
SPLITS=('train','dev','general-dev','test','general-test')

def assertion(value,message):
    if not value:raise ValueError(message)

def arithmetic(operation,a,b):
    if operation=='add':return a+b
    if operation=='sub':return a-b
    if operation=='mul':return a*b
    if operation=='div':
        assertion(b!=0 and a%b==0,'Nonintegral division fixture');return a//b
    raise ValueError('Unsupported arithmetic operation')

def oracle_answer(row):
    o=row['oracle'];kind=o['kind']
    if kind=='arithmetic':
        value=arithmetic(o['operation'],o['a'],o['b'])
        assertion(o['result']==value,'Wrong recorded arithmetic oracle')
        # Verify that the actual context supplies these operands too.
        text='\n'.join(m['content'] for m in row['messages'][:-1])
        if row.get('general_type')=='word-count':
            m=re.search(r'清单有(\d+)个盒子，每盒(\d+)个零件',text)
            assertion(m and tuple(map(int,m.groups()))==(o['a'],o['b']),'Word problem/oracle mismatch')
        elif '本题给定A=' in text:
            m=re.search(r'A=(\d+)、B=(\d+)',text)
            assertion(m and tuple(map(int,m.groups()))==(o['a'],o['b']),'Context operands/oracle mismatch')
        else:
            m=re.search(r'(\d+)([+\-×÷])(\d+)',text)
            assertion(m and (int(m[1]),int(m[3]))==(o['a'],o['b']),'Question operands/oracle mismatch')
        return str(value)
    if kind=='sum':return str(sum(o['values']))
    if kind=='order':return (max if o['select']=='max' else min)(o['values'],key=o['values'].get)
    if kind=='json':return json.dumps(o['expected'],ensure_ascii=False,separators=(',',':'))
    raise ValueError('Unknown deterministic oracle')

def numeric_signature(row):
    if row.get('oracle',{}).get('kind')=='arithmetic':
        o=row['oracle'];op,a,b=o['operation'],o['a'],o['b']
    else:
        text=' '.join(m['content'] for m in row['messages'][:-1])
        m=re.search(r'只输出(\d+)([+\-×÷])(\d+)的结果',text)
        subtraction=re.search(r'只输出(\d+)减去(\d+)的结果',text)
        word=re.search(r'有(\d+)盒彩笔，每盒(\d+)支',text)
        if m:op={'+':'add','-':'sub','×':'mul','÷':'div'}[m[2]];a,b=int(m[1]),int(m[3])
        elif subtraction:op='sub';a,b=map(int,subtraction.groups())
        elif word:op='mul';a,b=map(int,word.groups())
        else:return None
    if op in ('add','mul'):a,b=sorted((a,b))
    return op,a,b

def structural_checks(splits):
    allrows=[r for rows in splits.values() for r in rows]
    assertion(len({r['id'] for r in allrows})==len(allrows),'Duplicate IDs')
    contexts=[fingerprint(r['messages'][:-1]) for r in allrows]
    assertion(len(set(contexts))==len(contexts),'Duplicate full input contexts')
    groups={s:{r['group'] for r in splits[s]} for s in ('train','dev','test')}
    for a,b in [('train','dev'),('train','test'),('dev','test')]:
        assertion(not groups[a]&groups[b],'Cross-split task family: '+a+'/'+b)
    for r in allrows:
        msgs=r['messages']
        assertion(len(msgs)>=2 and len(msgs)%2==0,'Malformed conversation: '+r['id'])
        assertion([m['role'] for m in msgs]==['user','assistant']*(len(msgs)//2),'Roles not alternating')
        assertion(all(m['content'].strip() for m in msgs),'Empty conversation turn')
        assertion(msgs[-1]['content']==r['reference'],'Target differs from reference')
        assertion(r['facts'] and r['sources'] and r['grading']['content']['required_facts'],'Missing evidence/criteria')
        assertion(format_check(r['reference'],r['grading']['format'])[0],'Invalid reference format: '+r['id'])
        assertion(not any(t in m['content'] for m in msgs for t in ('<|im_start|>','<|im_end|>')),'Embedded chat control token')
    # These aliases were previously counted as independent held-out cases.
    assertion('troubleshooting/dev-shell-status' not in groups['dev'],'Duplicate exit-status question returned')
    assertion('environment/test-process-status' not in groups['test'],'Duplicate RSS question returned')
    assertion('linux/diskinode' not in groups['test'],'Same inode task in dev/test returned')
    assertion('interaction/given-scope-only' not in groups['test'],'Training definition copied to test')
    signatures={s:{numeric_signature(r) for r in rows if numeric_signature(r)} for s,rows in splits.items()}
    train=signatures['train'];heldout=set().union(*(signatures[s] for s in SPLITS if s!='train'))
    assertion(not train&heldout,'Arithmetic operands leaked across train/held-out: '+str(train&heldout))
    for r in splits['train']:
        if r['kind']=='general':assertion(oracle_answer(r)==r['reference'],'Oracle/reference mismatch: '+r['id'])
    heldout_arithmetic=0
    for s in SPLITS[1:]:
        for r in splits[s]:
            signature=numeric_signature(r)
            if signature:
                assertion(str(arithmetic(*signature))==r['reference'],'Held-out arithmetic reference incorrect: '+r['id'])
                heldout_arithmetic+=1
    return {'passed':True,'records':len(allrows),'duplicate_ids':0,'duplicate_full_input_contexts':0,
        'task_family_overlap':0,'train_heldout_arithmetic_operand_overlap':0,
        'deterministic_general_answers_checked':sum('oracle' in r for r in splits['train']),
        'heldout_arithmetic_answers_checked':heldout_arithmetic,
        'semantic_alias_guards':4,'all_references_satisfy_explicit_format':True}

def near_duplicates(splits):
    """Surface high similarity for review; similarity alone is not a leakage verdict."""
    tasks=[(s,r) for s,rows in splits.items() for r in rows if r['kind']=='task']
    findings=[]
    for i,(a,left) in enumerate(tasks):
        l=' '.join(m['content'] for m in left['messages'][:-1] if m['role']=='user')
        for b,right in tasks[i+1:]:
            if a==b:continue
            rr=' '.join(m['content'] for m in right['messages'][:-1] if m['role']=='user')
            score=SequenceMatcher(None,l,rr).ratio()
            if score>=.80:
                findings.append({'left':left['id'],'left_split':a,'right':right['id'],'right_split':b,
                    'similarity':round(score,4),'left_input':l,'right_input':rr,
                    'left_reference':left['reference'],'right_reference':right['reference']})
    return sorted(findings,key=lambda x:-x['similarity'])

def audit():
    from transformers import AutoTokenizer
    REPORT.mkdir(parents=True,exist_ok=True)
    manifest=verify_data({'data':DATA});splits={s:jsonl(DATA/(s+'.jsonl')) for s in SPLITS}
    structure=structural_checks(splits)
    provenance=json.loads((DATA/'curation/build-inputs.json').read_text('utf8'))
    archived=WORKSPACE/provenance['snapshot']
    for name,digest in provenance['files'].items():assertion(sha256(archived/name)==digest,'Original archive changed: '+name)
    seal=json.loads((DATA/'curation/active-seal.json').read_text('utf8'))
    assertion(seal['manifest_sha256']==sha256(DATA/'manifest.json'),'Manifest differs from active seal')
    for name,digest in seal['test_files'].items():assertion(sha256(DATA/name)==digest,'Sealed acceptance changed')
    c={'system_prompt':(DATA/'curation/system-prompt.txt').read_text('utf8'),'max_length':1024}
    model=WORKSPACE/'models/Qwen/Qwen3.5-2B'
    tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
    lengths=[];targets=[];encoded=[]
    for r in splits['train']:
        e=encode_training(c,tokenizer,r);encoded.append(e)
        positions=[i for i,t in enumerate(e['labels']) if t!=-100];target=[e['labels'][i] for i in positions]
        assertion(positions==list(range(positions[0],positions[-1]+1)),'Noncontiguous supervision')
        assertion(e['labels'][positions[-1]]==tokenizer.convert_tokens_to_ids('<|im_end|>'),'Missing supervised end token')
        assertion(tokenizer.decode(target,skip_special_tokens=True)==r['reference'],'Wrong supervised answer: '+r['id'])
        assertion(len(target)<=256,'Reference answer exceeds generation budget: '+r['id'])
        lengths.append(len(e['input_ids']));targets.append(len(target))
    padded=Padding(tokenizer.pad_token_id)([min(encoded,key=lambda e:len(e['input_ids'])),max(encoded,key=lambda e:len(e['input_ids']))])
    assertion(bool((padded['labels'][padded['attention_mask']==0]==-100).all()),'Padding is supervised')
    eval_lengths=[];eval_targets=[]
    for s in SPLITS[1:]:
        for r in splits[s]:
            context=evaluation_messages(c,r)
            assertion(context[1:]==r['messages'][:-1] and context[-1]['role']=='user','Reference turn leaked to inference')
            tokens=tokenizer(render(tokenizer,context,True),add_special_tokens=False)['input_ids'];eval_lengths.append(len(tokens))
            assertion(len(tokens)+256<=2048,'Evaluation input exceeds budget')
            target=tokenizer(r['reference'],add_special_tokens=False)['input_ids'];eval_targets.append(len(target))
            assertion(len(target)<=256,'Held-out reference exceeds 256 output Tokens')
    native={'passed':True,'samples':len(lengths),'train_sha256':sha256(DATA/'train.jsonl'),
        'native_tokenizer':'Qwen3.5-2B','model_revision':json.loads((model/'download_manifest.json').read_text('utf8'))['hf_revision'],
        'tokenizer_files':{p.name:sha256(p) for p in model.glob('*') if p.is_file() and ('tokenizer' in p.name or 'chat_template' in p.name)},
        'max_length_limit':1024,'max_input_tokens':max(lengths),'mean_input_tokens':sum(lengths)/len(lengths),
        'min_supervised_tokens':min(targets),'max_supervised_tokens':max(targets),'total_supervised_tokens':sum(targets),
        'all_final_assistant_targets_match':True,'all_prior_turns_masked':True,'padding_masked':True,
        'thinking_enabled':False,'no_truncation':True,'evaluation_contexts_checked':len(eval_lengths),
        'max_evaluation_input_tokens':max(eval_lengths),'max_evaluation_reference_tokens':max(eval_targets),
        'heldout_read_scope':'static data/mask audit only; no model generations','system_prompt_sha256':hashlib.sha256(c['system_prompt'].encode()).hexdigest()}
    native['audited_code_sha256']={str(Path(__file__).with_name(name).relative_to(WORKSPACE)).replace('\\','/'):sha256(Path(__file__).with_name(name))
        for name in ('quality_audit.py','modeling.py','common.py','semantic_grading.py')}
    sources={}
    for s,rows in splits.items():
        for r in rows:
            for source in r['sources']:
                entry=sources.setdefault(source,{'source':source,'record_ids':[]})
                entry['record_ids'].append(r['id'])
    for source,entry in sources.items():
        if source.startswith('https://'):
            entry['type']='upstream_documentation';entry['web_snapshot_pinned']=False
            entry['version_policy']='URL versions/commit used when provided; unversioned manual is not a byte-pinned snapshot'
        elif source=='problem://given-facts':entry['type']='self_contained_problem'
        else:
            p=(WORKSPACE/source).resolve();assertion(p.is_relative_to(WORKSPACE) and p.is_file(),'Invalid local source: '+source)
            entry.update(type='local_source_snapshot',sha256=sha256(p))
            snapshot=REPORT/'source-snapshots'/entry['sha256'];snapshot.parent.mkdir(parents=True,exist_ok=True)
            if snapshot.exists():assertion(sha256(snapshot)==entry['sha256'],'Local source snapshot changed')
            else:shutil.copy2(p,snapshot)
            entry['snapshot']=str(snapshot.relative_to(WORKSPACE)).replace('\\','/')
    write_json(REPORT/'source-index.json',list(sources.values()))
    write_json(REPORT/'native-token-audit.json',native)
    write_json(REPORT/'data-checks.json',{**structure,'original_archive_files_checked':len(provenance['files']),
        'previous_acceptance_bytes_preserved':True,'active_acceptance_seal_matches':True,
        'current_manifest_sha256':sha256(DATA/'manifest.json'),'sources':len(sources)})
    write_json(REPORT/'near-duplicate-candidates.json',near_duplicates(splits))
    print(json.dumps({'data_checks':structure,'native':native},ensure_ascii=False))
    return native

def finalize():
    raise RuntimeError('Automatic dataset approval is retired. Engineering audits cannot write per-record content acceptance or training approval.')
    """Sign only current passing evidence; approval is invalidated by any later edit."""
    verify_data({'data':DATA})
    checks=json.loads((REPORT/'data-checks.json').read_text('utf8'))
    native=json.loads((REPORT/'native-token-audit.json').read_text('utf8'))
    linux=json.loads((REPORT/'linux-runtime-checks.json').read_text('utf8'))
    tests=json.loads((REPORT/'tests.json').read_text('utf8'))
    repeat=json.loads((REPORT/'reproducible-build.json').read_text('utf8'))
    for label,result in [('structure',checks),('native',native),('Linux',linux),('tests',tests),('repeat',repeat)]:
        assertion(result.get('passed') is True,'Quality gate failed: '+label)
    assertion(checks['current_manifest_sha256']==sha256(DATA/'manifest.json'),'Stale structure checks')
    assertion(native['train_sha256']==sha256(DATA/'train.jsonl'),'Stale native audit')
    for name,digest in native['audited_code_sha256'].items():assertion(sha256(WORKSPACE/name)==digest,'Native-audit code changed: '+name)
    assertion(tests.get('tested_files'),'Tests lack code identity')
    for name,digest in tests['tested_files'].items():assertion(sha256(WORKSPACE/name)==digest,'Code changed after regression tests: '+name)
    assertion(linux['fixture_code_sha256']==sha256(Path(__file__).with_name('linux_data_fixtures.py')),'Stale Linux fixture results')
    assertion(repeat['manifest_sha256']==sha256(DATA/'manifest.json'),'Stale deterministic build')
    candidates=json.loads((REPORT/'near-duplicate-candidates.json').read_text('utf8'))
    decisions=json.loads((REPORT/'near-duplicate-review.json').read_text('utf8'))
    assertion(decisions['candidates_sha256']==sha256(REPORT/'near-duplicate-candidates.json'),'Stale semantic duplicate review')
    assertion(len(decisions['decisions'])==len(candidates),'Incomplete semantic duplicate review')
    for candidate,decision in zip(candidates,decisions['decisions']):
        assertion(decision['left']==candidate['left'] and decision['right']==candidate['right'] and decision.get('reason') and decision.get('same_case') is False,'Unresolved same-case overlap')
    rows=jsonl(DATA/'train.jsonl');ledger=[]
    # Required facts are the case-specific review notes, never a claim of external
    # expert review or a claim that every SDK/hardware scenario was executed.
    for r in rows:
        ledger.append({'id':r['id'],'group':r['group'],'record_sha256':fingerprint(r),'accepted':True,
            'reviewer':'Codex data-source/criteria review','reason':'核对题内条件与参考答案：'+'；'.join(r['facts']),
            'source_urls':r['sources'],'independent_human_review':False,'linux_board_execution':False,
            'runtime_scope':'Only fixed behaviors listed in linux-runtime-checks.json were executed; other scenarios are documentation/logic review.'})
    write_jsonl(REPORT/'approved-training-records.jsonl',ledger)
    evidence_names=['data-checks.json','native-token-audit.json','linux-runtime-checks.json','tests.json',
        'reproducible-build.json','source-index.json','near-duplicate-candidates.json','near-duplicate-review.json']
    evidence={str((REPORT/name).relative_to(WORKSPACE)).replace('\\','/'):sha256(REPORT/name) for name in evidence_names}
    code_names=['build_quality_dataset.py','quality_cases.py','quality_audit.py','semantic_grading.py','common.py',
        'modeling.py','review.py','review_template.py','review_training_data.py','train.py','evaluate.py','baseline.py','retrain.py',
        'test_quality_data.py','test_reviewed_training.py','test_pipeline.py','test_baseline.py','linux_data_fixtures.py',
        'select.py','final_round.py','start_baseline.ps1','start_retrain.ps1']
    for name in code_names:
        p=Path(__file__).with_name(name);evidence[str(p.relative_to(WORKSPACE)).replace('\\','/')]=sha256(p)
    evidence['datasets/linux-assistant-unified/curation/active-seal.json']=sha256(DATA/'curation/active-seal.json')
    for name in ('deploy/linux_assistant/acceptance.py','configs/linux-assistant-qwen35-baseline-quality.json','configs/linux-assistant-qwen35-2b-quality.json'):
        p=WORKSPACE/'LubanLM'/name;evidence[str(p.relative_to(WORKSPACE)).replace('\\','/')]=sha256(p)
    for entry in json.loads((REPORT/'source-index.json').read_text('utf8')):
        if entry.get('snapshot'):evidence[entry['snapshot']]=entry['sha256']
    result={'passed':True,'completed_utc':utc(),'reviewer':'Codex data-source/criteria review',
        'review_scope':'All active training records; static held-out integrity/schema review only; no training or model evaluation',
        'train_sha256':sha256(DATA/'train.jsonl'),'manifest_sha256':sha256(DATA/'manifest.json'),
        'source_review_sha256':sha256(DATA/'source-review.jsonl'),'ledger_sha256':sha256(REPORT/'approved-training-records.jsonl'),
        'system_prompt_sha256':native['system_prompt_sha256'],'evidence':evidence,
        'external_expert_review':False,'linux_board_execution':False,'new_baseline_required':True,
        'not_a_model_quality_pass':True}
    pending=REPORT/'training-approval.pending.json'
    write_json(pending,result)
    from scripts.linux_assistant.review_training_data import validate_training_review
    validated=validate_training_review({'data':DATA,'training_review':str(pending.relative_to(WORKSPACE)),
        'system_prompt':(DATA/'curation/system-prompt.txt').read_text('utf8')})
    pending.replace(REPORT/'training-approval.json')
    print(json.dumps(validated,ensure_ascii=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--finalize',action='store_true')
    args=parser.parse_args();finalize() if args.finalize else audit()
