"""Finite full-output encoding/provenance audit before unattended training."""
import argparse
from collections import Counter
import json
from pathlib import Path
from tokenizers import Tokenizer
from scripts.common import ROOT, read_json, write_json, sha256, utc
from scripts.prepare_general_2b import key
from scripts.policy_versions import sft_identity
from scripts.sft_content_checks import postcheck
from scripts.resource_guard import check_resources
from sft.template import encode_messages

def audit(plan_path):
    plan=read_json(plan_path);data=Path(plan['data_root']);runs=Path(plan['runs_root'])
    folder=data/'processed'/plan['sft']['data_version'];manifest=read_json(folder/'manifest.json')
    if manifest['policy_identity']!=sft_identity(plan_path):raise ValueError('SFT audit policy mismatch')
    tokenizer=Tokenizer.from_file(str(Path(plan['legacy_data'])/'tokenizers/vocab-32000/tokenizer.json'))
    sources={s['repo']:s for s in plan['sft']['raw_sources']}
    blobs={repo:{b['rfilename']:b for b in read_json(ROOT/s['metadata'])['siblings']} for repo,s in sources.items()}
    groups=set();families={};reports={}
    for split in ('train','validation','test'):
        tasks=Counter();counts=Counter();inputs=targets=n=0
        path=folder/f'{split}-tokens.jsonl'
        if sha256(path)!=manifest[split+'_sha256']:raise ValueError('SFT output digest mismatch')
        with path.open(encoding='utf-8') as stream:
            for n,line in enumerate(stream,1):
                row=json.loads(line);source=sources[row['source']]
                if row['revision']!=source['revision'] or row['file'] not in source['files'] or row['raw_sha256']!=blobs[row['source']][row['file']]['lfs']['sha256']:
                    raise ValueError('Unpinned provenance')
                filekey=row['source']+'/'+row['revision']+'/'+row['file']
                if row['source_row']<1 or row['group']!=key(filekey+':'+str(row['source_row'])):raise ValueError('Unstable raw row identity')
                if row['group'] in groups:raise ValueError('Repeated group across exports')
                groups.add(row['group'])
                if row['family'] in families and families[row['family']]!=split:raise ValueError('Family split leakage')
                families[row['family']]=split
                encoded=encode_messages(tokenizer,row['messages'],plan['sft']['max_length'],plan['sft']['max_answer_tokens'],system=row['system'])
                if encoded is None or any(row[k]!=v for k,v in encoded.items()):raise ValueError('Encoded tokens/labels no longer match public text')
                checked,reason=postcheck(row)
                if reason or checked['task']!=row['task']:raise ValueError('Context postcheck was not applied')
                tasks[row['task']]+=1;counts[row['source']]+=1;inputs+=len(row['input_ids'])-1;targets+=row['assistant_tokens']
                if n%4096==0:check_resources(plan,'sft')
        if n!=manifest['examples'][split] or dict(tasks)!=manifest['categories'][split]:raise ValueError('Export totals mismatch')
        reports[split]=dict(examples=n,categories=dict(tasks),sources=dict(counts),input_tokens=inputs,assistant_tokens=targets)
    train=reports['train']
    if train['examples']!=plan['sft']['train_minimum'] or train['categories'].get('daily',0)<plan['sft']['daily_minimum'] or train['categories'].get('concept',0)>plan['sft']['concept_maximum']:raise ValueError('SFT hard quotas failed')
    for repo,cap in plan['sft']['source_fraction_caps'].items():
        if train['sources'].get(repo,0)>int(cap*plan['sft']['train_minimum']):raise ValueError('Source cap failed')
    report=dict(created_utc=utc(),policy_identity=manifest['policy_identity'],files_sha256={s:manifest[s+'_sha256'] for s in reports},splits=reports,all_examples_reencoded=True,all_provenance_checked=True,distinct_groups=len(groups),families_disjoint=True)
    write_json(runs/'reports/sft_encoding_audit.json',report)
    print(json.dumps(report,ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);args=p.parse_args()
    from scripts.worker_health import WorkerHealth
    with WorkerHealth(args.plan,'preparing_sft_cpu'):audit(args.plan)
