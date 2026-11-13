"""Collect all preparation defects; engineering checks never approve content."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from scripts.common import WORKSPACE, sha256, utc, write_json
from scripts.linux_assistant.common import config, fingerprint, jsonl,review_digest
from scripts.linux_assistant.dataset_quality import row_issues, split_issues

CATEGORIES = ('environment_check','linux_operation','deployment',
              'troubleshooting','interaction','general')
SPLIT_FILES = {'train':('train.jsonl','general-train.jsonl'),
               'dev':('dev.jsonl','general-dev.jsonl'),
               'test':('test.jsonl','general-test.jsonl')}


def review_coverage(rows, decisions, label, other_reviewers=None):
    issues=[]; by_id={}
    for decision in decisions:
        ident=decision.get('id')
        if ident in by_id: issues.append({'id':ident,'issue':label+'_duplicate_decision'})
        by_id[ident]=decision
    for row in rows:
        decision=by_id.get(row['id'],{})
        try:digest=review_digest(decision)
        except ValueError:
            digest=None;issues.append({'id':row['id'],'issue':label+'_conflicting_digest_aliases'})
        reviewer=decision.get('reviewer_id')
        if not decision:
            issues.append({'id':row['id'],'issue':label+'_missing'})
            continue
        if (decision.get('decision')!='accept' or digest!=fingerprint(row) or
                not reviewer or reviewer==row.get('created_by') or not decision.get('reason')):
            issues.append({'id':row['id'],'issue':label+'_not_accepted_or_stale'})
        if other_reviewers and reviewer in other_reviewers.get(row['id'],set()):
            issues.append({'id':row['id'],'issue':label+'_reviewer_not_independent'})
    return issues


def inspect_records(c, splits, encoded_by_id=None):
    """Return every issue, including quota deficits, without modifying rows."""
    issues=split_issues(splits)
    rows=[r for split in splits.values() for r in split]
    case_splits={};atom_splits={}
    for split,records in splits.items():
        for row in records:
            case=row.get('case_id')
            if case and case in case_splits and case_splits[case]!=split:
                issues.append({'id':row['id'],'issue':'cross_split_case_id','other_split':case_splits[case]})
            if case: case_splits[case]=split
            atom=row.get('source_atom_id')
            if atom:
                if atom in atom_splits and atom_splits[atom]!=split:
                    issues.append({'id':row['id'],'issue':'cross_split_source_atom',
                                   'source_atom_id':atom,'other_split':atom_splits[atom]})
                atom_splits[atom]=split
    source_cache={}
    for row in rows:
        ident=row.get('id','?')
        if row.get('grading',{}).get('format'):
            from scripts.linux_assistant.semantic_grading import format_check
            try:
                ok,reason=format_check(row.get('reference',''),row['grading']['format'])
                if not ok:issues.append({'id':ident,'issue':'reference_violates_own_format','detail':reason})
            except (ValueError,KeyError,TypeError) as error:
                issues.append({'id':ident,'issue':'invalid_format_contract','detail':str(error)})
        for field in ('skill_objective','input_dependencies','counterfactual_change',
                      'skill_tags','evidence_refs','rubric_support','case_id','group','created_by'):
            if not row.get(field): issues.append({'id':ident,'issue':'missing_'+field})
        if row.get('context_sufficient') is not True:
            issues.append({'id':ident,'issue':'context_not_sufficient'})
        if not isinstance(row.get('sources'),list) or not row['sources']:
            issues.append({'id':ident,'issue':'missing_pinned_sources'})
        for source in row.get('sources',[]):
            if not isinstance(source,dict):
                issues.append({'id':ident,'issue':'source_not_locally_pinned'})
                continue
            path=source.get('path'); digest=source.get('sha256')
            if not path or not digest:
                issues.append({'id':ident,'issue':'source_path_or_digest_missing'})
                continue
            resolved=(WORKSPACE/path).resolve()
            if not resolved.is_relative_to(WORKSPACE.resolve()):
                issues.append({'id':ident,'issue':'source_escapes_workspace'})
            elif not resolved.is_file():
                issues.append({'id':ident,'issue':'source_file_missing','path':path})
            else:
                if resolved not in source_cache: source_cache[resolved]=sha256(resolved)
                if source_cache[resolved]!=digest:
                    issues.append({'id':ident,'issue':'source_digest_changed','path':path})
    counts={name:dict(Counter(r.get('category') for r in split)) for name,split in splits.items()}
    train=splits.get('train',[])
    train_tokens=[encoded_by_id[r['id']] for r in train if encoded_by_id and r['id'] in encoded_by_id]
    supervised=sum(sum(t!=-100 for t in e['labels']) for e in train_tokens)
    doc_tokens=sum(sum(t!=-100 for t in encoded_by_id[r['id']]['labels'])
                   for r in train if encoded_by_id and r['id'] in encoded_by_id
                   and 'document_explanation' in r.get('skill_tags',[]))
    return {'row_count':len(rows),'split_counts':{k:len(v) for k,v in splits.items()},
            'category_counts':counts,'issues':issues,
            'unique_cases_by_split':{name:len({r.get('case_id') for r in records}) for name,records in splits.items()},
            'unique_source_atoms_by_split':{name:len({r['source_atom_id'] for r in records if r.get('source_atom_id')}) for name,records in splits.items()},
            'training_multiturn':sum(len(r.get('messages',[]))>=4 for r in train),
            'training_supervised_tokens':supervised,
            'document_supervised_token_fraction':doc_tokens/supervised if supervised else None,
            'row_fingerprints':{r['id']:fingerprint(r) for r in rows}}


def audit(c, data, output, pilot=False):
    from transformers import AutoTokenizer
    from scripts.linux_assistant.modeling import encode_training
    if pilot:
        splits={'train':jsonl(data/'pilot.jsonl')}
    else:
        splits={name:[r for file in files if (data/file).is_file() for r in jsonl(data/file)]
                for name,files in SPLIT_FILES.items()}
    tokenizer=AutoTokenizer.from_pretrained(c['model'],local_files_only=True)
    encoded={};token_issues=[]
    for rows in splits.values():
        for row in rows:
            try: encoded[row['id']]=encode_training(c,tokenizer,row)
            except (ValueError,KeyError,TypeError) as error:
                token_issues.append({'id':row.get('id'),'issue':'encoding_failed','detail':str(error)})
    result=inspect_records(c,splits,encoded)
    result['issues'].extend(token_issues)
    boundary=WORKSPACE/'datasets/staging/linux-assistant-1000-250-250-20261022/old-sealed-groups.json'
    reserved=json.loads(boundary.read_text('utf8')).get('groups',{}) if boundary.is_file() else {}
    for split,rows in splits.items():
        for row in rows:
            source_groups={row.get('group')}
            for source in row.get('sources',[]):
                if isinstance(source,dict):
                    path=str(source.get('path','')).replace('\\','/')
                    if '/tldr-pages-tldr/' in path:
                        source_groups.add('tldr:'+path.split('/tldr-pages-tldr/',1)[1])
            for group in source_groups:
                old=reserved.get(group)
                if old and (old.get('split')=='test' and split!='test' or old.get('split')=='dev' and split=='train'):
                    result['issues'].append({'id':row['id'],'issue':'historical_holdout_group_reused','source_group':group,'previous_split':old.get('split')})
    if pilot:
        for cat in CATEGORIES:
            actual=result['category_counts']['train'].get(cat,0)
            if actual!=10:
                result['issues'].append({'issue':'pilot_category_quota','category':cat,'expected':10,'actual':actual})
    else:
        contract=c['dataset_contract']
        for cat,expected in contract['training_scenarios'].items():
            actual=result['category_counts']['train'].get(cat,0)
            if actual!=expected:
                result['issues'].append({'issue':'train_category_quota','category':cat,'expected':expected,'actual':actual})
        general=Counter(tag for row in splits['train'] if row.get('category')=='general'
                        for tag in row.get('skill_tags',[]) if tag in contract.get('training_general_subtypes',{}))
        for subtype,expected in contract.get('training_general_subtypes',{}).items():
            if general[subtype]!=expected:
                result['issues'].append({'issue':'train_general_quota','subtype':subtype,'expected':expected,'actual':general[subtype]})
        if result['training_multiturn']<contract['multiturn_min']:
            result['issues'].append({'issue':'insufficient_multiturn','actual':result['training_multiturn'],
                                     'expected':contract['multiturn_min']})
        training=splits['train']
        command_log=sum(bool(set(row.get('skill_tags',[]))&{'command_generation','log_troubleshooting'}) for row in training)
        documents=sum('document_explanation' in row.get('skill_tags',[]) for row in training)
        result.update(training_command_or_log_records=command_log,training_document_records=documents)
        if command_log<contract['command_and_troubleshooting_min']:
            result['issues'].append({'issue':'insufficient_command_or_log_skill_coverage','actual':command_log,
                                     'expected':contract['command_and_troubleshooting_min']})
        if documents>contract['document_records_max']:
            result['issues'].append({'issue':'document_record_cap_exceeded','actual':documents,
                                     'maximum':contract['document_records_max']})
        fraction=result['document_supervised_token_fraction']
        if fraction is not None and fraction>contract['document_supervised_token_fraction_max']+1e-12:
            result['issues'].append({'issue':'document_supervised_token_share_exceeded','actual':fraction,
                                     'maximum':contract['document_supervised_token_fraction_max']})
        for split in ('dev','test'):
            expected=contract['evaluation']['total']
            if len(splits[split])!=expected:
                result['issues'].append({'issue':'evaluation_total','split':split,'expected':expected,'actual':len(splits[split])})
            for field,wanted in (('evaluation_capability',contract['evaluation']['capabilities']),
                                 ('evaluation_subskill',contract['evaluation']['general_subtypes'])):
                counts=Counter(r.get(field) for r in splits[split])
                for key,n in wanted.items():
                    if counts[key]!=n:
                        result['issues'].append({'issue':'evaluation_quota','split':split,'field':field,'value':key,'expected':n,'actual':counts[key]})
        all_rows=[r for rows in splits.values() for r in rows]
        for row in all_rows:
            if not isinstance(row.get('source_atom_id'),str) or not row['source_atom_id']:
                result['issues'].append({'id':row.get('id'),'issue':'missing_source_atom_id'})
        for name,label in (('content-review.jsonl','content_review'),('source-review.jsonl','source_review')):
            path=data/name
            result['issues'].extend(review_coverage(all_rows,jsonl(path) if path.is_file() else [],label))
    result.update(utc=utc(),engineering_passed=not result['issues'],
                  content_review_status='not_granted_by_engineering_audit',
                  training_approved=False,model_loaded=False,
                  input_files={p.name:sha256(p) for p in data.glob('*.jsonl')},
                  maximum_sequence_tokens=max((len(e['input_ids']) for e in encoded.values()),default=0))
    def distribution(values):
        values=sorted(values)
        if not values:return {'count':0,'total':0}
        def quantile(p):
            index=(len(values)-1)*p;low=int(index);high=min(low+1,len(values)-1)
            return values[low]+(values[high]-values[low])*(index-low)
        return {'count':len(values),'total':sum(values),'min':values[0],
                'median':quantile(.5),'p95':quantile(.95),'max':values[-1]}
    result['token_distributions']={name:distribution(values) for name,values in {
        'full_sequence':[len(e['input_ids']) for e in encoded.values()],
        'context':[next(i for i,t in enumerate(e['labels']) if t!=-100) for e in encoded.values()],
        'supervised_answer':[sum(t!=-100 for t in e['labels']) for e in encoded.values()]}.items()}
    write_json(output,result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--pilot',action='store_true')
    a=p.parse_args(); result=audit(config(a.config),a.data.resolve(),a.output,a.pilot)
    print(json.dumps({k:result[k] for k in ('engineering_passed','split_counts','category_counts','training_multiturn','maximum_sequence_tokens')},ensure_ascii=False))
    raise SystemExit(0 if result['engineering_passed'] else 2)


if __name__=='__main__':main()
