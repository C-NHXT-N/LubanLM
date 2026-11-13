"""Freeze independently reviewed data; never manufacture review decisions.

Creates a new, inactive release directory. Promotion to the sole active dataset
is separate so a failed preparation cannot replace the currently retained data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from scripts.common import WORKSPACE, sha256, utc, write_json
from scripts.linux_assistant.common import (config, jsonl, training_rows,review_digest,
                                          verify_data, write_jsonl)

SPLITS=('train.jsonl','general-train.jsonl','dev.jsonl','general-dev.jsonl',
        'test.jsonl','general-test.jsonl')
LEDGERS=('content-review.jsonl','source-review.jsonl','cross-review.jsonl')


def checked_workspace_path(path):
    path=Path(path).resolve()
    if not path.is_relative_to(WORKSPACE.resolve()) or path==WORKSPACE.resolve():
        raise ValueError('Release paths must stay below the workspace')
    return path


def verify_pilot_qualification(path):
    proof=json.loads(Path(path).read_text('utf8'))
    if (proof.get('pilot_method_qualified') is not True or
            proof.get('expansion_allowed') is not True or proof.get('issues')):
        raise ValueError('Pilot construction method has not qualified')
    names=('data','primary','source','cross','engineering')
    for name in names:
        pinned=proof.get('input_paths',{}).get(name)
        if not pinned or sha256(checked_workspace_path(pinned))!=proof.get('input_sha256',{}).get(name):
            raise ValueError('Pilot qualification evidence is missing or stale: '+name)
    from scripts.linux_assistant.qualify_pilot import qualify
    paths=proof['input_paths']
    replay=qualify(jsonl(paths['data']),jsonl(paths['primary']),jsonl(paths['source']),
                   jsonl(paths['cross']),json.loads(Path(paths['engineering']).read_text('utf8')))
    if not replay['pilot_method_qualified']:
        raise ValueError('Pilot qualification cannot be reproduced from its evidence')
    return proof


def freeze(c, source, output, report, qualification):
    """Bind existing per-row decisions only after all release gates pass."""
    from scripts.linux_assistant.modeling import training_data
    from scripts.linux_assistant.preparation_audit import audit
    from scripts.linux_assistant.release_contract import validate_release_contract
    from scripts.linux_assistant.review_training_data import validate_training_review
    from transformers import AutoTokenizer

    source,output,report,qualification=map(checked_workspace_path,
                                         (source,output,report,qualification))
    if c.get('experiment_scope')!='controlled_1000_skill_rebuild':
        raise ValueError('Freeze supports only controlled_1000_skill_rebuild scope')
    destinations=(source,output,report,qualification)
    for index,left in enumerate(destinations):
        for right in destinations[index+1:]:
            if left==right or left.is_relative_to(right) or right.is_relative_to(left):
                raise ValueError('Release input, output, report and qualification paths must not overlap')
    verify_pilot_qualification(qualification)
    if output.exists() or report.exists() or output==source:
        raise FileExistsError('Release and report destinations must be new directories')
    for name in SPLITS+LEDGERS:
        if not (source/name).is_file():
            raise ValueError('Release input missing: '+name)
    # Capture immutable source identities before copying. Never relabel rows,
    # rebind reviewer hashes, or infer semantic acceptance here.
    inputs={name:sha256(source/name) for name in SPLITS+LEDGERS}
    output.mkdir(parents=True)
    report.mkdir(parents=True)
    try:
        for name in inputs:
            shutil.copyfile(source/name,output/name)
            if sha256(output/name)!=inputs[name] or sha256(source/name)!=inputs[name]:
                raise ValueError('Data changed during release copy: '+name)
        manifest={
            'dataset_schema':'lubanlm.linux-assistant-dataset.v2',
            'created_utc':utc(),'release_status':'complete',
            'content_review_status':'complete','training_approval':False,
            'data_splits':list(SPLITS),'files':dict(inputs),
            'content_review_file':'content-review.jsonl',
            'content_review_sha256':inputs['content-review.jsonl'],
            'cross_review_file':'cross-review.jsonl',
            'cross_review_sha256':inputs['cross-review.jsonl'],
            'review_mode':'independent_agent_cross_review',
            'qualification_sha256':sha256(qualification),
            'source_input_directory':str(source.relative_to(WORKSPACE)),
            'public_release':False,
        }
        write_json(output/'source-review-lock.json',{
            'review_schema':'lubanlm.source-review.v2','status':'complete',
            'sha256':inputs['source-review.jsonl'],
            'review_count':sum(len(jsonl(source/name)) for name in SPLITS)})
        manifest['files']['source-review-lock.json']=sha256(output/'source-review-lock.json')
        write_json(output/'manifest.json',manifest)
        current={**c,'data':output}
        verify_data(current)
        engineering=audit(current,output,report/'engineering.json')
        if not engineering['engineering_passed']:
            raise ValueError('Preparation audit failed; inspect engineering.json')
        tokenizer=AutoTokenizer.from_pretrained(c['model'],local_files_only=True)
        rows,encoded=training_data(current,tokenizer)
        contract=validate_release_contract(current,rows,encoded)
        write_json(report/'release-contract.json',contract)
        decisions={r['id']:r for r in jsonl(output/'content-review.jsonl')}
        # A subset of an already checked, independently authored ledger; these
        # are not generated decisions and retain the original reviewer/reason.
        write_jsonl(report/'approved-training-records.jsonl',[{
            **decisions[row['id']],
            'record_sha256':review_digest(decisions[row['id']])}
            for row in rows])
        manifest['training_approval']=True
        write_json(output/'manifest.json',manifest)
        evidence_paths=[report/'engineering.json',report/'release-contract.json',qualification]
        approval={
            'approval_schema':'lubanlm.dataset-content-review.v2',
            'review_mode':'independent_agent_cross_review','passed':True,
            'scope':'controlled_1000_skill_rebuild','completed_utc':utc(),
            'reviewer':'independent per-record model reviewers; see immutable ledgers',
            'train_sha256':sha256(output/'train.jsonl'),
            'general_train_sha256':sha256(output/'general-train.jsonl'),
            'manifest_sha256':sha256(output/'manifest.json'),
            'source_review_sha256':sha256(output/'source-review.jsonl'),
            'ledger_sha256':sha256(report/'approved-training-records.jsonl'),
            'system_prompt_sha256':hashlib.sha256(c['system_prompt'].encode()).hexdigest(),
            'evidence':{p.relative_to(WORKSPACE).as_posix():sha256(p) for p in evidence_paths},
            'external_expert_review':False,'public_release':False,
        }
        write_json(report/'training-approval.json',approval)
        current['training_review']=(report/'training-approval.json').relative_to(WORKSPACE).as_posix()
        validate_training_review(current)
        result={'completed':True,'utc':utc(),'directory':str(output),
                'approval':str(report/'training-approval.json'),
                'active_dataset_replaced':False,'training_started':False,
                'input_sha256':inputs}
        write_json(report/'freeze-result.json',result)
        return result
    except Exception as error:
        # Preserve evidence, but ensure the incomplete copy fails every default
        # baseline/training entry point, even if failure happened after pinning.
        if (output/'manifest.json').is_file():
            failed=json.loads((output/'manifest.json').read_text('utf8'))
            failed.update(release_status='failed',training_approval=False)
            write_json(output/'manifest.json',failed)
        if (report/'training-approval.json').is_file():
            failed=json.loads((report/'training-approval.json').read_text('utf8'))
            failed.update(passed=False,error=str(error))
            write_json(report/'training-approval.json',failed)
        write_json(report/'freeze-result.json',{'completed':False,'utc':utc(),
                   'error':str(error),'active_dataset_replaced':False,'training_started':False})
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','source','output','report','qualification'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(freeze(config(a.config),a.source,a.output,a.report,a.qualification),ensure_ascii=False))


if __name__=='__main__':main()
