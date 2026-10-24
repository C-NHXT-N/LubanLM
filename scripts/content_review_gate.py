"""Bind one-time human sample inspection to exact data, not a model verdict."""
from collections import Counter
from pathlib import Path
from scripts.common import read_json, sha256
from scripts.policy_versions import data_identity

def policy_identity(plan_path,kind='pretrain'):
    return data_identity(plan_path,kind)

def review_artifacts(plan,kind):
    data=Path(plan['data_root']);runs=Path(plan['runs_root'])
    if kind=='pretrain':
        return runs/'reports/pretrain-content-samples.json',{s:sha256(data/f'processed/pretrain-{s}/manifest.json') for s in ('train','validation')}
    folder=data/'processed'/plan['sft']['data_version']
    return runs/'reports/sft_semantic_samples.json',{s:sha256(folder/f'{s}-tokens.jsonl') for s in ('train','validation','test')}

def verify_content_review(plan_path,kind):
    plan=read_json(plan_path);runs=Path(plan['runs_root'])
    sample_path,artifacts=review_artifacts(plan,kind)
    review=read_json(runs/f'reports/{kind}_content_review.json')
    if review.get('policy_identity')!=policy_identity(plan_path,kind) or review.get('samples_sha256')!=sha256(sample_path) or review.get('artifact_sha256')!=artifacts:
        raise ValueError('Content review refers to stale data or policy')
    samples=read_json(sample_path)
    if isinstance(samples,dict):samples=samples['samples']
    minimum=200 if kind=='pretrain' else 600
    keys={(r['group'],r['split']) for r in samples}
    marks=review['marks'];marked={(r['group'],r['split']) for r in marks}
    if len(samples)<minimum or len(keys)!=len(samples) or marked!=keys or len(marks)!=len(keys):raise ValueError('Sample inspection incomplete')
    if any(r['result'] not in ('pass','minor','reject') for r in marks):raise ValueError('Invalid sample decision')
    failures=[r for r in marks if r['result']=='reject']
    if any(not r.get('reason') for r in failures):raise ValueError('Missing rejection reason')
    if failures:raise ValueError('Known rejected samples must be removed and selection/review rebuilt before training')
    if len(failures)>.05*len(marks) or any(n>=3 for n in Counter(r['reason'] for r in failures).values()):raise ValueError('Repeated content defects require source/rule repair')
    return dict(inspected=len(marks),rejected=len(failures),minor=sum(r['result']=='minor' for r in marks),samples_sha256=sha256(sample_path),artifact_sha256=artifacts,factual_correctness_of_entire_dataset_not_certified=True)
