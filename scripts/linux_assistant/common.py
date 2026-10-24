"""Lightweight shared I/O, immutable manifests and conservative rubric screening."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import re
from scripts.common import WORKSPACE, ROOT, sha256, write_json

DEFAULT_CONFIG = ROOT / 'configs/linux-assistant-qwen35-2b-quality.json'


def config(path=DEFAULT_CONFIG):
    value = json.loads(Path(path).read_text(encoding='utf8'))
    for key in ('model', 'comparison_model', 'data', 'runs', 'report', 'export'):
        value[key] = (WORKSPACE / value[key]).resolve()
    return value


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf8').splitlines() if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf8')


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def review_digest(decision):
    """Accept either historical field name, but never conflicting hash aliases."""
    values=[decision.get(name) for name in ('record_sha256','final_row_sha256') if decision.get(name)]
    if len(set(values))>1:raise ValueError('Conflicting review digest aliases: '+str(decision.get('id')))
    return values[0] if values else None


def training_rows(c):
    """Read both declared training splits; never load development or sealed tests."""
    manifest = json.loads((c['data'] / 'manifest.json').read_text(encoding='utf8'))
    names = manifest.get('data_splits', [])
    if 'train.jsonl' not in names:
        raise ValueError('Manifest must explicitly declare the task training split')
    return [row for name in ('train.jsonl', 'general-train.jsonl') if name in names
            for row in jsonl(c['data'] / name)]


def verify_model(path, expected_revision=None):
    manifest = json.loads((path / 'download_manifest.json').read_text(encoding='utf8'))
    if expected_revision and manifest['hf_revision'] != expected_revision:
        raise ValueError('Unexpected base-model revision')
    for row in manifest['files']:
        if sha256(path / row['name']) != row['sha256']:
            raise ValueError(f'Model file checksum mismatch: {row["name"]}')
    return manifest['hf_revision']


def verify_data(c):
    manifest = json.loads((c['data'] / 'manifest.json').read_text(encoding='utf8'))
    if manifest.get('dataset_schema') != 'lubanlm.linux-assistant-dataset.v2':
        raise ValueError('Legacy or untyped Linux-assistant dataset manifest rejected')
    exploration=manifest.get('exploratory_training',{})
    scoped_exploration=(manifest.get('release_status')=='partial' and
        c.get('experiment_scope')=='reviewed_small_scale_exploration' and
        exploration.get('scope')=='reviewed_small_scale_exploration' and
        exploration.get('user_choice')==2 and exploration.get('coverage_incomplete') is True and
        exploration.get('pending_records_excluded') is True and manifest.get('training_approval') is True)
    if manifest.get('release_status') != 'complete' and not scoped_exploration:
        raise ValueError('Partial Linux-assistant dataset release is blocked from baseline and training')
    if manifest.get('content_review_status') != 'complete':
        raise ValueError('Dataset is not fully reviewed; baseline and training are blocked')
    for name, digest in manifest['files'].items():
        if sha256(c['data'] / name) != digest:
            raise ValueError(f'Frozen data changed: {name}')
    source_review=c['data']/'source-review.jsonl'
    review_lock_path=c['data']/'source-review-lock.json'
    if not review_lock_path.is_file() or not source_review.is_file():
        raise ValueError('Pinned source-review ledger and lock are required')
    locked=json.loads(review_lock_path.read_text(encoding='utf8'))
    if (locked.get('review_schema') != 'lubanlm.source-review.v2' or
            locked.get('status') != 'complete' or sha256(source_review)!=locked.get('sha256')):
        raise ValueError('Frozen source review is missing or changed')
    review_path=c['data']/manifest.get('content_review_file','')
    if not review_path.is_file() or sha256(review_path)!=manifest.get('content_review_sha256'):
        raise ValueError('Independent content-review ledger is missing or changed')
    from scripts.linux_assistant.dataset_quality import split_issues
    split_names=manifest.get('data_splits')
    if not isinstance(split_names,list) or not split_names:
        raise ValueError('Manifest has no explicit data split list')
    splits={name[:-6].replace('-', '_'):jsonl(c['data']/name) for name in split_names}
    all_rows=[r for rows in splits.values() for r in rows]
    issues=split_issues(splits)
    if issues:
        raise ValueError('Dataset structural/hidden-constraint audit failed: '+json.dumps(issues[:5],ensure_ascii=False))
    source_decisions=jsonl(source_review)
    source_by_id={r.get('id'):r for r in source_decisions}
    if (len(source_by_id)!=len(source_decisions) or set(source_by_id)!={r.get('id') for r in all_rows} or
            locked.get('review_count')!=len(all_rows)):
        raise ValueError('Pinned source-review ledger does not exactly cover all split rows')
    for row in all_rows:
        decision=source_by_id[row['id']]
        author=row.get('created_by') or row.get('evidence',{}).get('created_by')
        if (decision.get('decision')!='accept' or not decision.get('reviewer_id') or
            decision.get('reviewer_id')==author or not decision.get('reason') or
            review_digest(decision)!=fingerprint(row) or
            decision.get('source_verified') is not True or decision.get('source_hash_verified') is not True or
            decision.get('license_reviewed') is not True or not decision.get('license') or
            decision.get('safety_reviewed') is not True or
            decision.get('safety_decision') not in ('clear','bounded') or
            (decision.get('safety_decision')=='bounded' and not decision.get('safety_scope'))):
            raise ValueError('Independent source/license/safety review missing or stale: '+row['id'])
    decisions=jsonl(review_path)
    by_id={r.get('id'):r for r in decisions}
    if len(by_id)!=len(decisions) or set(by_id)!={r.get('id') for r in all_rows}:
        raise ValueError('Independent content-review ledger does not exactly cover all split rows')
    for row in all_rows:
        decision=by_id[row['id']]
        author=row.get('created_by') or row.get('evidence',{}).get('created_by')
        if (decision.get('decision')!='accept' or not decision.get('reviewer_id') or
            decision.get('reviewer_id')==author or not decision.get('reason') or
            review_digest(decision)!=fingerprint(row)):
            raise ValueError('Independent content review missing or stale: '+row['id'])
    manifest={**manifest,'source_review_sha256':locked['sha256'],
              'content_review_sha256':manifest['content_review_sha256']}
    return manifest


def screen(row, answer):
    """Necessary-condition screening only. A regex pass is NOT semantic acceptance."""
    if 'grading' in row:
        from scripts.linux_assistant.semantic_grading import format_check
        ok,reason=format_check(answer,row['grading']['format'])
        return dict(screen_pass=None if ok else False,failed_conditions=[] if ok else [reason],
            semantic_review='pending',accepted=None,screen_scope='format_only_content_unscored',
            format_compliant=ok,content_correct=None,steps_complete=None)
    text = answer.strip()
    reasons = []
    for criterion in row['rubric'].split(';'):
        if criterion.startswith('EXACT:'):
            # Raw strict matching: surrounding whitespace matters for explicit exact tasks.
            ok = answer == criterion[6:]
        elif criterion.startswith('JSON:'):
            expected = {}
            for part in criterion[5:].split(','):
                k, v = part.split('=', 1)
                try:
                    v = json.loads(v)
                except json.JSONDecodeError:
                    pass
                expected[k] = v
            try:
                actual = json.loads(answer)
                ok = type(actual) is dict and actual.keys() == expected.keys() and all(
                    type(actual[k]) is type(v) and actual[k] == v for k, v in expected.items())
            except (ValueError, TypeError):
                ok = False
        elif criterion.startswith('MAXLINES:'):
            ok = len(text.splitlines()) <= int(criterion.split(':')[1])
        else:
            ok = re.search(criterion, text, flags=re.I) is not None
        if not ok:
            reasons.append(criterion)
    return dict(screen_pass=bool(text) and not reasons, failed_conditions=reasons,
                semantic_review='pending', accepted=None)


def gate(baseline_task, candidate_task, baseline_general, candidate_general, thresholds,
         baseline_content=None, candidate_content=None):
    if baseline_task < thresholds['task_min']:
        task_ok = (candidate_task + 1e-9 >= thresholds['task_min'] and
                   candidate_task - baseline_task + 1e-9 >= thresholds['task_gain'])
    else:
        task_ok = ((1 - candidate_task) <= (1 - baseline_task) *
                   (1 - thresholds['error_reduction']) + 1e-9)
    content_ok=(baseline_content is None and candidate_content is None) or (
        baseline_content is not None and candidate_content is not None and
        baseline_content-candidate_content<=thresholds['general_max_drop']+1e-9)
    return bool(task_ok and content_ok and baseline_general - candidate_general <= thresholds['general_max_drop'] + 1e-9)


def reviewed_scores(rows):
    if any(r.get('reconstruction') and r.get('replay_verified') is not True for r in rows):
        raise ValueError('Reconstructed stop prefixes require successful independent-scenario GPU replay')
    if any(type(row.get('accepted')) is not bool or not row.get('review_reason') for row in rows):
        raise ValueError('All answers require a documented semantic decision; regex scores cannot select a model')
    for row in rows:
        if 'grading' in row:
            from scripts.linux_assistant.semantic_grading import validate_model_review
            validate_model_review(row,row)
            if row['kind']=='general' and row.get('content_correct')!=row['axes']['content_correct']:
                raise ValueError('General content score conflicts with semantic axis')
        if row['accepted']:
            for criterion in row.get('failed_conditions',[]):
                if criterion.startswith(('EXACT:','JSON:','MAXLINES:')) or not row.get('equivalents',{}).get(criterion):
                    raise ValueError('Accepted answer is missing a required condition or its documented equivalent: '+row['id']+' '+criterion)
    def rate(kind):
        selected = [r for r in rows if r['kind'] == kind]
        if not selected:
            raise ValueError(f'Missing evaluation category {kind}')
        return sum(r['accepted'] for r in selected) / len(selected)
    general=[r for r in rows if r['kind']=='general']
    def conclusion(row):
        if 'final_conclusion_correct' in row:return row.get('final_conclusion_correct')
        return row.get('content_correct')
    if any(type(conclusion(r)) is not bool for r in general):
        raise ValueError('General answers require a separate factual/content decision to avoid format hiding capability loss')
    return {'task': rate('task'), 'general': rate('general'),
            'general_content':sum(conclusion(r) for r in general)/len(general)}


def verify_final_rows(questions,raw,reviewed):
    expected=[q['id'] for q in questions]
    if len(set(expected))!=len(expected) or [r['id'] for r in raw]!=expected or [r['id'] for r in reviewed]!=expected:
        raise ValueError('Final results must exactly cover the locked question order, without duplicates')
    for question,original,decision in zip(questions,raw,reviewed):
        if question.get('grading')!=original.get('grading') or original.get('grading')!=decision.get('grading'):
            raise ValueError('Final semantic scoring contract changed: '+question['id'])
        for key in ('messages','kind','group','category','rubric','reference'):
            if original[key]!=question[key]:raise ValueError('Final input differs from locked question: '+question['id'])
        for key in ('id','answer','messages','kind','group','category','rubric','reference','input_ids','generated_ids','model_sha256','final_lock_sha256'):
            if original.get(key)!=decision.get(key):raise ValueError('Review changed actual final output: '+question['id'])
