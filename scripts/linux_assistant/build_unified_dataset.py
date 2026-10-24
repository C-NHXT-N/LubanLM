"""Combine assistant data into one categorized set while preserving evaluation splits."""
from __future__ import annotations
import hashlib
import json
from collections import Counter
from scripts.common import WORKSPACE,write_json
from scripts.linux_assistant.common import jsonl, write_jsonl, fingerprint

OLD = WORKSPACE / 'datasets/linux-assistant-v1-reviewed'
COMBINED = WORKSPACE / 'datasets/linux-assistant-v2-candidate'
OUT = WORKSPACE / 'datasets/linux-assistant-unified'
LABELS = {'environment': '环境检查', 'linux': 'Linux操作',
          'deployment': '部署流程', 'troubleshooting': '故障排查',
          'interaction': '交互规范', 'general': '通用能力'}


def normalize_group(group):
    category, _, slug = group.partition('/')
    if slug.startswith('v2-'):
        slug = slug[3:]
    return f'{category}/{slug}'


def normalize_row(row, new=False):
    row = dict(row)
    row['group'] = normalize_group(row['group'])
    row['category_label'] = LABELS[row.get('category', 'general')]
    # Some scenario records store the answer only in `reference`; materialize
    # it as the supervised assistant turn so training and eval use the same target.
    if row.get('messages') and (row['messages'][-1].get('role') != 'assistant' or
                                not row['messages'][-1].get('content', '').strip()):
        row['messages'] = [*row['messages'], {'role': 'assistant', 'content': row['reference']}]
    return row


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def commit_jsonl(path,rows):
    temporary=path.with_name(path.name+'.building')
    write_jsonl(temporary,rows)
    temporary.replace(path)


def main():
    raise RuntimeError('This legacy builder is retired: its v1/v2 source corpora were removed. Use the verified-source rebuild pipeline only.')
    if (OUT/'curation/quality-contract.json').exists():
        from scripts.linux_assistant.build_quality_dataset import build
        return build()
    old_train = jsonl(OLD / 'train.jsonl')
    candidate_train = jsonl(COMBINED / 'train.jsonl')
    review = jsonl(COMBINED / 'source-review.jsonl')
    candidate_groups = {normalize_group(r['group']) for r in review
                        if r.get('review_status') == 'candidate_author_review'}
    new_train = [r for r in candidate_train if r.get('source_review') == 'candidate-author-review']
    splits = {'train': [normalize_row(r) for r in old_train] +
                       [normalize_row(r, True) for r in new_train]}
    for split in ('dev', 'test'):
        splits[split] = [normalize_row(r, normalize_group(r['group']) in candidate_groups)
                         for r in jsonl(COMBINED / f'{split}.jsonl')]
    for split in ('general-dev', 'general-test'):
        splits[split] = [normalize_row(r) for r in jsonl(COMBINED / f'{split}.jsonl')]

    for row in review:
        row['group'] = normalize_group(row['group'])
        category = row['group'].partition('/')[0]
        if category in LABELS:
            row['category_label'] = LABELS[category]

    curation=OUT/'curation'
    frozen_path=curation/'frozen-evaluation.json'
    frozen=json.loads(frozen_path.read_text(encoding='utf8')) if frozen_path.exists() else {}
    for name,digest in frozen.items():
        if sha(OUT/name)!=digest:raise ValueError('Frozen evaluation changed: '+name)
        if jsonl(OUT/name)!=splits[name.removesuffix('.jsonl')]:
            raise ValueError('Rebuild would alter frozen evaluation: '+name)
    original_train=list(splits['train'])
    quarantine=json.loads((curation/'training-quarantine.json').read_text(encoding='utf8')) if (curation/'training-quarantine.json').exists() else {}
    revisions=json.loads((curation/'training-revisions.json').read_text(encoding='utf8')) if (curation/'training-revisions.json').exists() else {}
    quarantined=[r for r in splits['train'] if r['group'] in quarantine]
    splits['train']=[r for r in splits['train'] if r['group'] not in quarantine]
    for row in splits['train']:
        if row['group'] in revisions:
            revision=revisions[row['group']]
            row['reference']=revision['reference'];row['rubric']=revision['rubric'];row['facts']=[row['reference']]
            row['messages']=[*row['messages'][:-1],{'role':'assistant','content':row['reference']}]
    review=[r for r in review if r['group'] not in quarantine]
    for row in review:
        if row['split']=='train' and row['group'] in revisions:
            original=row['evidence']['reference_sha256']
            row['reference']=revisions[row['group']]['reference']
            row['evidence']={**row['evidence'],'previous_reference_sha256':original,
                'reference_sha256':hashlib.sha256(row['reference'].encode()).hexdigest(),
                'revision_note':'Changed reviewer-style instruction into an actual grounded assistant reply; author review only.'}
    additions=jsonl(curation/'train-additions.jsonl') if (curation/'train-additions.jsonl').exists() else []
    splits['train'].extend(normalize_row(r) for r in additions)
    added_reviews=jsonl(curation/'source-additions.jsonl') if (curation/'source-additions.jsonl').exists() else []
    for row in added_reviews:row['category_label']=LABELS[row['group'].partition('/')[0]]
    review.extend(added_reviews)

    # Content-review overlays apply only to training; held-out bytes stay frozen.
    overlay={};overlay_path=curation/'content-corrections.json'
    if overlay_path.exists():
        overlay=json.loads(overlay_path.read_text(encoding='utf8'))
        kept=[]
        for row in splits['train']:
            decision=overlay['groups'].get(row['group'],{})
            if decision.get('exclude'):
                quarantined.append(row);continue
            correction=overlay.get('records',{}).get(row['id'],decision)
            if 'reference' in correction:
                row['reference']=correction['reference'];row['facts']=[row['reference']]
                row['messages'][-1]['content']=row['reference']
            if 'rubric' in correction:row['rubric']=correction['rubric']
            if 'sources' in correction:row['sources']=correction['sources']
            if 'messages' in correction:row['messages']=correction['messages']
            if 'quality_criteria' in row:
                row['quality_criteria']['required_facts']=[row['reference']]
            row['group']=decision.get('merge_into',row['group'])
            kept.append(row)
        splits['train']=kept
        old_review={r['group']:r for r in review if r['split']=='train'}
        grouped={}
        for row in kept:grouped.setdefault(row['group'],[]).append(row)
        review=[r for r in review if r['split']!='train']
        for group,batch in grouped.items():
            prior=old_review[group]
            references=list(dict.fromkeys(r['reference'] for r in batch))
            review.append({**prior,'reference':references[0],'example_references':references,
                'source_urls':sorted({s for r in batch for s in r['sources']}),
                'evidence':{**prior['evidence'],
                    'reference_sha256':hashlib.sha256(references[0].encode()).hexdigest(),
                    'reference_hashes':[hashlib.sha256(v.encode()).hexdigest() for v in references],
                    'content_corrections_sha256':sha(overlay_path)}})

    counts = {split: len(rows) for split, rows in splits.items()}
    scenarios = {split: len({r['group'] for r in rows}) for split, rows in splits.items()}
    categories = {split: dict(sorted(Counter(r['category_label'] for r in rows).items()))
                  for split, rows in splits.items()}
    per_group = {split: dict(sorted(Counter(r['group'] for r in rows).items()))
                 for split, rows in splits.items()}
    task_groups = {s: {r['group'] for r in splits[s]} for s in ('train', 'dev', 'test')}
    overlap = {f'{a}_{b}': len(task_groups[a] & task_groups[b])
               for a, b in (('train', 'dev'), ('train', 'test'), ('dev', 'test'))}
    records = [r for rows in splits.values() for r in rows]
    ids = [r['id'] for r in records]
    questions = [next(m['content'] for m in reversed(r['messages']) if m['role'] == 'user')
                 for r in records]
    review_groups = {r['group'] for r in review}
    missing_review = sorted({r['group'] for s in ('train', 'dev', 'test') for r in splits[s]} - review_groups)
    bad_messages = [r['id'] for r in records if not all(
        isinstance(m, dict) and {'role', 'content'} <= set(m) for m in r.get('messages', []))]
    bad_targets = [r['id'] for r in records if not r.get('messages') or
                   r['messages'][-1].get('role') != 'assistant' or
                   not r['messages'][-1].get('content', '').strip()]
    if len(ids) != len(set(ids)) or len(questions) != len(set(questions)):
        raise ValueError('Duplicate record ID or user question')
    if any(overlap.values()) or bad_messages or bad_targets or missing_review:
        raise ValueError(f'Integrity check failed: overlap={overlap}, bad_messages={bad_messages[:3]}, bad_targets={bad_targets[:3]}, missing_review={missing_review[:3]}')

    # Validate complete targets and source identities before replacing any active training file.
    from scripts.linux_assistant.common import screen
    review_by_group={r['group']:r for r in review}
    if len(review_by_group)!=len(review):raise ValueError('Duplicate source review groups')
    for row in splits['train']:
        if not screen(row,row['reference'])['screen_pass']:raise ValueError('Reference/rubric contradiction: '+row['id'])
        evidence=review_by_group[row['group']]['evidence']
        hashes=evidence.get('reference_hashes',[evidence['reference_sha256']])
        if hashlib.sha256(row['reference'].encode()).hexdigest() not in hashes:
            raise ValueError('Reference/source review mismatch: '+row['id'])
    OUT.mkdir(parents=True,exist_ok=True)
    for split,rows in splits.items():
        name=split+'.jsonl'
        if name not in frozen:commit_jsonl(OUT/name,rows)
    commit_jsonl(OUT/'source-review.jsonl',review)
    if quarantine:commit_jsonl(curation/'quarantined-training.jsonl',quarantined)

    manifest = {
        'dataset': 'linux-assistant-unified', 'status': 'combined_candidate_pending_review',
        'files': {str(p.relative_to(OUT)).replace('\\','/'):sha(p) for p in sorted([*OUT.glob('*.jsonl'),*curation.glob('*')]) if p.is_file()},
        'record_counts': counts, 'scenario_counts': scenarios, 'category_counts': categories,
        'category_labels': LABELS, 'scenario_group_record_counts': per_group,
        'total_records': sum(counts.values()), 'task_split_group_overlap': overlap,
        'duplicate_ids': 0, 'duplicate_questions': 0,
        'source_review_records': len(review),
        'source_review_status_counts': dict(sorted(Counter(r['review_status'] for r in review).items())),
        'new_records_requiring_independent_review': sum(r['review_status'] == 'candidate_author_review' for r in review),
        'independent_review_pending_scenarios':sum(not r.get('evidence',{}).get('independent_review_completed',False) for r in review),
        'quarantined_training_records':len(quarantined),'quarantined_training_groups':len({r['group'] for r in quarantined}),
        'content_review_corrections':overlay,
        'source_checked_new_training_records':len(additions),'source_checked_new_scenarios':len(added_reviews),
        'unique_training_references':len({r['reference'] for r in splits['train']}),
        'multi_turn_training_records':sum(len(r['messages'])>2 for r in splits['train']),
        'known_semantic_overlap_policy':quarantine,'frozen_evaluation_sha256':frozen,
        'training_policy': 'Use train.jsonl only; keep development and acceptance data out of training.',
        'final_test_policy': 'Keep test.jsonl and general-test.jsonl sealed until model and scoring rules are locked.'}
    write_json(OUT/'manifest.json',manifest)
    write_json(OUT/'source-review-lock.json',
        {'sha256': sha(OUT / 'source-review.jsonl'),
         'review_status':f"{manifest['independent_review_pending_scenarios']} scenario reviews pending independent review; legacy verified labels mean author review only"})
    readme=f'''# Linux助手统一微调数据集

当前训练{counts['train']}条/{scenarios['train']}个场景，任务开发与验收各{counts['dev']}条/{scenarios['dev']}个场景，通用开发与验收各40条。开发和验收文件字节摘要保持不变。只用train.jsonl训练。

已隔离{len(quarantined)}条已知语义重叠或质量问题记录，补充{len(additions)}条来源核对样本，覆盖{len(added_reviews)}个新场景。每个新场景有两个不同观察条件/约束的案例，模拟日志不冒充板端实测。训练内容复核修正和同场景合并规则在curation/content-corrections.json。

全部{manifest['independent_review_pending_scenarios']}个任务场景尚未作独立外部复核；旧台账verified仅表示作者核对，不能等同专家审阅。本次逐条代理复核及训练门禁见../../reports/linux-assistant-training-review-20261020/；它不等于独立人工专家复核。旧数据保存在provenance快照与复核报告before/。旧提示包装变体保留用于表达增广，不视为独立知识点。详细分布见../../reports/linux-assistant-dataset/inventory.json。
'''
    (OUT / 'README.md').write_text(readme,encoding='utf8')
    report=WORKSPACE/'reports/linux-assistant-dataset';report.mkdir(parents=True,exist_ok=True)
    inventory={k:manifest[k] for k in ('dataset','status','record_counts','scenario_counts','total_records','category_counts','unique_training_references','multi_turn_training_records','frozen_evaluation_sha256')}
    inventory['category_summary']={s:{label:{'records':sum(r['category_label']==label for r in rows),
        'scenarios':len({r['group'] for r in rows if r['category_label']==label})}
        for label in sorted({r['category_label'] for r in rows})} for s,rows in splits.items()}
    write_json(report/'inventory.json',inventory)
    baseline_identity=WORKSPACE/'reports/linux-assistant-baseline-qwen35-20261020/identity.json'
    baseline_same=baseline_identity.exists() and json.loads(baseline_identity.read_text(encoding='utf8'))['rows']==fingerprint(splits['dev']+splits['general-dev'])
    audit={'before_records':len(original_train),'after_records':counts['train'],
        'before_unique_references':len({r['reference'] for r in original_train}),
        'after_unique_references':manifest['unique_training_references'],'after_scenarios':scenarios['train'],
        'new_records':len(additions),'new_scenarios':len(added_reviews),'quarantined_records':len(quarantined),
        'quarantined_groups':quarantine,'revised_answer_groups':list(revisions),
        'duplicate_ids':0,'duplicate_last_user_questions':0,'group_overlap':overlap,
        'all_training_references_pass_necessary_rubric':True,'all_training_reference_hashes_match_review':True,
        'frozen_evaluation_unchanged':all(sha(OUT/name)==digest for name,digest in frozen.items()),
        'completed_clean_baseline_inputs_still_identical':baseline_same,
        'review_status':'Author source check only; independent review and real Linux runtime checks remain pending.',
        'limitation':'Known near-duplicate cases were excluded; group-name isolation does not prove absence of all semantic overlap.'}
    write_json(report/'quality-audit.json',audit)
    print(json.dumps({'dataset': str(OUT), 'record_counts': counts,
                      'scenario_counts': scenarios, 'category_counts': categories,
                      'total_records': sum(counts.values()), 'overlap': overlap}, ensure_ascii=False))


if __name__ == '__main__':
    main()
