"""Verify separately authored pilot reviews; never approve rows automatically."""
from __future__ import annotations
import argparse
from collections import Counter
import json
import math
from pathlib import Path

from scripts.common import sha256,utc,write_json
from scripts.linux_assistant.common import jsonl,fingerprint
from scripts.linux_assistant.preparation_audit import CATEGORIES,review_coverage


def qualify(rows, primary, source, cross, engineering):
    issues=[]
    if not engineering.get('engineering_passed'):
        issues.append({'issue':'engineering_audit_not_passed'})
    if engineering.get('row_fingerprints')!={r['id']:fingerprint(r) for r in rows}:
        issues.append({'issue':'engineering_audit_stale'})
    if Counter(r.get('category') for r in rows)!=Counter({c:10 for c in CATEGORIES}):
        issues.append({'issue':'pilot_needs_exactly_ten_each_category'})
    issues.extend(review_coverage(rows,primary,'primary'))
    issues.extend(review_coverage(rows,source,'source'))
    by_source={r.get('id'):r for r in source}
    for row in rows:
        decision=by_source.get(row['id'],{})
        if (not all(decision.get(k) is True for k in
                    ('source_verified','source_hash_verified','license_reviewed','safety_reviewed')) or
                not decision.get('license') or decision.get('safety_decision') not in ('clear','bounded') or
                decision.get('safety_decision')=='bounded' and not decision.get('safety_scope')):
            issues.append({'id':row['id'],'issue':'source_license_safety_review_incomplete'})
    primary_reviewers={r.get('id'):{r.get('reviewer_id')} for r in primary}
    cross_ids={r.get('id') for r in cross}
    selected=[r for r in rows if r['id'] in cross_ids]
    if len(selected)<math.ceil(.1*len(rows)):
        issues.append({'issue':'crossreview_coverage_below_ten_percent'})
    if {r['category'] for r in selected}!=set(CATEGORIES):
        issues.append({'issue':'crossreview_not_stratified_across_categories'})
    if len(selected)!=len(cross_ids):
        issues.append({'issue':'crossreview_contains_unknown_record'})
    issues.extend(review_coverage(selected,cross,'cross',primary_reviewers))
    return {'utc':utc(),'pilot_method_qualified':not issues,'issues':issues,
            'training_approved':False,'expansion_allowed':not issues,
            'review_mode':'independent_model_review_not_human_expert_review',
            'records':len(rows),'crossreview_records':len(selected)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','primary','source','cross','engineering','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    result=qualify(jsonl(a.data),jsonl(a.primary),jsonl(a.source),jsonl(a.cross),
                   json.loads(a.engineering.read_text('utf8')))
    result['input_sha256']={name:sha256(getattr(a,name)) for name in
                           ('data','primary','source','cross','engineering')}
    result['input_paths']={name:str(getattr(a,name).resolve()) for name in
                          ('data','primary','source','cross','engineering')}
    write_json(a.output,result)
    print(json.dumps(result,ensure_ascii=False))
    raise SystemExit(0 if result['pilot_method_qualified'] else 2)


if __name__=='__main__':main()
