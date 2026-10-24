"""Apply explicit, content-bound offline decisions; never infer semantic scores."""
import argparse
import json
from pathlib import Path
from scripts.common import read_json,sha256,utc
from scripts.shortchat_io import write_json
from scripts.shortchat_common import digest
from scripts.shortchat_preparation import invalidate,validate_boundary_review
from scripts.prepare_shortchat import Builder


def apply_boundary(config,decisions_path):
    b=Builder(config)
    if not b.db.execute("SELECT 1 FROM cursors WHERE name='reuse-qualified-v2-v3' AND complete=1").fetchone():
        b.db.close();raise ValueError('Candidate reclassification is still running')
    decisions=read_json(decisions_path)
    if len({r['id'] for r in decisions})!=len(decisions):raise ValueError('Duplicate review decisions')
    bad_families=set();used=[]
    for mark in decisions:
        if any(type(mark.get(k)) is not bool for k in ('category_correct','usable','severe_error')) or not mark.get('reason'):
            raise ValueError('Explicit semantic decisions required')
        found=b.db.execute('SELECT family,payload FROM candidates WHERE id=?',(mark['id'],)).fetchone()
        if found:
            r=json.loads(found[1])
            if digest(r['messages'])!=mark['messages_sha256']:raise ValueError('Reviewed content has changed')
            used.append(mark)
            if mark['severe_error'] or not mark['category_correct'] or not mark['usable']:bad_families.add(found[0])
    if bad_families:
        invalidate(b,'explicit public boundary semantic review rejected families')
        (b.folder/'preparation-state.json').unlink(missing_ok=True)
        for family in bad_families:
            b.db.execute('INSERT OR IGNORE INTO blocked VALUES(?)',(family,));b.db.execute('DELETE FROM candidates WHERE family=?',(family,))
        b.db.commit();b.select()
    write_json(b.runs/'reports/public_boundary_corrections.json',dict(policy=b.policy,used_decisions=used,removed_families=sorted(bad_families),updated_utc=utc()))
    samples=read_json(b.folder/'public-boundary-review-samples.json');by_id={r['id']:r for r in decisions}
    applied=[];pending=[]
    for r in samples:
        mark=by_id.get(r['id'])
        if mark is None:pending.append(r);continue
        if digest(r['messages'])!=mark['messages_sha256']:raise ValueError('Reviewed sample changed')
        if mark['severe_error'] or not mark['usable'] or not mark['category_correct']:raise ValueError('Rejected family remains after removal')
        applied.append(mark)
    result=dict(policy=b.policy,samples_sha256=sha256(b.folder/'public-boundary-review-samples.json'),marks=applied)
    write_json(b.folder/'public-boundary-review-pending.json',pending)
    if pending:
        (b.folder/'public-boundary-review-marks.json').unlink(missing_ok=True)
        write_json(b.folder/'public-boundary-review-marks.partial.json',result)
    else:
        write_json(b.folder/'public-boundary-review-marks.json',result);validate_boundary_review(b.folder,b.policy)
    print(json.dumps(dict(reviewed=len(applied),pending=len(pending),removed_families=len(bad_families)),ensure_ascii=False));b.db.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--boundary-decisions',required=True);args=p.parse_args()
    apply_boundary(args.plan,args.boundary_decisions)

if __name__=='__main__':main()
