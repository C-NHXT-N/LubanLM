"""Remove reviewed bad families only from this experiment and reseal selection."""
import argparse
from scripts.common import read_json,sha256,utc
from scripts.shortchat_io import write_json
from scripts.prepare_shortchat import Builder

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);args=p.parse_args()
    b=Builder(args.plan);folder=b.folder
    manifest=read_json(folder/'manifest.json');samples=read_json(folder/'review-samples.json');marks=read_json(folder/'review-marks.json')
    if marks['samples_sha256']!=sha256(folder/'review-samples.json') or marks['artifacts']!=manifest['artifacts']:raise ValueError('Review binding changed')
    if {r['id'] for r in samples}!={r['id'] for r in marks['marks']} or len(marks['marks'])!=len(samples):raise ValueError('Incomplete review')
    previous=b.db.execute("SELECT value FROM meta WHERE name='review_attempt'").fetchone();attempt=int(previous[0]) if previous else 0
    if attempt>=3:raise ValueError('Three repair attempts exhausted; do not train')
    bad={r['id'] for r in marks['marks'] if r['severe_error'] or not r['category_correct'] or not r['usable']}
    families={r['family_id'] for r in samples if r['id'] in bad}
    if not families:raise ValueError('No rejected families')
    from scripts.shortchat_preparation import invalidate
    invalidate(b,'failed content audit: remove problem families before refill')
    state=folder/'preparation-state.json'
    if state.exists():state.unlink()
    for family in families:
        b.db.execute('INSERT OR IGNORE INTO blocked VALUES(?)',(family,));b.db.execute('DELETE FROM candidates WHERE family=?',(family,))
    b.db.execute("INSERT OR REPLACE INTO meta VALUES('review_attempt',?)",(str(attempt+1),));b.db.commit()
    write_json(folder/f'review-marks-attempt-{attempt+1:02d}.json',marks)
    write_json(folder/f'review-samples-attempt-{attempt+1:02d}.json',samples)
    (folder/'review-marks.json').unlink(missing_ok=True)
    ok=b.select()
    write_json(b.runs/'reports/review_repair.json',dict(attempt=attempt+1,removed_families=sorted(families),reselected=ok,updated_utc=utc()))
    b.db.close()

if __name__=='__main__':main()
