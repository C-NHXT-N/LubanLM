"""Journaled additive postfilter migration over immutable public candidate text."""
import argparse
import ast
from collections import Counter
import json
from pathlib import Path
import shutil
import sqlite3
from scripts.common import ROOT, read_json, write_json, utc, sha256
from scripts.policy_versions import sft_identity
from scripts.sft_content_checks import postcheck

ADDITION = '''            from scripts.sft_content_checks import postcheck
            item, reason = postcheck(item)
            if reason:
                count['context_'+reason] += 1
                continue
            if source.get('allowed_tasks') and item['task'] not in source['allowed_tasks']:
                count['supplement_context_task_not_requested'] += 1
                continue
            yield item'''

def run(plan_path, recovery_name='sft-context-repair-20261007'):
    from scripts.sft_disk import connect
    from scripts.resource_guard import check_resources
    from scripts.content_review_gate import verify_content_review
    plan = read_json(plan_path)
    data, runs = Path(plan['data_root']), Path(plan['runs_root'])
    recovery = (runs/'recovery'/recovery_name).resolve()
    if not recovery.is_relative_to((runs/'recovery').resolve()):
        raise ValueError('Recovery must remain in runs/recovery')
    if not (runs/'reports/maintenance.json').exists():
        raise RuntimeError('Migration requires the preparation controller to be in maintenance')
    # Only the postcheck call changed in the old parser; selection is identical.
    old = (recovery/'scripts/prepare_sft_100k.py').read_text(encoding='utf-8-sig')
    now = (ROOT/'scripts/prepare_sft_100k.py').read_text(encoding='utf-8-sig')
    if now.count(ADDITION) != 1 or (now != old and now.replace(ADDITION, '            yield item') != old):
        raise ValueError('Unexpected legacy parser mutation')
    for name in ('insert_candidate','collapse_lineage','assign_splits','finalize'):
        def body(path):
            tree = ast.parse(path.read_text(encoding='utf-8-sig'))
            return ast.dump(next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)
        if body(ROOT/'scripts/sft_disk.py') != body(recovery/'scripts/sft_disk.py'):
            raise ValueError('Selection changed: '+name)
    verify_content_review(plan_path, 'pretrain')
    policy = sft_identity(plan_path)
    out = data/'processed'/plan['sft']['data_version']
    db = connect(out/'candidates.sqlite')
    journal = sqlite3.connect(recovery/'candidate_changes.sqlite')
    journal.executescript('PRAGMA synchronous=FULL; CREATE TABLE IF NOT EXISTS changes(id TEXT PRIMARY KEY,old_task TEXT,removed_row TEXT,reason TEXT);')
    state_path = recovery/'migration.json'
    state = read_json(state_path) if state_path.exists() else None
    if state is None:
        before = read_json(recovery/'before.json')
        actual = json.loads(db.execute("SELECT value FROM meta WHERE name='policy'").fetchone()[0])
        if actual != before['policy'] or db.execute('SELECT COUNT(*) FROM candidates').fetchone()[0] != before['candidate_count']:
            raise ValueError('Candidate pool differs from the inspected snapshot')
        for src in (out/'manifest.json', runs/'reports/sft_semantic_samples.json', runs/'reports/sft_filter_report.json'):
            shutil.copy2(src, recovery/src.name)
        state = dict(started_utc=utc(),original_count=before['candidate_count'],old_policy=actual,new_policy=policy,last_rowid=0,processed=0,removed=0,reclassified=0,reasons={},status='migrating')
        write_json(state_path, state)
    elif state['new_policy'] != policy:
        raise ValueError('Postfilter policy changed mid-migration')
    if state['status']=='completed':
        if (out/'manifest.json').exists() and read_json(out/'manifest.json')['policy_identity']!=policy:
            (out/'manifest.json').unlink()
        journal.close();db.close()
        return
    supplements = {s['repo']:s.get('allowed_tasks') for s in plan['sft']['raw_sources'] if s.get('supplement')}
    reasons = Counter(state['reasons'])
    # Content edits make count-only caches unsafe; invalidate explicitly.
    db.execute("DELETE FROM meta WHERE name='quota_cache'")
    db.commit()
    while True:
        batch = db.execute('SELECT rowid,* FROM candidates WHERE rowid>? ORDER BY rowid LIMIT 512',(state['last_rowid'],)).fetchall()
        if not batch:
            break
        actions=[]
        for row in batch:
            rowid, identity = row[:2]
            payload = json.loads(row[8])
            checked, reason = postcheck(payload)
            if checked is not None and supplements.get(payload['source']) and checked['task'] not in supplements[payload['source']]:
                checked, reason = None, 'supplement_context_task_not_requested'
            if reason:
                journal.execute('INSERT OR IGNORE INTO changes VALUES (?,?,?,?)',(identity,row[5],json.dumps(row[1:],ensure_ascii=False),reason))
                actions.append(('remove', identity, reason))
            elif checked['task'] != payload['task']:
                journal.execute('INSERT OR IGNORE INTO changes VALUES (?,?,NULL,?)',(identity,payload['task'],'context_task'))
                actions.append(('task',identity,checked))
        journal.commit() # Persist recovery BEFORE changing the active pool.
        for kind, identity, value in actions:
            if kind=='remove':
                db.execute('DELETE FROM prefixes WHERE id=?',(identity,))
                db.execute('DELETE FROM lineage_info WHERE id=?',(identity,))
                db.execute('DELETE FROM candidates WHERE id=?',(identity,))
                state['removed']+=1;reasons[value]+=1
            else:
                db.execute('UPDATE candidates SET task=?,payload=? WHERE id=?',(value['task'],json.dumps(value,ensure_ascii=False),identity))
                state['reclassified']+=1
        db.commit()
        state.update(last_rowid=batch[-1][0],processed=state['processed']+len(batch),reasons=dict(reasons),updated_utc=utc())
        write_json(state_path,state)
        if state['processed']%16384==0:
            check_resources(plan,'sft');print(json.dumps({k:state[k] for k in ('processed','removed','reclassified','updated_utc')}),flush=True)
    db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)',('policy',json.dumps(policy)))
    db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)',('context_migration',json.dumps(dict(exclusions_sha256=sha256(ROOT/'configs/sft-content-exclusions-20261007.json'),recovery=str(recovery)))))
    db.commit()
    removed=journal.execute('SELECT COUNT(*) FROM changes WHERE removed_row IS NOT NULL').fetchone()[0]
    reclassified=journal.execute('SELECT COUNT(*) FROM changes WHERE removed_row IS NULL').fetchone()[0]
    remaining=db.execute('SELECT COUNT(*) FROM candidates').fetchone()[0]
    if remaining!=state['original_count']-removed:raise ValueError('Unexpected concurrent candidate mutation')
    state.update(status='completed',finished_utc=utc(),processed=state['original_count'],removed=removed,reclassified=reclassified,remaining=remaining,available=dict(db.execute('SELECT task,COUNT(*) FROM candidates GROUP BY task')))
    write_json(state_path,state)
    journal.close();db.close()
    # Exports are atomically regenerated under the new policy after migration.
    (out/'manifest.json').unlink()
    for name in ('sft_content_review.json','sft_outcome.json'):
        (runs/'reports'/name).unlink(missing_ok=True)
    write_json(runs/'reports/sft_context_migration.json',state)
    print(json.dumps(state,ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--recovery-name',default='sft-context-repair-20261007');args=p.parse_args()
    from scripts.worker_health import WorkerHealth
    with WorkerHealth(args.plan,'preparing_sft_cpu'):
        run(args.plan,args.recovery_name)
