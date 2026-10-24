"""One-time authorized switch. Supervisor must already be disabled."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import time
import psutil
from scripts.common import ROOT,read_json,write_json,sha256,utc
from scripts.policy_versions import pretrain_identity,sft_identity,digest
from scripts.worker_health import process_matches

def prefix_hash(path,length):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while length:
            block=f.read(min(length,1<<20))
            if not block:raise ValueError('Committed index truncated')
            length-=len(block);h.update(block)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);args=parser.parse_args()
    plan_path=Path(args.plan).resolve();plan=read_json(plan_path)
    data=Path(plan['data_root']).resolve();runs=Path(plan['runs_root']).resolve();reports=runs/'reports'
    if data!=Path('D:/LubanLM-Data/general-10b-teaching-v2').resolve() or runs!=Path('D:/LubanLM-Runs/general-10b-teaching-v2').resolve():raise ValueError('Unexpected switch targets')
    if shutil.disk_usage(data).free<plan['disk_free_bytes_floor']+(6<<30):raise RuntimeError('Insufficient backup/index reserve')
    state=read_json(reports/'controller.json')
    if state['status']!='preparing_10b_cpu':raise ValueError('Switch expects CPU pretrain preparation only')
    old=ROOT/'docs/run-results/teaching-repair-20261006/minimal-before'
    if pretrain_identity(plan_path)!=pretrain_identity(old/'plan.json',old):raise ValueError('Pretrain policy changed')
    maintenance=reports/'maintenance.json';write_json(maintenance,dict(status='minimal_repair',started_utc=utc()))
    stamp=time.strftime('%Y%m%d-%H%M%S');backup=runs/'recovery'/('minimal-repair-'+stamp);backup.mkdir(parents=True)
    for name in ('controller.json','supervisor.json','cpu_preparation_budget.json','preparation_progress.json'):
        if (reports/name).exists():shutil.copy2(reports/name,backup/name)
    keepers=[]
    for p in psutil.process_iter(['pid','cmdline']):
        try:
            command=p.info['cmdline'] or []
            if 'scripts.keep_awake' in command and str(state['pid']) in command:keepers.append((p.pid,p.create_time()))
        except psutil.Error:pass
    stopped=[]
    for pid,created,modules in [(state['pid'],state['process_created'],('scripts.run_general_10b',)),(state['child_pid'],state['child_created'],('scripts.prepare_general_10b',)),*[(pid,created,('scripts.keep_awake',)) for pid,created in keepers]]:
        if process_matches(pid,created,modules):
            process=psutil.Process(pid);process.terminate();process.wait(timeout=10);stopped.append(pid)
    ended=time.time();entries=read_json(reports/'cpu_preparation_budget.json')
    # Original parent-owned entry had no process identity. Settle its real interval once.
    last=entries[-1]
    if last.get('phase')=='preparing_10b_cpu' and not last.get('finished'):
        from datetime import datetime
        start=datetime.fromisoformat(last['started_utc']).timestamp()
        last.update(seconds=max(last['seconds'],ended-start),finished=True,ended_utc=utc(),migration='Settled legacy controller-owned interval at confirmed worker stop')
    write_json(reports/'cpu_preparation_budget.json',entries)
    db=sqlite3.connect((data/'dedup.sqlite').as_uri()+'?mode=ro',uri=True)
    try:
        policy=json.loads(db.execute("SELECT value FROM metadata WHERE name='policy'").fetchone()[0])
        if policy!=pretrain_identity(plan_path):raise ValueError('Stored pretrain policy mismatch')
        checkpoints=list(db.execute('SELECT name,state FROM checkpoints ORDER BY name'))
        dest=sqlite3.connect(backup/'dedup.sqlite')
        try:db.backup(dest,pages=4096)
        finally:dest.close()
    finally:db.close()
    witness=[];tokens=0;validated=0
    for name,text in checkpoints:
        row=json.loads(text);tokens+=row.get('train',{}).get('tokens',0)
        for split in ('train','validation'):
            for shard in row.get(split,{}).get('shards',[]):
                path=Path(shard['path']).resolve()
                if not path.is_relative_to(data):raise ValueError('Shard outside data root')
                if path.stat().st_size!=shard['tokens']*2 or sha256(path)!=shard['sha256']:raise ValueError('Committed shard failed verification')
                witness.append(dict(path=str(path),sha256=shard['sha256'],tokens=shard['tokens']));validated+=1
                if validated%250==0:print(json.dumps(dict(stage='verify_committed_shards',validated=validated)),flush=True)
        index=data/'processed'/name/'document-index.jsonl'
        if 'index_bytes' in row:
            witness.append(dict(path=str(index),committed_bytes=row['index_bytes'],prefix_sha256=prefix_hash(index,row['index_bytes'])))
    sft_path=data/'processed'/plan['sft']['data_version']/'candidates.sqlite'
    sft_info={}
    if sft_path.exists():
        db=sqlite3.connect(sft_path)
        try:
            counts={table:db.execute('SELECT COUNT(*) FROM '+table).fetchone()[0] for table in ('candidates','cursors','legacy')}
            dest=sqlite3.connect(backup/'sft-before-v4.sqlite')
            try:db.backup(dest,pages=4096)
            finally:dest.close()
            if counts['candidates'] or counts['cursors']:raise ValueError('SFT rescan migration requires empty active candidate/cursor tables')
            before=db.execute("SELECT value FROM meta WHERE name='policy'").fetchone()
            db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)',('policy',json.dumps(sft_identity(plan_path))));db.commit()
            sft_info=dict(counts=counts,previous_policy=json.loads(before[0]) if before else None,new_policy=sft_identity(plan_path),legacy_archive_preserved=True)
        finally:db.close()
    result=dict(created_utc=utc(),backup=str(backup),stopped_pids=stopped,committed_train_tokens=tokens,checkpoint_state_sha256=digest(checkpoints),validated_shards=validated,artifacts=witness,pretrain_policy_unchanged=True,sft_migration=sft_info,parent_sha256=sha256(plan['parent_checkpoint']),historical_ledgers={path:sha256(path) for path in state['historical_ledger_hashes']},cpu_seconds=sum(r['seconds'] for r in entries))
    if result['parent_sha256']!=state['parent_sha256'] or result['historical_ledgers']!=state['historical_ledger_hashes']:raise ValueError('Immutable parent/history changed')
    write_json(reports/'minimal_repair_switch.json',result)
    write_json(ROOT/'docs/run-results/teaching-repair-20261006/minimal-switch.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('artifacts','sft_migration')},ensure_ascii=False),flush=True)
    # Leave maintenance in place until operator verifies backup and enables the task.

if __name__=='__main__':main()
