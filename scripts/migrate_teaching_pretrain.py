"""Prove rule equivalence, backup SQLite, then migrate identities transactionally."""
import argparse
import json
from pathlib import Path
import sqlite3
from scripts.common import ROOT,read_json,write_json,sha256,utc
from scripts.policy_versions import pretrain_identity

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',default='configs/50m-general-pretrain-10b-teaching.json');args=p.parse_args()
    plan=read_json(args.plan);archive=ROOT/'docs/run-results/teaching-repair-20261006/pre-execution'
    previous=read_json(archive/'policy.json')
    for filename,expected in previous['code'].items():
        if sha256(archive/filename)!=expected:raise ValueError('Archived rule source changed')
    policy=pretrain_identity(args.plan)
    if policy!=pretrain_identity(archive/'plan.json',archive):raise ValueError('Pretrain rules differ; refuse reuse')
    data,runs=Path(plan['data_root']),Path(plan['runs_root']);db=sqlite3.connect(data/'dedup.sqlite',timeout=60)
    try:
        saved=json.loads(db.execute("SELECT value FROM metadata WHERE name='policy'").fetchone()[0])
        if saved==policy:print('Already migrated');return
        if saved!=previous:raise ValueError('Checkpoint policy is not the archived identity')
        if db.execute('PRAGMA quick_check').fetchone()[0]!='ok':raise ValueError('Dedup integrity failed')
        backup=runs/'recovery/pretrain-before-policy-v3.sqlite';backup.parent.mkdir(parents=True,exist_ok=True)
        if not backup.exists():
            target=sqlite3.connect(backup)
            try:db.backup(target)
            finally:target.close()
        states=[];shards=0;train=validation=0
        for name,payload in db.execute('SELECT name,state FROM checkpoints'):
            state=json.loads(payload)
            if state.get('policy_identity')!=previous:raise ValueError('Mixed checkpoint policies')
            for split in ('train','validation'):
                value=state[split]
                if value['tokens']!=sum(r['tokens'] for r in value['shards']):raise ValueError('Extent mismatch')
                for row in value['shards']:
                    path=Path(row['path'])
                    if not path.resolve().is_relative_to((data/'processed').resolve()):raise ValueError('Shard outside task data')
                    if path.stat().st_size!=2*row['tokens'] or sha256(path)!=row['sha256']:raise ValueError('Shard mismatch')
                    shards+=1
                if split=='train':train+=value['tokens']
                else:validation+=value['tokens']
            state['legacy_policy_identity']=previous;state['policy_identity']=policy;states.append((name,state))
        db.execute('BEGIN IMMEDIATE')
        db.execute('UPDATE metadata SET value=? WHERE name=\'policy\'',(json.dumps(policy),))
        for name,state in states:db.execute('UPDATE checkpoints SET state=? WHERE name=?',(json.dumps(state,ensure_ascii=False),name))
        db.commit()
        write_json(runs/'reports/pretrain_policy_migration.json',dict(created_utc=utc(),old=previous,new=policy,backup=str(backup),all_shards_verified=shards,train_tokens_preserved=train,validation_tokens_preserved=validation,rules_identical=True))
        print(json.dumps(dict(shards=shards,train_tokens=train,validation_tokens=validation)))
    finally:db.close()

if __name__=='__main__':main()
