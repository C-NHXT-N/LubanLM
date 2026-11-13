"""Failure regressions. Fixtures are isolated and never used for training."""
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import psutil
import torch
from scripts.common import ROOT,read_json,write_json
from scripts.worker_health import acquire_lock,release_lock,health_check,reconcile_cpu,durable_progress
from scripts.sft_disk import connect,insert_candidate,assign_splits
from scripts.policy_versions import pretrain_identity
from training.batches import SFTPairs

class RepairTests(unittest.TestCase):
    def temp(self):return tempfile.TemporaryDirectory(prefix='repair-fixture-',dir=ROOT/'.cache')
    def test_lock_malformed_recycled_and_release_identity(self):
        with self.temp() as t:
            path=Path(t)/'lock'
            for value in ('bad',json.dumps(dict(pid=os.getpid(),created=psutil.Process().create_time()+1))):
                path.write_text(value);owner=acquire_lock(path,('-m',));self.assertIsNotNone(owner)
                self.assertIsNone(acquire_lock(path,('-m',)))
                release_lock(path,dict(owner,token='wrong'));self.assertTrue(path.exists())
                release_lock(path,owner);self.assertFalse(path.exists())
            path.write_text('');os.utime(path,(time.time()-10,time.time()-10))
            owner=acquire_lock(path,('-m',));self.assertIsNotNone(owner);release_lock(path,owner)
    def test_work_stall_two_checks_and_heartbeat(self):
        first,stalled=health_check({},'a',0,now=0);self.assertFalse(stalled)
        suspect,stalled=health_check(first,'a',0,now=1800);self.assertFalse(stalled)
        _,stalled=health_check(suspect,'a',0,now=2100);self.assertTrue(stalled)
        changed,stalled=health_check(suspect,'b',0,now=2100);self.assertFalse(stalled);self.assertIsNone(changed['stall_suspected_at'])
        first,_=health_check({},'c',181,now=20)
        _,stalled=health_check(first,'changed_cpu',181,now=320);self.assertTrue(stalled)
    def test_cpu_orphan_reconciliation_idempotent(self):
        with self.temp() as t:
            path=Path(t)/'budget.json';write_json(path,[dict(seconds=50,finished=True)])
            with patch('scripts.worker_health.time.time',return_value=130):
                self.assertEqual(reconcile_cpu(path,123,4,'preparing',100),80)
                self.assertEqual(reconcile_cpu(path,123,4,'preparing',100),80)
            with patch('scripts.worker_health.time.time',return_value=200):
                self.assertEqual(reconcile_cpu(path,123,4,'preparing',100,finished=True),150)
            with patch('scripts.worker_health.time.time',return_value=500):
                self.assertEqual(reconcile_cpu(path,123,4,'preparing',100),150)
            self.assertEqual(len(read_json(path)),2)
    def test_durable_signature_ignores_uncommitted_wal(self):
        with self.temp() as t:
            root=Path(t);db=sqlite3.connect(root/'dedup.sqlite');db.execute('PRAGMA journal_mode=WAL');db.execute('CREATE TABLE checkpoints(name TEXT,state TEXT)')
            db.execute('INSERT INTO checkpoints VALUES (?,?)',('x',json.dumps(dict(train=dict(tokens=10),done_files=['a']))));db.commit()
            before=durable_progress(root)
            db.execute('UPDATE checkpoints SET state=?',(json.dumps(dict(train=dict(tokens=99),done_files=['a','b'])),))
            self.assertEqual(durable_progress(root),before);db.commit();self.assertNotEqual(durable_progress(root),before);db.close()
    def test_covering_query_results_unchanged(self):
        db=sqlite3.connect(':memory:');db.execute('CREATE TABLE near(bucket INTEGER,fp TEXT,hash TEXT)')
        db.executemany('INSERT INTO near VALUES (?,?,?)',[(i%7,str(i%20),str(i)) for i in range(500)])
        query='SELECT DISTINCT fp FROM near WHERE bucket IN (?,?,?,?)';args=(1,2,3,4)
        before=set(db.execute(query,args));db.execute('CREATE INDEX near_covering ON near(bucket,fp)')
        self.assertEqual(before,set(db.execute(query,args)));self.assertIn('COVERING',str(list(db.execute('EXPLAIN QUERY PLAN '+query,args))));db.close()
    def test_pretrain_policy_unchanged(self):
        old=ROOT/'docs/run-results/teaching-repair-20261006/minimal-before'
        self.assertEqual(pretrain_identity(ROOT/'configs/50m-general-pretrain-10b-teaching.json'),pretrain_identity(old/'plan.json',old))
    def test_prefix_family_cross_source_and_generic_greeting(self):
        with self.temp() as t:
            db=connect(Path(t)/'candidates.sqlite')
            def item(identity,messages):return dict(group=identity,question_sha256=identity,system='助手',task='daily',source=identity,messages=messages,input_ids=[1,3,2],labels=[-100,3,2])
            short=[dict(role='user',content='周末有点累，想轻松休息一下。'),dict(role='assistant',content='可以留一点时间休息，选择轻松的活动。')]
            full=short+[dict(role='user',content='下午应该怎样安排？'),dict(role='assistant',content='先休息，再散步，晚上早点睡。')]
            for i,m in [('short',short),('full',full)]:self.assertIsNone(insert_candidate(db,item(i,m),1))
            common=[dict(role='user',content='你好'),dict(role='assistant',content='您好')]
            self.assertIsNone(insert_candidate(db,item('generic-short',common),1))
            for i,q in [('g1','厨房怎么整理？'),('g2','工作压力怎么办？')]:self.assertIsNone(insert_candidate(db,item(i,common+[dict(role='user',content=q),dict(role='assistant',content='先处理简单的事项。')]),1))
            for i in range(30):
                id='other'+str(i);db.execute('INSERT INTO candidates VALUES (?,?,?,?,?,?,?,?,NULL,0)',(id,id,id,id,'daily','other',id,'{}'))
            db.commit();assign_splits(db,dict(sft=dict(test_examples=5,validation_examples=5)))
            families=dict(db.execute('SELECT id,family FROM candidates'))
            self.assertEqual(families['short'],families['full']);self.assertNotEqual(families['g1'],families['g2'])
            self.assertEqual(db.execute("SELECT split FROM candidates WHERE id='short'").fetchone()[0],'excluded')
            self.assertEqual(db.execute("SELECT COUNT(*) FROM representatives WHERE id='full'").fetchone()[0],1);db.close()
    def test_mapped_reader_batches_cache_and_resume(self):
        with self.temp() as t:
            path=Path(t)/'tokens.jsonl';rows=[dict(input_ids=[1]+[i+3]*(i%7+1)+[2],labels=[-100]+[i+3]*(i%7+1)+[2],task='daily',encoding_version='v1',messages=['extra metadata'*100]) for i in range(50)]
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
            datasets=[]
            try:
                legacy=SFTPairs(path,0,torch.Generator().manual_seed(1));mapped=SFTPairs(path,0,torch.Generator().manual_seed(1),batching='flat');datasets.append(mapped)
                for _ in range(20):
                    for a,b in zip(legacy.batch(8,None),mapped.batch(8,None)):self.assertTrue(torch.equal(a,b))
                state=mapped.state();rng=mapped.generator.get_state();expected=mapped.batch(8,None)
                other=SFTPairs(path,0,torch.Generator(),batching='flat');datasets.append(other);other.restore(state);other.generator.set_state(rng)
                for a,b in zip(expected,other.batch(8,None)):self.assertTrue(torch.equal(a,b))
                self.assertNotIn('messages',mapped.rows[0])
            finally:
                for d in datasets:d.rows.close()
            cached=next((path.parent/'.sft-cache').glob('*/ids.bin'));cached.write_bytes(b'broken')
            rebuilt=SFTPairs(path,0,torch.Generator(),batching='flat');self.assertEqual(rebuilt.rows[0]['input_ids'].tolist(),rows[0]['input_ids']);rebuilt.rows.close()
    def test_shortfall_is_typed_and_outcome_fails_closed(self):
        from scripts.sft_disk import DataShortfall,main
        with self.temp() as t:
            plan=Path(t)/'plan.json';write_json(plan,dict(runs_root=t))
            with patch('sys.argv',['fixture','--plan',str(plan)]),patch('scripts.sft_disk.sft_identity',return_value={'v':4}),patch('scripts.worker_health.WorkerHealth'),patch('scripts.sft_disk.run',side_effect=DataShortfall('DATA_SHORTFALL')):
                with self.assertRaises(DataShortfall):main()
            self.assertEqual(read_json(Path(t)/'reports/sft_outcome.json')['status'],'data_shortfall')
            with patch('sys.argv',['fixture','--plan',str(plan)]),patch('scripts.sft_disk.sft_identity',return_value={'v':4}),patch('scripts.worker_health.WorkerHealth'),patch('scripts.sft_disk.run',side_effect=ValueError('checksum')):
                with self.assertRaises(ValueError):main()
            self.assertEqual(read_json(Path(t)/'reports/sft_outcome.json')['status'],'failed')
    def test_controller_trains_base_despite_sft_shortfall(self):
        from scripts.run_general_10b import main
        with self.temp() as t:
            root=Path(t);runs=root/'runs';reports=runs/'reports';reports.mkdir(parents=True)
            plan=read_json(ROOT/'configs/50m-general-pretrain-10b-teaching.json')
            parent=root/'parent.pt';parent.write_bytes(b'fixture only')
            plan.update(data_root=str(root/'data'),runs_root=str(runs),parent_checkpoint=str(parent))
            path=root/'plan.json';write_json(path,plan)
            write_json(reports/'preparation_complete.json',dict(policy_identity={'fixture':1},training_target_tokens=plan['additional_tokens'],parent_overlap='fixture'))
            write_json(reports/'pretrain_content_review.json',{'fixture':1})
            from scripts.common import sha256
            train_manifest=root/'data/processed/pretrain-train/manifest.json'
            write_json(train_manifest,dict(fixture=True))
            write_json(reports/'teaching-throughput-v3.json',dict(data_manifest_sha256=sha256(train_manifest),parent_checkpoint_sha256=sha256(parent),rows=[dict(micro_batch=24,status='completed',tokens_per_hour=100000000)]))
            calls=[]
            class Fake:
                pid=os.getpid()
                def __init__(self,cmd,**kwargs):
                    module=cmd[cmd.index('-m')+1];calls.append(module);self.code=0
                    if module=='scripts.prepare_sft_100k':
                        self.code=1;write_json(reports/'sft_outcome.json',dict(status='data_shortfall',pid=self.pid,policy_identity={'v':4}))
                        write_json(reports/'sft_shortfall.json',dict(gaps={'train':1}))
                    if module=='training.train':
                        output=Path(cmd[cmd.index('--output')+1]);output.mkdir(parents=True)
                        write_json(output/'summary.json',dict(supervised_tokens=plan['additional_tokens'],cumulative_pretrain_tokens=plan['target_cumulative_tokens']))
                def poll(self):return self.code
                def wait(self,**kwargs):return self.code
                def terminate(self):pass
            state=dict(settings=dict(stage='pretrain'),cumulative_pretrain_tokens=plan['parent_tokens'],tokenizer_sha256=plan['tokenizer_sha256'])
            with patch('sys.argv',['fixture','--plan',str(path)]),patch('scripts.run_general_10b.subprocess.Popen',Fake),patch('scripts.run_general_10b.verify_manifest',return_value={'tokens':plan['additional_tokens']+1}),patch('scripts.run_general_10b.pretrain_identity',return_value={'fixture':1}),patch('scripts.run_general_10b.sft_identity',return_value={'v':4}),patch('scripts.run_general_10b.verify_content_review',return_value={'fixture':True}),patch('torch.load',return_value=state),patch('builtins.print'):
                main()
            self.assertIn('training.train',calls)
            self.assertEqual(read_json(reports/'controller.json')['status'],'base_complete_sft_data_shortfall')
    def test_live_orphan_worker_and_fresh_resource_report(self):
        from scripts.teaching_supervisor import tick
        from scripts.common import utc
        with self.temp() as t:
            root=Path(t);plan=read_json(ROOT/'configs/50m-general-pretrain-10b-teaching.json')
            plan.update(data_root=str(root/'data'),runs_root=str(root/'runs'),disk_free_bytes_floor=1,supervisor_start_memory_bytes=1)
            (root/'data').mkdir();path=root/'plan.json';write_json(path,plan)
            reports=root/'runs/reports';reports.mkdir(parents=True)
            script="from scripts.worker_health import WorkerHealth;import time\nwith WorkerHealth("+repr(str(path))+",'preparing_10b_cpu'):time.sleep(2)"
            process=subprocess.Popen([sys.executable,'-B','-c',script,'scripts.prepare_general_10b'],creationflags=subprocess.CREATE_NO_WINDOW)
            try:
                for _ in range(100):
                    if (reports/'worker_heartbeat.json').exists():break
                    time.sleep(.02)
                created=psutil.Process(process.pid).create_time()
                write_json(reports/'controller.json',dict(pid=99999999,process_created=0,status='preparing_10b_cpu',child_pid=process.pid,child_created=created,started_utc=utc(),updated_utc=utc()))
                write_json(reports/'supervisor.json',dict(available_memory_bytes=1,disk_free_bytes=1))
                with patch('scripts.teaching_supervisor.subprocess.Popen',side_effect=AssertionError('must not launch duplicate')):tick(path)
                report=read_json(reports/'supervisor.json')
                self.assertEqual(report['status'],'orphan_worker_adopted')
                self.assertGreater(report['available_memory_bytes'],1);self.assertGreater(report['disk_free_bytes'],1)
                process.wait(timeout=10)
                self.assertTrue(read_json(reports/'cpu_preparation_budget.json')[0]['finished'])
                self.assertGreaterEqual(read_json(reports/'cpu_preparation_budget.json')[0]['seconds'],1.9)
            finally:
                if process.poll() is None:process.terminate();process.wait()

if __name__=='__main__':unittest.main()
