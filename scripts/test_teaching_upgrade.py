"""Regression fixtures only; these records never enter a training dataset."""
from collections import Counter
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import psutil
from scripts.common import ROOT,read_json,write_json
from scripts.corpus_quality import arithmetic
from scripts.prepare_sft_100k import category,candidates
from scripts.policy_versions import pretrain_identity
from scripts.sft_disk import connect,insert_candidate,checkpoint_records,finalize
from scripts.balanced_sft import soft_allocations
from scripts.teaching_supervisor import alive,decision

class UpgradeTests(unittest.TestCase):
    def test_compound_arithmetic_and_false_final_number(self):
        self.assertFalse(arithmetic('计算 1+2*3=?','答案是3。'))
        self.assertTrue(arithmetic('计算 (1+2)*3=?','结果是9。'))
        self.assertFalse(arithmetic('计算 2+3=?','2+3=6，再给出5个例子。'))
        self.assertIsNone(arithmetic('计算 2+3=?','这里还有5个例子。'))
        self.assertFalse(arithmetic('计算 2+3=?','结果是6，答案是5。'))
        self.assertFalse(arithmetic('计算 1/0=?','答案是0。'))
    def test_natural_identity_and_greeting(self):
        self.assertEqual(category('everyday.parquet','你能介绍一下自己吗？'),'identity')
        self.assertEqual(category('everyday.parquet','你好，今天过得怎么样？'),'greeting')
        self.assertEqual(category('everyday.parquet','什么是植物的光合作用？'),'concept')
        self.assertEqual(category('information-seeking.parquet','最近工作让我压力很大，怎么办？'),'daily')
    def test_pretrain_reuse_proof(self):
        old=ROOT/'docs/run-results/teaching-repair-20261006/pre-execution'
        self.assertEqual(pretrain_identity(ROOT/'configs/50m-general-pretrain-10b-teaching.json'),pretrain_identity(old/'plan.json',old))
    def test_soft_flow_with_rerouting(self):
        counts={('daily','A'):60,('daily','B'):60,('concept','A'):40,('instruction','B'):40}
        allocation,gaps=soft_allocations(counts,{'A':.25,'B':1},100,50,30)
        self.assertFalse(any(gaps.values()));self.assertEqual(sum(allocation.values()),100)
        self.assertGreaterEqual(sum(n for (t,s),n in allocation.items() if t=='daily'),50)
        self.assertLessEqual(sum(n for (t,s),n in allocation.items() if t=='concept'),30)
        self.assertLessEqual(sum(n for (t,s),n in allocation.items() if s=='A'),25)
    def test_full_context_generic_greeting_and_transaction_recovery(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.cache') as temp:
            path=Path(temp)/'pool.sqlite';db=connect(path)
            def item(identity,question):return dict(group=identity,question_sha256=identity,task='daily',source='source',system='助手',messages=[dict(role='user',content='你好'),dict(role='assistant',content='您好'),dict(role='user',content=question),dict(role='assistant',content='请先列出具体事项，再逐项处理。')],input_ids=[1,3,2],labels=[-100,3,2])
            self.assertIsNone(insert_candidate(db,item('1','我想整理一下厨房。'),9));db.commit()
            self.assertIsNone(insert_candidate(db,item('2','最近工作让我有些疲惫。'),9));db.close() # uncommitted row is rolled back
            db=connect(path);self.assertEqual(db.execute('SELECT COUNT(*) FROM candidates').fetchone()[0],1)
            self.assertIsNone(insert_candidate(db,item('2','最近工作让我有些疲惫。'),9));db.commit()
            self.assertEqual(db.execute('SELECT COUNT(*) FROM candidates').fetchone()[0],2);db.close()
    def test_soft_targets_retain_available_rare_tasks(self):
        counts={(t,'A'):n for t,n in dict(daily=100,concept=100,instruction=100,greeting=5,identity=5,arithmetic=5).items()}
        allocation,gaps=soft_allocations(counts,{},100,50,30)
        self.assertFalse(any(gaps.values()))
        for task in ('greeting','identity','arithmetic'):self.assertEqual(allocation[task,'A'],5)
    def test_stream_legacy_checkpoint(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.cache') as temp:
            path=Path(temp)/'checkpoint.json';values=[dict(id=i,text='中文'*5000) for i in range(110)]
            write_json(path,dict(counts={'rows':110},completed_files=['a'],selected=values))
            self.assertEqual(list(checkpoint_records(path)),values)
    def test_supervisor_process_exit_and_retry_boundaries(self):
        process=subprocess.Popen([sys.executable,'-B','-c','import time;time.sleep(5)'],creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            created=psutil.Process(process.pid).create_time();self.assertTrue(alive(process.pid,created,('-c',)))
            self.assertFalse(alive(process.pid,created+1,('-c',)))
        finally:process.terminate();process.wait()
        self.assertFalse(alive(process.pid,created,('-c',)))
        plan=dict(disk_free_bytes_floor=20,supervisor_start_memory_bytes=4)
        self.assertEqual(decision(plan,{'status':'preparing'}, {},8,100,'x'),'start')
        self.assertEqual(decision(plan,{}, {'signature':'x','failed_without_progress':3},8,100,'x'),'repair_wait')
        self.assertEqual(decision(plan,{}, {},2,100,'x'),'resource_wait')
        self.assertEqual(decision(plan,{'status':'awaiting_pretrain_content_inspection'}, {},8,100,'x'),'review_wait')
        self.assertEqual(decision(plan,{}, {},8,100,'x','MemoryError'),'repair_wait')
    def test_actual_record_filter_runs(self):
        from tokenizers import Tokenizer
        tokenizer=Tokenizer.from_file('D:/LubanLM-Data/chat-v2/tokenizers/vocab-32000/tokenizer.json')
        plan=read_json(ROOT/'configs/50m-general-pretrain-10b-teaching.json')
        row=dict(instruction='你好，今天过得怎么样？',output='你好！我可以陪你聊聊，你今天过得怎么样？')
        result=list(candidates(row,{'repo':'test/public','revision':'pinned'},'everyday.parquet','test/file',1,{'lfs':{'sha256':'sha'}},plan,tokenizer,set(),set(),Counter()))
        self.assertEqual(len(result),1);self.assertEqual(result[0]['task'],'greeting');self.assertEqual(result[0]['source_row'],1)
        row=dict(instruction='计算 1+2*3=?',output='结果是7。')
        result=list(candidates(row,{'repo':'test/public','revision':'pinned'},'everyday.parquet','test/file',2,{'lfs':{'sha256':'sha'}},plan,tokenizer,set(),set(),Counter()))
        self.assertEqual(len(result),1);self.assertEqual(result[0]['task'],'arithmetic')
    def test_supervisor_restarts_crashed_controller_end_to_end(self):
        from scripts.teaching_supervisor import tick
        original_popen=subprocess.Popen
        original_iter=psutil.process_iter
        with tempfile.TemporaryDirectory(dir=ROOT/'.cache') as temp:
            folder=Path(temp);data=folder/'data';data.mkdir();runs=folder/'runs'
            plan=read_json(ROOT/'configs/50m-general-pretrain-10b-teaching.json')
            plan.update(data_root=str(data),runs_root=str(runs),disk_free_bytes_floor=1,supervisor_start_memory_bytes=1)
            path=folder/'plan.json';write_json(path,plan)
            script="import os,psutil,time;from pathlib import Path;from scripts.common import write_json,utc;p=Path("+repr(str(runs))+ ");p.mkdir(parents=True,exist_ok=True);(p/'controller.lock').write_text(str(os.getpid()));write_json(p/'reports/controller.json',dict(pid=os.getpid(),process_created=psutil.Process().create_time(),status='preparing_10b_cpu',started_utc=utc(),updated_utc=utc()));time.sleep(.4);os._exit(99)"
            workers=[]
            def launch(command,**kwargs):
                worker=original_popen([sys.executable,'-B','-c',script,'scripts.run_general_10b','--plan',str(path)],**kwargs);workers.append(worker);return worker
            def fixture_processes(attributes):
                for process in original_iter(attributes):
                    if str(path) in (process.info.get('cmdline') or []):yield process
            with patch('scripts.teaching_supervisor.subprocess.Popen',side_effect=launch),patch('scripts.teaching_supervisor.psutil.process_iter',side_effect=fixture_processes):
                tick(path);workers[-1].wait(timeout=10)
                self.assertTrue((runs/'controller.lock').exists())
                tick(path);workers[-1].wait(timeout=10)
                status=read_json(runs/'reports/supervisor.json')
                self.assertEqual(status['attempt'],2);self.assertEqual(len(workers),2)
                self.assertTrue((runs/'supervisor-logs/0001.stderr.log').exists())
                self.assertTrue((runs/'supervisor-logs/0002.stderr.log').exists())
    def test_stratified_selection_stream_export_and_600_sample_uniqueness(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.cache') as temp:
            root=Path(temp);out=root/'outputs';out.mkdir();db=connect(root/'pool.sqlite')
            for i in range(1400):
                identity=str(i);task='daily' if i%10<6 else ('concept' if i%10<8 else 'instruction')
                source='A' if i%3 else 'B'
                item=dict(group=identity,task=task,source=source,input_ids=[1,3,2],labels=[-100,3,2],messages=[dict(role='user',content='存储测试'),dict(role='assistant',content='测试')])
                db.execute('INSERT INTO candidates VALUES (?,?,?,?,?,?,?,?,NULL,0)',(identity,identity,identity,identity,task,source,identity.zfill(6),json.dumps(item)))
            db.commit()
            plan=dict(seed=1,tokenizer_sha256='fixture',sft=dict(train_minimum=1000,validation_examples=100,test_examples=100,daily_minimum=500,concept_maximum=300,source_fraction_caps={'A':.8}))
            finalize(db,plan,{'fixture':True},out,root,{})
            report=read_json(out/'manifest.json');self.assertEqual(report['examples'],dict(train=1000,validation=100,test=100))
            self.assertGreaterEqual(report['categories']['train']['daily'],500)
            samples=read_json(root/'reports/sft_semantic_samples.json');self.assertEqual(len(samples),600);self.assertEqual(len({r['group'] for r in samples}),600)
            for split in ('validation','test'):
                self.assertTrue(all(n>=5 for n in report['categories'][split].values()))
            sets=[]
            for split in ('train','validation','test'):
                with (out/f'{split}-tokens.jsonl').open(encoding='utf-8') as stream:sets.append({json.loads(line)['group'] for line in stream})
            self.assertFalse(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2]);db.close()

if __name__=='__main__':unittest.main()
