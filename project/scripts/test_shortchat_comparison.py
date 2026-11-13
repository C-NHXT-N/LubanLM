"""Meaningful CPU checks for isolation, resume and fail-closed comparison."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch,Mock
from contextlib import ExitStack,nullcontext,redirect_stdout
import io
import sqlite3
import torch
from training.sft_audit import SFTOrderAudit
from training.batches import SFTPairs,PrefetchedSFT
from scripts.shortchat_common import classify,repetition,partition,weighted_score
from scripts.prepare_shortchat import allocate

class ComparisonTests(unittest.TestCase):
    def test_resume_coverage_and_order(self):
        uninterrupted=SFTOrderAudit(32)
        for start in range(0,32,8):uninterrupted.add(list(range(start,start+8)),80)
        resumed=SFTOrderAudit(32);resumed.add(list(range(8)),80)
        resumed=SFTOrderAudit(32,resumed.state())
        for start in range(8,32,8):resumed.add(list(range(start,start+8)),80)
        self.assertEqual(resumed.state(),uninterrupted.state())
        broken=SFTOrderAudit(16)
        with self.assertRaises(ValueError):broken.add([0]*16,16)

    def test_prefetch_audits_consumed_not_produced(self):
        cache=Path('.cache/shortchat-tests');cache.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as tmp:
            p=Path(tmp)/'data.jsonl'
            rows=[dict(input_ids=[1,i+3,2],labels=[-100,i+3,2],encoding_version='prefix-stable-v1',task='daily') for i in range(32)]
            p.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
            a=SFTPairs(p,0,torch.Generator().manual_seed(2026),batching='legacy')
            pre=PrefetchedSFT(a,8,2);audit=SFTOrderAudit(32)
            try:
                for _ in range(2):pre.batch(8,a.generator);audit.add(a.last_indices,16)
                state=a.state();rng=a.generator.get_state();saved=audit.state()
            finally:pre.close()
            b=SFTPairs(p,0,torch.Generator().manual_seed(2026),batching='legacy');b.restore(state);b.generator.set_state(rng)
            resumed=SFTOrderAudit(32,saved)
            for _ in range(2):b.batch(8,b.generator);resumed.add(b.last_indices,16)
            self.assertTrue(resumed.completed[0]['every_example_once'])
            self.assertEqual(resumed.targets,64)

    def test_source_caps_and_shortfall(self):
        from scripts.shortchat_common import CATEGORIES
        counts={(source,c):1000 for source in ('A','B','C','BelleGroup/train_1M_CN','rule-basics-v1') for c in CATEGORIES}
        flow,allocation=allocate(counts,1000);self.assertEqual(flow,1000)
        self.assertLessEqual(sum(v for (s,c),v in allocation.items() if s=='rule-basics-v1'),100)
        self.assertLessEqual(sum(v for (s,c),v in allocation.items() if s=='BelleGroup/train_1M_CN'),150)
        self.assertLessEqual(allocate({('A',c):1000 for c in CATEGORIES},1000)[0],400)

    def test_misclassification_and_list_repetition(self):
        messages=[dict(role='user',content='描述艺术史学科的特点和目标。'),dict(role='assistant',content='艺术史研究艺术的发展。')]
        self.assertEqual(classify(messages,'everyday.parquet'),'concept')
        self.assertTrue(repetition('1. 把书放进书包。\n2. 把书放进书包。\n3. 把书放进书包。'))
        self.assertEqual(partition('a-family'),partition('a-family'))
        for question in ('请把这段话改成口语化表达：我最近工作很累。','这句话需要语法改正：我好久没打球了。','请帮我改进这段内容：今天我想休息。'):
            self.assertEqual(classify([dict(role='user',content=question),dict(role='assistant',content='今天我想休息。')]),'instruction')
        self.assertEqual(classify([dict(role='user',content='学生在学习中应遵守学习纪律。'),dict(role='assistant',content='学生在学习过程中应遵守纪律。')],'editing.parquet'),'instruction')

    def test_missing_grades_cannot_pass(self):
        from scripts.shortchat_common import CATEGORIES
        rows=[dict(category=c,grading=None) for c in CATEGORIES]
        with self.assertRaises(ValueError):weighted_score(rows)

    def test_source_tools_persona_and_human_dialogue_are_not_admitted(self):
        from scripts.shortchat_source_quality import row_reason,message_reason
        self.assertEqual(row_reason({'conversations':[{'role':'assistant','content':'好了','tool_calls':[{'name':'lookup'}]}]},{'repo':'jingyaogong/minimind_dataset'}),'tool_protocol')
        self.assertEqual(row_reason({'language':'Chinese','toxic':True},{'repo':'allenai/WildChat-1M'}),'source_language_or_risk')
        for answer in ('看过啊，那是我小时候最喜欢的动漫之一！','我最近挺好的，谢谢关心。','我最近尝试了不少新菜谱。','我是jingyaogong开发的小模型。'):
            self.assertIsNotNone(message_reason([dict(role='user',content='最近过得怎么样？'),dict(role='assistant',content=answer)]))
        good=[dict(role='user',content='工作忙，感觉有点累。'),dict(role='assistant',content='可以先休息一会儿，再处理事情。')]
        self.assertIsNone(message_reason(good,human=True))
        self.assertEqual(message_reason([good[0],dict(role='assistant',content='我也很累，刚下班。')],human=True),'human_persona')

    def test_physical_probe_does_not_flash_console(self):
        from scripts.direct_mirror import PhysicalMirror
        with patch('scripts.direct_mirror.subprocess.check_output',return_value='[{"index":4,"name":"Ethernet","ip":"192.168.0.2"}]') as call:
            PhysicalMirror()
        self.assertIn('creationflags',call.call_args.kwargs)

    def test_explicit_physical_address_matches_and_never_falls_back(self):
        from scripts.direct_mirror import PhysicalMirror
        rows=json.dumps([dict(index=9,name='Ethernet',ip='192.168.0.2'),dict(index=10,name='WLAN',ip='192.168.43.4')])
        with patch.dict('os.environ',{},clear=True),patch('scripts.direct_mirror.subprocess.check_output',return_value=rows):
            mirror=PhysicalMirror(interface='WLAN',bind_ip='192.168.43.4')
            self.assertEqual(mirror.adapter['index'],10)
            with self.assertRaises(RuntimeError):PhysicalMirror(interface='WLAN',bind_ip='10.183.83.36')
            with self.assertRaises(RuntimeError):PhysicalMirror(interface='Clash',bind_ip='198.18.0.1')
            with self.assertRaises(ValueError):PhysicalMirror(bind_ip='invalid-address')
        with patch.dict('os.environ',dict(LUBAN_DOWNLOAD_INTERFACE='WLAN',LUBAN_DOWNLOAD_BIND_IP='192.168.43.4')),patch('scripts.direct_mirror.subprocess.check_output',return_value=rows):
            self.assertEqual(PhysicalMirror().adapter['ip'],'192.168.43.4')
        with patch.dict('os.environ',{},clear=True),patch('scripts.direct_mirror.subprocess.check_output',return_value='null'):
            with self.assertRaises(RuntimeError):PhysicalMirror(bind_ip='192.168.43.4')

    def test_pinned_cached_metadata_rfilename_and_optional_lfs_size(self):
        from scripts.prepare_shortchat import Builder
        from scripts.common import write_json,sha256
        cache=Path('.cache/shortchat-tests');cache.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as tmp:
            root=Path(tmp);raw=root/'cached';raw.mkdir();file=raw/'zh.jsonl';file.write_text('{}\n',encoding='utf-8')
            write_json(root/'metadata.json',dict(sha='pinned',siblings=[dict(rfilename='zh.jsonl',size=file.stat().st_size,lfs=dict(sha256=sha256(file)))]))
            builder=Builder.__new__(Builder);builder.plan={};builder.data=root/'data';builder.data.mkdir();builder.runs=root/'runs'
            builder.source_plan=dict(sft_raw_root=str(raw),sft=dict(raw_sources=[dict(repo='shareAI/ShareGPT-Chinese-English-90k',revision='pinned',metadata='metadata.json',files=['zh.jsonl'])]))
            builder.db=sqlite3.connect(':memory:');builder.db.execute('CREATE TABLE cursors(name TEXT,position INTEGER,complete INTEGER)')
            builder.ingest=Mock(side_effect=lambda name,it:list(it));builder.inventories=Mock(return_value=[]);builder.feasible=Mock(return_value={})
            with patch('scripts.prepare_shortchat.ROOT',root),patch('scripts.prepare_sft_100k.rows',return_value=[]):builder.supplements()
            builder.ingest.assert_called_once();self.assertTrue(file.exists());builder.db.close()

    def test_real_epoch_checkpoint_pause_and_resume_on_cpu(self):
        # Real tiny model/optimizer and checkpoint I/O; only GPU plumbing is replaced.
        from training import train
        from scripts.common import write_json,read_json
        cache=Path('.cache/shortchat-tests');cache.mkdir(parents=True,exist_ok=True)
        class Budget:
            def __init__(self,*args):pass
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def remaining(self):return 10000
            def tick(self):return False
        with tempfile.TemporaryDirectory(dir=cache) as tmp,ExitStack() as stack:
            root=Path(tmp);tokenizer=root/'tokenizers/vocab-32000';tokenizer.mkdir(parents=True);(tokenizer/'tokenizer.json').write_text('{}',encoding='utf-8')
            cfg=json.loads(Path('configs/config_50m.json').read_text(encoding='utf-8'));cfg.update(vocab_size=64,hidden_size=16,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=8,intermediate_size=32,context_length=32)
            write_json(root/'config.json',cfg);data=root/'train.jsonl'
            data.write_text(''.join(json.dumps(dict(input_ids=[1,10+i,2],labels=[-100,10+i,2],task='daily',encoding_version='prefix-stable-v1'))+'\n' for i in range(8)),encoding='utf-8')
            for target,value in [('training.train.DATA',root),('training.train.require_gpu',lambda:'cpu'),('training.train.GPUBudget',Budget),('torch.autocast',lambda *a,**k:nullcontext()),('torch.cuda.is_bf16_supported',lambda:True),('torch.cuda.get_device_name',lambda *a:'test CPU'),('torch.cuda.reset_peak_memory_stats',lambda:None),('torch.cuda.synchronize',lambda:None),('torch.cuda.get_rng_state_all',lambda:[]),('torch.cuda.set_rng_state_all',lambda x:None),('torch.cuda.max_memory_allocated',lambda:0),('torch.cuda.max_memory_reserved',lambda:0)]:stack.enter_context(patch(target,value))
            args=['train','--config',str(root/'config.json'),'--stage','sft','--output',str(root/'output'),'--minutes','10','--epochs','2','--micro-batch','4','--accumulation','1','--sft-train',str(data),'--sft-validation',str(data),'--sft-batching','bucketed','--prefetch-batches','2','--validation-micro-batch','2','--full-sft-validation','--validation-every-steps','2','--save-epoch-checkpoints','--pause-after-checkpoint','--execution','optimized']
            with patch('sys.argv',args),redirect_stdout(io.StringIO()):train.main()
            first=train.load_checkpoint(root/'output/epoch-01.pt');self.assertEqual(first['step'],2);self.assertEqual(len(first['sft_order_audit']['completed']),1)
            with patch('sys.argv',args+['--resume',str(root/'output/epoch-01.pt')]),redirect_stdout(io.StringIO()):train.main()
            last=train.load_checkpoint(root/'output/epoch-02.pt');self.assertEqual(last['step'],4)
            self.assertEqual(len(last['sft_order_audit']['completed']),2);self.assertEqual(last['sft_order_audit']['consumed'],16)
            self.assertTrue(all(r['every_example_once'] for r in read_json(root/'output/epoch_audit.json')['epochs']))
            uninterrupted=args[:];uninterrupted[uninterrupted.index('--output')+1]=str(root/'full-output');uninterrupted.remove('--pause-after-checkpoint')
            with patch('sys.argv',uninterrupted),redirect_stdout(io.StringIO()):train.main()
            full=train.load_checkpoint(root/'full-output/epoch-02.pt')
            self.assertEqual(last['sft_order_audit'],full['sft_order_audit'])
            self.assertTrue(all(torch.equal(last['model'][k],v) for k,v in full['model'].items()))
            self.assertEqual(read_json(root/'output/summary.json')['initial_validation_loss'],read_json(root/'full-output/summary.json')['initial_validation_loss'])

    def test_evaluation_budget_pause_cannot_pass_and_partial_resume(self):
        from scripts.compare_shortchat import evaluate,BudgetPause,code_identity,graded
        from scripts.common import write_json,read_json,sha256
        cache=Path('.cache/shortchat-tests');cache.mkdir(parents=True,exist_ok=True)
        class Budget:
            left=0
            def __init__(self,*args):pass
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def remaining(self):return self.left
            def tick(self):return False
        with tempfile.TemporaryDirectory(dir=cache) as tmp:
            root=Path(tmp);folder=root/'data/processed';folder.mkdir(parents=True);runs=root/'runs';checkpoint=root/'model.pt';checkpoint.write_bytes(b'test')
            plan=dict(data_root=str(root/'data'),runs_root=str(runs));alias='a'*12
            suite=dict(questions=[dict(id='one',category='daily_single',messages=[dict(role='user',content='你好')],reference='你好')]);write_json(folder/'dev-suite.json',suite)
            with patch('scripts.common.GPUBudget',Budget),patch('scripts.common.require_gpu',return_value='cpu'),patch('inference.generate.load_model',return_value=(None,None)),patch('training.execution.configure_model'),patch('inference.generate.complete',return_value=('你好',[10,2])):
                with self.assertRaises(BudgetPause):evaluate(plan,checkpoint,'dev',alias)
                out=runs/'evaluations'/alias;self.assertEqual(read_json(out/'dev-meta.json')['status'],'budget_paused')
                with self.assertRaises(ValueError):graded(plan,alias,'dev')
                # Interrupted append is discarded before the same frozen suite resumes.
                (out/'dev.jsonl').write_bytes(b'{"unfinished":')
                Budget.left=10000;evaluate(plan,checkpoint,'dev',alias)
                self.assertEqual(read_json(out/'dev-meta.json')['status'],'complete')
                self.assertEqual(len((out/'dev.jsonl').read_text(encoding='utf-8').splitlines()),1)
                self.assertIsNone(graded(plan,alias,'dev'))

    def test_quality_allocation_and_stratified_cap_rounding(self):
        from scripts.shortchat_allocation import allocate_quality,stratified_subset,source_cap
        from scripts.shortchat_common import CATEGORIES
        from collections import Counter
        counts={(s,c,cost):100000 for s in ('A','B','C','BelleGroup/train_1M_CN') for c in CATEGORIES for cost in (-2,8)}
        flow,allocation,cost=allocate_quality(counts,100000);self.assertEqual(flow,100000);self.assertLessEqual(cost,0)
        self.assertFalse(any(n for (s,c,v),n in allocation.items() if v>0))
        bins={}
        for (s,c,v),n in allocation.items():
            if n:bins[s,c,1,v]=n-3;bins[s,c,0,v]=3
        selected=stratified_subset(bins);self.assertEqual(sum(selected.values()),20000)
        by_source=Counter();by_category=Counter()
        for (s,c,short,v),n in selected.items():
            self.assertLessEqual(abs(n-bins[s,c,short,v]/5),1);by_source[s]+=n;by_category[c]+=n
        self.assertEqual(dict(by_category),{c:n//5 for c,n in CATEGORIES.items()})
        self.assertTrue(all(n<=source_cap(s,20000) for s,n in by_source.items()))
        long={(s,c,8):100000 for s in ('A','B','C') for c in CATEGORIES}
        self.assertGreater(allocate_quality(long,100000)[2],0)

    def test_windows_report_reader_lock_retry_is_bounded(self):
        from scripts.shortchat_io import write_json
        with patch('scripts.shortchat_io.atomic_write_json',side_effect=[PermissionError('reader lock'),PermissionError('reader lock'),None]) as call,patch('scripts.shortchat_io.time.sleep'):
            write_json('unused.json',dict(stage='running'));self.assertEqual(call.call_count,3)
        with patch('scripts.shortchat_io.atomic_write_json',side_effect=PermissionError('permanent denial')) as call,patch('scripts.shortchat_io.time.sleep'):
            with self.assertRaises(PermissionError):write_json('unused.json',{})
            self.assertEqual(call.call_count,8)

    def test_real_builder_keeps_one_complete_multiturn_and_instruction_intent(self):
        from scripts.prepare_shortchat import Builder
        from scripts.common import read_json,write_json
        cache=Path('.cache/shortchat-tests');cache.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as tmp:
            root=Path(tmp);plan=read_json('configs/50m-shortchat-base-compare.json');plan.update(data_root=str(root/'data'),runs_root=str(root/'runs'))
            write_json(root/'plan.json',plan);builder=Builder(root/'plan.json')
            try:
                meta=dict(source='opencsg/smoltalk-chinese',revision='test-fixture',raw_sha256='fixture',file='everyday.parquet',source_row=42)
                conversation=[dict(role='user',content='我最近想整理露台上的书架，先从哪里开始？'),dict(role='assistant',content='先把书分类，再擦拭书架。'),dict(role='user',content='整理好后怎样保持整洁呢？'),dict(role='assistant',content='看完后把书放回原位，每周检查一次。')]
                builder.add(meta,conversation[:2]);builder.add(meta,conversation)
                self.assertEqual(builder.db.execute('SELECT count(*) FROM candidates').fetchone()[0],1)
                payload=json.loads(builder.db.execute('SELECT payload FROM candidates').fetchone()[0]);self.assertEqual(payload['category'],'daily_multi');self.assertEqual(len(payload['messages']),4)
                self.assertEqual(sum(x==2 for x in payload['labels']),2)
                other=dict(meta,source_row=43,file='editing.parquet');builder.add(other,[dict(role='user',content='学生在学习中应遵守学习纪律。'),dict(role='assistant',content='学生应遵守学习纪律。')])
                self.assertEqual(builder.db.execute("SELECT count(*) FROM candidates WHERE category='instruction'").fetchone()[0],1)
            finally:builder.db.close()

    def test_program_waits_only_at_explicit_review_gates(self):
        from scripts.compare_shortchat import pending_review_files
        plan=dict(data_root='D:/new-data',runs_root='D:/new-runs')
        self.assertEqual(len(pending_review_files(plan,dict(stage='awaiting_data_review'))),1)
        self.assertEqual(len(pending_review_files(plan,dict(stage='awaiting_dev_review',aliases=['a','b','c']))),3)
        self.assertEqual(pending_review_files(plan,dict(stage='data_shortfall')),[])
        self.assertEqual(pending_review_files(plan,dict(stage='budget_paused')),[])

if __name__=='__main__':unittest.main()
