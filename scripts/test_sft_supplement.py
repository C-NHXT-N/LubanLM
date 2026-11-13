"""Supplement parser and quota regressions; isolated fixtures only."""
import unittest
from collections import Counter
from scripts.prepare_sft_100k import messages,candidates
from scripts.balanced_sft import soft_allocations
from scripts.common import ROOT,read_json
from scripts.policy_versions import pretrain_identity

class SupplementTests(unittest.TestCase):
    def test_pretrain_startup_does_not_rebuild_discarded_index(self):
        import tempfile,sqlite3
        from pathlib import Path
        from unittest.mock import patch
        from scripts.prepare_general_10b import Preparation
        from scripts.common import write_json
        with tempfile.TemporaryDirectory(dir=ROOT/'.cache') as t:
            root=Path(t);legacy=root/'legacy/tokenizers/vocab-32000';legacy.mkdir(parents=True)
            (legacy/'tokenizer.json').write_text('{}',encoding='utf-8')
            path=root/'plan.json';write_json(path,dict(data_root=str(root/'data'),runs_root=str(root/'runs'),legacy_data=str(root/'legacy'),tokenizer_sha256='fixture'))
            statements=[];original=sqlite3.connect
            def traced(*a,**k):
                db=original(*a,**k);db.set_trace_callback(statements.append);return db
            with patch('scripts.prepare_general_10b.PhysicalMirror'),patch('scripts.prepare_general_10b.policy_identity',return_value={'fixture':1}),patch('scripts.prepare_general_10b.sha256',return_value='fixture'),patch('scripts.prepare_general_10b.Tokenizer'),patch('scripts.prepare_general_10b.sqlite3.connect',side_effect=traced):
                preparation=Preparation(path);preparation.db.close()
            self.assertFalse(any('CREATE INDEX' in s and 'near_bucket' in s for s in statements))
            self.assertTrue(any('CREATE INDEX' in s and 'near_covering' in s for s in statements))
    def test_quota_cache_invalidates_on_new_candidate_or_settings(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from scripts.sft_disk import connect,assign_splits
        from scripts.sft_supplement import quota_ready
        with tempfile.TemporaryDirectory(dir=ROOT/'.cache') as t:
            db=connect(Path(t)/'pool.sqlite')
            for i in range(40):
                k=str(i).zfill(4);db.execute('INSERT INTO candidates VALUES (?,?,?,?,?,?,?,?,NULL,0)',(k,k,k,k,'daily','other',k,'{}'))
            db.commit()
            plan=dict(seed=1,sft=dict(source_fraction_caps={},train_minimum=20,daily_minimum=10,concept_maximum=6,validation_examples=6,test_examples=6))
            with patch('scripts.sft_disk.assign_splits',wraps=assign_splits) as split:
                self.assertTrue(quota_ready(db,plan)[0]);self.assertTrue(quota_ready(db,plan)[0]);self.assertEqual(split.call_count,1)
                db.execute('INSERT INTO candidates VALUES (?,?,?,?,?,?,?,?,NULL,0)',('new','new','new','new','daily','other','new','{}'));db.commit()
                quota_ready(db,plan);self.assertEqual(split.call_count,2)
                plan['sft']['daily_minimum']=11;quota_ready(db,plan);self.assertEqual(split.call_count,3)
            db.close()
    def test_moss_zip_preserves_text_and_excludes_tool_or_thought(self):
        import tempfile,json,zipfile
        from pathlib import Path
        from scripts.prepare_sft_100k import rows
        from scripts.public_sft_formats import moss_dialogue
        turn={'Human':'<|Human|>: 工作压力怎么办？<eoh>\n','MOSS':'<|MOSS|>: 可以先休息，再分步骤处理。<eom>\n','Commands':'<|Commands|>: None<eoc>\n','Tool Responses':'<|Results|>: None<eor>\n','Inner Thoughts':'<|Inner Thoughts|>: None<eot>\n'}
        row=dict(num_turns=1,chat=dict(turn_1=turn))
        with tempfile.TemporaryDirectory(dir=ROOT/'.cache') as t:
            path=Path(t)/'data.zip'
            with zipfile.ZipFile(path,'w') as archive:archive.writestr('moss-003-sft-no-tools.jsonl',json.dumps(row,ensure_ascii=False)+'\n')
            self.assertEqual(messages(next(rows(path))),[dict(role='user',content='工作压力怎么办？'),dict(role='assistant',content='可以先休息，再分步骤处理。')])
        for field,value in [('Commands','<|Commands|>: Search(foo)<eoc>'),('Inner Thoughts','<|Inner Thoughts|>: private trace<eot>')]:
            changed=dict(num_turns=1,chat=dict(turn_1=dict(turn,**{field:value})))
            self.assertEqual(moss_dialogue(changed),{})
    def test_supplement_rejects_daily_tail_on_non_daily_context(self):
        row=dict(conversations=[dict(role='user',content='什么是三角形？'),dict(role='assistant',content='这是几何图形。'),dict(role='user',content='工作压力怎么办？'),dict(role='assistant',content='可以休息一下。')])
        count=Counter()
        self.assertEqual(list(candidates(row,dict(repo='new',first_daily_only=True),'x','x',1,{},None,None,set(),set(),count)),[])
        self.assertEqual(count['supplement_not_daily_context'],1)
    def test_reuse_rejects_changed_legacy_parser(self):
        import tempfile
        from pathlib import Path
        from scripts.reuse_sft_candidates import compatible_parser,BACKUP
        old=BACKUP/'scripts/prepare_sft_100k.py';new=ROOT/'scripts/prepare_sft_100k.py'
        unchanged=BACKUP.parent/'sft-context-repair-20261007/scripts/prepare_sft_100k.py'
        compatible_parser(old,unchanged)
        # Postfiltering intentionally changed supervision; the old shortcut
        # must reject it and require the journaled full candidate migration.
        with self.assertRaises(AssertionError):compatible_parser(old,new)
        with tempfile.TemporaryDirectory(dir=ROOT/'.cache') as t:
            path=Path(t)/'changed.py'
            path.write_text(unchanged.read_text(encoding='utf-8-sig').replace("return 'identity'","return 'daily'",1),encoding='utf-8')
            with self.assertRaises(AssertionError):compatible_parser(old,path)
    def test_stale_benchmark_never_reused(self):
        import tempfile
        from pathlib import Path
        from scripts.run_general_10b import benchmark_matches
        from scripts.common import write_json
        with tempfile.TemporaryDirectory(dir=ROOT/'.cache') as t:
            path=Path(t)/'bench.json';write_json(path,dict(data_sha256='old',checkpoint_sha256='parent'))
            self.assertFalse(benchmark_matches(path,dict(data_sha256='new',checkpoint_sha256='parent')))
            self.assertTrue(benchmark_matches(path,dict(data_sha256='old',checkpoint_sha256='parent')))
    def test_sft_near_query_is_covering(self):
        import sqlite3,tempfile
        from pathlib import Path
        from scripts.sft_disk import connect
        with tempfile.TemporaryDirectory(dir=ROOT/'.cache') as t:
            db=connect(Path(t)/'pool.sqlite')
            query='SELECT DISTINCT fp FROM near WHERE bucket IN (?,?,?,?)'
            self.assertIn('COVERING',str(list(db.execute('EXPLAIN QUERY PLAN '+query,(1,2,3,4)))))
            db.close()
    def test_public_human_dialogue_complete_prefix(self):
        self.assertEqual(messages(dict(dialog=['心情 不好','可以 休息 一下','谢谢'])),[dict(role='user',content='心情不好'),dict(role='assistant',content='可以休息一下')])
    def test_token_removal_across_shard_boundary(self):
        from scripts.repair_pretrain_review import without_intervals
        import numpy as np
        values=np.arange(10,dtype=np.uint16)
        self.assertEqual(np.concatenate(without_intervals(values[:5],0,[(3,7)])).tolist(),[0,1,2])
        self.assertEqual(np.concatenate(without_intervals(values[5:],5,[(3,7)])).tolist(),[7,8,9])
    def test_sharegpt_pair_and_wildchat_roles(self):
        expected=[dict(role='user',content='你好'),dict(role='assistant',content='您好')]
        self.assertEqual(messages(dict(conversation=[dict(human=' 你好 ',assistant='您好')])),expected)
        self.assertEqual(messages(dict(conversation=[dict(role='user',content='你好'),dict(role='assistant',content='您好',language='Chinese')])),expected)
    def test_legacy_unchanged_and_malformed_pair(self):
        expected=[dict(role='user',content='问题'),dict(role='assistant',content='回答')]
        self.assertEqual(messages(dict(instruction='问题',output='回答')),expected)
        self.assertEqual(messages(dict(conversations=[dict(from_='unused',role='user',content='问题'),dict(from_='unused',role='assistant',content='回答')])),expected)
        self.assertEqual(messages(dict(conversation=[dict(human=None,assistant='回答')])),[])
    def test_wildchat_flags_filter_before_encoding(self):
        for row in (dict(language='English'),dict(language='Chinese',toxic=True),dict(language='Chinese',conversation=[dict(redacted=True)])):
            count=Counter()
            self.assertEqual(list(candidates(row,dict(repo='allenai/WildChat-1M'),'x','x',1,{},None,None,set(),set(),count)),[])
            self.assertEqual(count['wildchat_language_or_flag'],1)
    def test_supplement_fills_daily_with_caps_kept(self):
        counts={('daily','Belle'):40957,('daily','other'):12344,('instruction','other'):90000,('concept','other'):80000}
        _,gaps=soft_allocations(counts,{'Belle':.25},100000,50000,30000)
        self.assertEqual(gaps['daily_gap'],12656)
        counts[('daily','new')]=16000
        allocation,gaps=soft_allocations(counts,{'Belle':.25},100000,50000,30000)
        self.assertFalse(any(n>0 for n in gaps.values()))
        self.assertEqual(sum(allocation.values()),100000)
        self.assertLessEqual(sum(n for (t,s),n in allocation.items() if s=='Belle'),25000)
    def test_pretrain_manifest_identity_preserved(self):
        p=ROOT/'configs/50m-general-pretrain-10b-teaching.json'
        old=ROOT/'docs/run-results/teaching-repair-20261006/minimal-before'
        self.assertEqual(pretrain_identity(p),pretrain_identity(old/'plan.json',old))

if __name__=='__main__':unittest.main()
