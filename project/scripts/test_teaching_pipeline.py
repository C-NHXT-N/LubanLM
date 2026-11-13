"""Focused regression checks: large extents, tails, resume, same-batch dedup."""
import json
import sqlite3
import tempfile
import unittest
import random
import hashlib
from pathlib import Path
from collections import Counter
from unittest.mock import patch
import numpy as np
import torch
from scripts.common import write_json,sha256
from training.batches import cumulative64,FullPassTokens,SFTPairs,PrefetchedSFT
from scripts.prepare_general_10b import Preparation,Writer
from scripts.prepare_sft_100k import category,messages,rows
from scripts.corpus_quality import clean_document,dialogue_reason,arithmetic,rejected_groups
from sft.template import encode_messages,prompt

class PipelineTests(unittest.TestCase):
    def test_10b_extents(self):
        self.assertEqual(cumulative64([2_000_000_000]*5),[2_000_000_000,4_000_000_000,6_000_000_000,8_000_000_000,10_000_000_000])
        with self.assertRaises(ValueError):cumulative64([2**63,1])
    def test_tail_and_resume(self):
        with tempfile.TemporaryDirectory(dir='.cache') as temp:
            root=Path(temp);shards=[]
            for i,tokens in enumerate((np.arange(1,8),np.arange(8,20))):
                path=root/f'{i}.bin';tokens.astype(np.uint16).tofile(path);shards.append(dict(path=str(path),tokens=len(tokens)))
            write_json(root/'manifest.json',dict(dtype='uint16',shards=shards,tokens=19))
            data=FullPassTokens(root,4,seed=9,token_limit=13)
            before=data.batch(1,None);state=data.state();expected=data.batch(3,None)
            restored=FullPassTokens(root,4,seed=9,token_limit=13);restored.restore(state);actual=restored.batch(3,None)
            for a,b in zip(expected[:2],actual[:2]):self.assertTrue(torch.equal(a,b))
            labels=torch.cat([before[1].flatten(),actual[1].flatten()]);self.assertEqual(sorted(labels[labels!=-100].tolist()),list(range(2,15)))
            self.assertEqual(restored.remaining,0)
            for dataset in (data,restored):
                for shard in dataset.shards:shard._mmap.close()
    def test_same_batch_near_duplicate(self):
        prep=Preparation.__new__(Preparation);prep.db=sqlite3.connect(':memory:')
        prep.db.executescript('CREATE TABLE docs(hash TEXT PRIMARY KEY,lead TEXT UNIQUE,split TEXT);CREATE TABLE near(bucket INTEGER,fp TEXT,hash TEXT);')
        counts=Counter();self.assertTrue(prep.accept_doc('a','lead-a',123456,'train',counts))
        self.assertFalse(prep.accept_doc('b','lead-b',123456^1,'validation',counts))
        # A close candidate beyond the former 128-row cap must still be found.
        fp=0x123456789abc0000;buckets=[(i<<16)|((fp>>(i*16))&65535) for i in range(4)]
        for n in range(200):prep.db.execute('INSERT INTO near VALUES (?,?,?)',(buckets[0],format((n<<16)|0xffff000000000000,'016x'),str(n)))
        prep.db.execute('INSERT INTO near VALUES (?,?,?)',(buckets[0],format(fp^0x10000,'016x'),'close'))
        self.assertFalse(prep.accept_doc('c','lead-c',fp,'train',counts));prep.db.close()
    def test_rules(self):
        self.assertIsNone(clean_document('整理房间可以从桌面开始，把不再使用的物品分类存放。随后擦拭地面，保持通道通畅。安排日常生活时，可以先列出必要的事项，再为休息留出时间。学习新的技能需要耐心，先从简单练习入手，逐渐积累经验。与朋友沟通时，认真倾听对方的想法，也清楚表达自己的需要。')[1])
        self.assertEqual(clean_document('舒曼1948年提出计划。'*20)[1],'known_factual_error')
        self.assertEqual(clean_document('介绍资料_ths_。'*20)[1],'known_scrape_defect')
        bad=[dict(role='user',content='请改写：这位患者58岁，建议先咨询医生。'),dict(role='assistant',content='这是一位58岁的男性，建议咨询医生。')]
        self.assertEqual(dialogue_reason(bad,'instruction'),'unsupported_gender')
        bad[1]['content']='这位患者58岁，建议先咨询医生。';self.assertIsNone(dialogue_reason(bad,'instruction'))
    def test_category_precedence(self):
        self.assertEqual(category('everyday.parquet','你好'),'greeting')
        self.assertEqual(category('rewrite.parquet','你是谁？'),'identity')
        self.assertTrue(arithmetic('请计算 2+3=?','2+3=5。'))
        self.assertTrue(arithmetic('计算 2/5=?','答案是2/5。'))
        self.assertFalse(arithmetic('计算 2/5=?','答案是0.5。'))
    def test_system_is_not_dropped(self):
        row={'conversations':[{'role':'system','content':'请保持简洁。'},{'role':'user','content':'你好'},{'role':'assistant','content':'你好！'}]}
        self.assertEqual(messages(row)[0]['role'],'system')
    def test_writer_durable_recovery(self):
        with tempfile.TemporaryDirectory(dir='.cache') as temp:
            root=Path(temp);writer=Writer(root,'test');writer.add([3,4,2]);state=writer.state()
            orphan=root/'99999.bin';orphan.write_bytes(b'bad')
            recovered=Writer(root,'test',state);self.assertFalse(orphan.exists());self.assertEqual(recovered.tokens,3)
            self.assertEqual(sha256(Path(state['shards'][0]['path'])),state['shards'][0]['sha256'])
    def test_prefetch_matches_consumed_resume(self):
        with tempfile.TemporaryDirectory(dir='.cache') as temp:
            path=Path(temp)/'sft.jsonl';rows=[dict(input_ids=[1,10+i,2],labels=[-100,10+i,2],task='daily',encoding_version='prefix-stable-v1') for i in range(7)]
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
            g=torch.Generator().manual_seed(19);data=SFTPairs(path,0,g,batching='bucketed');prefetch=PrefetchedSFT(data,2)
            prefetch.batch(2,g);state=data.state();rng=g.get_state();expected=prefetch.batch(2,g);prefetch.close()
            g2=torch.Generator();restored=SFTPairs(path,0,g2,batching='bucketed');restored.restore(state);g2.set_state(rng)
            for a,b in zip(expected,restored.batch(2,g2)):self.assertTrue(torch.equal(a,b))
            data.rows.close();restored.rows.close()
    def test_known_bad_groups_registered(self):
        self.assertGreaterEqual(len(rejected_groups()),7)
    def test_streaming_json_array(self):
        with tempfile.TemporaryDirectory(dir='.cache') as temp:
            path=Path(temp)/'input.json';expected=[dict(instruction='你好'*600000,output='好'),dict(instruction='第二条',output='回答')]
            path.write_text(json.dumps(expected,ensure_ascii=False),encoding='utf-8');self.assertEqual(list(rows(path)),expected)
            path.write_text('[{"x":1}',encoding='utf-8')
            with self.assertRaises(ValueError):list(rows(path))
    def test_source_caps_do_not_starve_categories(self):
        from scripts.balanced_sft import select_balanced
        pool=[dict(task='daily',source='A'),dict(task='daily',source='B'),dict(task='concept',source='A')]
        chosen=select_balanced(pool,dict(daily=1,concept=1),dict(A=.5,B=.5),2)
        self.assertEqual(Counter(r['task'] for r in chosen),dict(daily=1,concept=1))
        self.assertEqual(Counter(r['source'] for r in chosen),dict(A=1,B=1))
    def test_content_gate_rejects_stale_and_incomplete_review(self):
        from scripts.content_review_gate import verify_content_review
        with tempfile.TemporaryDirectory(dir='.cache') as temp:
            root=Path(temp);plan=root/'plan.json';sample=root/'sample.json';decision=root/'reports/pretrain_content_review.json'
            write_json(plan,dict(runs_root=str(root)))
            samples=[dict(group=str(i),split='train') for i in range(200)];write_json(sample,dict(samples=samples))
            value=dict(policy_identity={'id':1},samples_sha256=sha256(sample),artifact_sha256={'train':'digest'},marks=[dict(r,result='pass') for r in samples]);write_json(decision,value)
            with patch('scripts.content_review_gate.policy_identity',return_value={'id':1}),patch('scripts.content_review_gate.review_artifacts',return_value=(sample,{'train':'digest'})):
                self.assertEqual(verify_content_review(plan,'pretrain')['inspected'],200)
                value['samples_sha256']='stale';write_json(decision,value)
                with self.assertRaises(ValueError):verify_content_review(plan,'pretrain')
                value['samples_sha256']=sha256(sample);value['marks'].pop();write_json(decision,value)
                with self.assertRaises(ValueError):verify_content_review(plan,'pretrain')

    def test_partial_file_resume_and_uncommitted_index(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        from scripts.prepare_general_2b import key
        from eval.chat_suite import normalize
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory(dir='.cache') as temp:
            root=Path(temp).resolve();metadata=root/'metadata.json';parquet=root/'input.parquet'
            rng=random.Random(33);texts=[]
            while len(texts)<8:
                text=''.join(chr(rng.randrange(0x4e00,0x9fa5)) for _ in range(180))
                if int(key(normalize(text)[:160])[:8],16)%100>=2:texts.append(text)
            pq.write_table(pa.Table.from_pylist([dict(text=t,score=1.0) for t in texts]),parquet)
            write_json(metadata,dict(sha='pinned',siblings=[dict(rfilename='input.parquet',size=parquet.stat().st_size,lfs=dict(sha256=sha256(parquet)))]))
            prep=Preparation.__new__(Preparation);prep.data=root/'data';prep.runs=root/'runs';prep.policy={'version':'test'};prep.plan={'seed':33,'tokenizer_sha256':'test','validation_tokens_per_source_cap':10000}
            prep.tokenizer=SimpleNamespace(encode_batch=lambda texts,**kw:[SimpleNamespace(ids=[3]*128) for t in texts])
            prep.db=sqlite3.connect(':memory:');prep.db.executescript('CREATE TABLE docs(hash TEXT PRIMARY KEY,lead TEXT UNIQUE,split TEXT);CREATE TABLE near(bucket INTEGER,fp TEXT,hash TEXT);CREATE TABLE checkpoints(name TEXT PRIMARY KEY,state TEXT);')
            prep.blocked=[];prep.guard=lambda:None;prep.progress=lambda **kw:None;prep.files=lambda source,files:iter((f,parquet) for f in files)
            source=dict(name='test',repo='test/repo',metadata=str(metadata),revision='pinned',prefix='',score_min=.8,train_tokens_target=200)
            first=prep.source(source);self.assertEqual(first['train']['tokens'],258);self.assertEqual(first['done_files'],[])
            index=prep.data/'processed/test/document-index.jsonl'
            with index.open('ab') as f:f.write(b'uncommitted trailing bytes')
            second=prep.source(source,500);self.assertEqual(second['train']['tokens'],516)
            records=[json.loads(line) for line in index.read_text().splitlines()]
            self.assertEqual(len(records),4);self.assertEqual(len({r['hash'] for r in records}),4)
            self.assertEqual([r['token_offset'] for r in records],[0,129,258,387])
            final=prep.source(source,2000);self.assertEqual(final['done_files'],['input.parquet']);self.assertEqual(final['train']['documents'],8)
            prep.db.close()

    def test_assistant_mask_and_system_prefix(self):
        from tokenizers import Tokenizer
        tokenizer=Tokenizer.from_file('D:/LubanLM-Data/chat-v2/tokenizers/vocab-32000/tokenizer.json')
        convo=[dict(role='user',content='你好'),dict(role='assistant',content='你好！')]
        encoded=encode_messages(tokenizer,convo,system='请简洁作答。')
        prefix=tokenizer.encode(prompt('你好','请简洁作答。'),add_special_tokens=False).ids
        self.assertEqual(encoded['input_ids'][:len(prefix)],prefix)
        self.assertTrue(all(label==-100 for label in encoded['labels'][:len(prefix)]))
        self.assertEqual(encoded['labels'][-1],2)

if __name__=='__main__':unittest.main()
