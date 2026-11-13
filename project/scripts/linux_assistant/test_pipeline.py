"""Meaningful guards for leakage, masking, acceptance, template parity and history bounds."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import io
import hashlib
from types import SimpleNamespace
from unittest.mock import patch
from scripts.common import ROOT
from scripts.linux_assistant.catalog import cases
from scripts.linux_assistant.common import config,jsonl,screen,gate,reviewed_scores,verify_final_rows,training_rows
from scripts.linux_assistant.modeling import encode_training,Padding,generation_eos_ids,evaluation_messages,render
from scripts.linux_assistant.parity import frontend


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        cls.c=config();cls.t=AutoTokenizer.from_pretrained(cls.c['model'],local_files_only=True)
        cls.front=frontend();cls.light=cls.front.NativeTokenizer(cls.c['model'])

    def test_dataset_counts_and_group_isolation(self):
        manifest=json.loads((self.c['data']/'manifest.json').read_text(encoding='utf8'))
        groups={}
        split_counts={}
        for filename in manifest['data_splits']:
            split=Path(filename).stem.replace('-','_')
            rows=jsonl(self.c['data']/filename)
            split_counts[split]=len(rows)
            groups[split]={r['group'] for r in rows}
        self.assertEqual(split_counts,manifest['split_counts'])
        nonempty={name:value for name,value in groups.items() if split_counts[name]}
        names=list(nonempty)
        for i,left in enumerate(names):
            for right in names[i+1:]:
                self.assertFalse(nonempty[left]&nonempty[right],f'group leaks across {left}/{right}')
        train=training_rows(self.c)
        multiturn=sum(len(r['messages'])>2 for r in train)
        self.assertEqual(multiturn,manifest['training_multiturn_records'])

    def test_final_assistant_first_token_and_eos_supervised_only(self):
        row=jsonl(self.c['data']/'train.jsonl')[0]
        e=encode_training(self.c,self.t,row);labels=e['labels'];indices=[i for i,l in enumerate(labels) if l!=-100]
        self.assertTrue(indices);self.assertGreater(indices[0],0)
        self.assertEqual(labels[indices[-1]],self.t.convert_tokens_to_ids('<|im_end|>'))
        decoded=self.t.decode([labels[i] for i in indices],skip_special_tokens=True)
        self.assertEqual(decoded,row['reference'])
        self.assertTrue(all(x==-100 for x in labels[:indices[0]]))

    def test_padding_does_not_mask_real_eos(self):
        pad=self.t.pad_token_id
        a={'input_ids':[4,pad],'attention_mask':[1,1],'labels':[-100,pad]}
        b={'input_ids':[4],'attention_mask':[1],'labels':[4]}
        result=Padding(pad)([a,b]);self.assertEqual(result['labels'][0,-1].item(),pad)
        self.assertEqual(result['labels'][1,-1].item(),-100)

    def test_real_template_excludes_target_but_preserves_multiturn_context(self):
        row={'id':'leak-guard','messages':[{'role':'user','content':'系统是Linux'},
            {'role':'assistant','content':'历史上下文保留'}, {'role':'user','content':'怎样检查内核？'},
            {'role':'assistant','content':'UNIQUE_REFERENCE_DO_NOT_SEND'}]}
        prompt=render(self.t,evaluation_messages(self.c,row),True)
        self.assertNotIn('UNIQUE_REFERENCE_DO_NOT_SEND',prompt)
        self.assertIn('历史上下文保留',prompt)
        self.assertIn('怎样检查内核？',prompt)

    def test_training_and_eval_both_reject_duplicated_target_in_history(self):
        row={'id':'duplicate-target','messages':[
            {'role':'user','content':'检查内核版本。'},
            {'role':'assistant','content':'uname -r'},
            {'role':'assistant','content':'uname -r'}], 'reference':'uname -r'}
        with self.assertRaisesRegex(ValueError,'conversation sequence'):
            encode_training(self.c,self.t,row)
        with self.assertRaisesRegex(ValueError,'conversation sequence'):
            evaluation_messages(self.c,row)

    def test_native_assistant_end_is_not_replaced_by_text_config_eos(self):
        native=self.t.convert_tokens_to_ids('<|im_end|>')
        text_end=self.t.convert_tokens_to_ids('<|endoftext|>')
        self.assertNotEqual(native,text_end)
        model=SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=text_end))
        ids=generation_eos_ids(model,self.t)
        self.assertEqual(ids[0],native);self.assertIn(text_end,ids)

    def test_no_silent_truncation_or_empty_supervision(self):
        r=jsonl(self.c['data']/'train.jsonl')[0];r['messages'][-1]['content']=''
        with self.assertRaises(ValueError):encode_training(self.c,self.t,r)
        r['messages'][-1]['content']='文字'*2000
        with self.assertRaises(ValueError):encode_training(self.c,self.t,r)

    def test_screen_is_not_acceptance_and_exact_is_strict(self):
        r={'rubric':'EXACT:pwd'}
        self.assertTrue(screen(r,'pwd')['screen_pass']);self.assertFalse(screen(r,'pwd\n')['screen_pass'])
        self.assertIsNone(screen(r,'pwd')['accepted'])
        with self.assertRaises(ValueError):reviewed_scores([{'kind':'task','accepted':None}])
        self.assertFalse(screen({'rubric':'JSON:verified=false'},'{"verified":0}')['screen_pass'])

    def test_threshold_edges(self):
        t=self.c['acceptance']
        self.assertTrue(gate(.70,.80,.9,.85,t));self.assertFalse(gate(.71,.80,.9,.85,t))
        self.assertTrue(gate(.80,.86,.9,.9,t));self.assertFalse(gate(.8,.84,.9,.9,t))
        self.assertFalse(gate(.70,.80,.05,.90,t,.8,.7))
        self.assertTrue(gate(.70,.80,.05,.90,t,.8,.75))

    def test_review_cannot_waive_missing_conditions_without_equivalent_evidence(self):
        task={'id':'task','kind':'task','accepted':True,'review_reason':'checked','failed_conditions':['实测']}
        general={'id':'general','kind':'general','accepted':True,'review_reason':'checked','content_correct':True,'failed_conditions':[]}
        with self.assertRaises(ValueError):reviewed_scores([task,general])
        task['equivalents']={'实测':'回答明确要求实际运行并测量单核和双核吞吐，语义满足实测条件。'}
        self.assertEqual(reviewed_scores([task,general])['task'],1)
        task['failed_conditions']=['EXACT:abc'];task['equivalents']['EXACT:abc']='meaning unchanged'
        with self.assertRaises(ValueError):reviewed_scores([task,general])

    def test_final_round_rejects_development_rows_or_changed_answer(self):
        question={'id':'final-1','messages':[],'kind':'task','group':'held-out','category':'linux','rubric':'EXACT:pwd','reference':'pwd'}
        raw={**question,'answer':'pwd','input_ids':[1,2],'generated_ids':[3]}
        reviewed={**raw,'accepted':True}
        verify_final_rows([question],[raw],[reviewed])
        with self.assertRaises(ValueError):verify_final_rows([question],[{**raw,'id':'dev-1'}],[reviewed])
        with self.assertRaises(ValueError):verify_final_rows([question],[raw],[{**reviewed,'answer':'fixed by reviewer'}])

    def test_review_reuse_rejects_changed_question_and_conflicting_verdicts(self):
        from scripts.linux_assistant.review_template import template
        row={'id':'x','messages':[{'role':'user','content':'只输出pwd'}],'reference':'pwd','rubric':'EXACT:pwd',
             'kind':'task','category':'linux','answer':'pwd','accepted':True,'reviewer':'author','review_reason':'checked'}
        sources=[(Path('first'), 'hash', [row])]
        self.assertTrue(template([row],sources)[0]['accepted'])
        self.assertIsNone(template([{**row,'messages':[]}],sources)[0]['accepted'])
        sources.append((Path('conflict'),'other',[{**row,'accepted':False}]))
        self.assertIsNone(template([row],sources)[0]['accepted'])

    def test_history_trims_complete_pairs_and_preserves_current(self):
        history=[{'role':'user','content':'旧问题'*200},{'role':'assistant','content':'旧答案'*200}]
        retained,prompt,ids=self.front.bounded_history(self.light,self.c['system_prompt'],history,'现在的问题',256)
        self.assertEqual(retained,[]);self.assertIn('现在的问题',prompt);self.assertIn(self.c['system_prompt'],prompt)
        self.assertEqual(len(history),2)
        with self.assertRaises(ValueError):self.front.bounded_history(self.light,'',[],' ',256)
        with self.assertRaises(ValueError):self.front.bounded_history(self.light,'',[],'长'*1000,32)

    def test_bundle_rejects_tampering_and_escape(self):
        self.c['report'].mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=self.c['report']) as d:
            root=Path(d);(root/'x').write_text('changed')
            (root/'bundle-manifest.json').write_text(json.dumps({'files':{'x':'bad'}}))
            with self.assertRaises(ValueError):self.front.verify_bundle(root)
            (root/'bundle-manifest.json').write_text(json.dumps({'files':{'../outside':'bad'}}))
            with self.assertRaises(ValueError):self.front.verify_bundle(root)

    def test_utf8_character_can_cross_sdk_callback_boundaries(self):
        events=[{'event':'chunk','id':'1','text_hex':'e4b8'},
                {'event':'chunk','id':'1','text_hex':'ad'},
                {'event':'done','id':'1','generate_tokens':1}]
        worker=self.front.Runner.__new__(self.front.Runner)
        worker.process=SimpleNamespace(stdin=io.StringIO(),stdout=io.StringIO(''.join(json.dumps(x)+'\n' for x in events)))
        worker.counter=0;worker.peak_combined_kib=1
        pieces=[];result=worker.generate([1,2],pieces.append)
        self.assertEqual(result['answer'],'中');self.assertEqual(''.join(pieces),'中')
        self.assertIsNotNone(result['end_to_end_first_text_seconds'])

    def test_worker_protocol_rejects_cross_request_response(self):
        worker=self.front.Runner.__new__(self.front.Runner)
        worker.process=SimpleNamespace(stdin=io.StringIO(),stdout=io.StringIO('{"event":"done","id":"wrong"}\n'))
        worker.counter=0;worker.peak_combined_kib=1
        with self.assertRaises(RuntimeError):worker.generate([1])

    def test_resumable_receiver_hashes_and_promotes_only_complete_file(self):
        import sys
        from scripts.linux_assistant.transfer import REMOTE
        with tempfile.TemporaryDirectory(dir=self.c['report']) as d:
            root=Path(d);payload='中文 model data'.encode();partial=root/'model.rkllm.part'
            partial.write_bytes(payload[:4]);expected=hashlib.sha256(payload).hexdigest();out=io.StringIO()
            with patch.object(sys,'argv',['receiver',str(root),'model.rkllm',expected,str(len(payload))]), \
                 patch.object(sys,'stdin',SimpleNamespace(buffer=io.BytesIO(payload[4:]))),patch.object(sys,'stdout',out):
                exec(compile(REMOTE,'receiver','exec'),{})
            records=[json.loads(s) for s in out.getvalue().splitlines()]
            self.assertEqual(records[0]['offset'],4);self.assertTrue(records[1]['verified'])
            self.assertEqual((root/'model.rkllm').read_bytes(),payload);self.assertFalse(partial.exists())

    def test_resumable_receiver_preserves_incomplete_upload(self):
        import sys
        from scripts.linux_assistant.transfer import REMOTE
        with tempfile.TemporaryDirectory(dir=self.c['report']) as d:
            root=Path(d);payload=b'correct';expected=hashlib.sha256(payload).hexdigest()
            with patch.object(sys,'argv',['receiver',str(root),'model.rkllm',expected,str(len(payload))]), \
                 patch.object(sys,'stdin',SimpleNamespace(buffer=io.BytesIO(b'cor'))),patch.object(sys,'stdout',io.StringIO()):
                with self.assertRaises(EOFError):exec(compile(REMOTE,'receiver','exec'),{})
            self.assertFalse((root/'model.rkllm').exists())
            self.assertEqual((root/'model.rkllm.part').read_bytes(),b'cor')


if __name__=='__main__':unittest.main()
