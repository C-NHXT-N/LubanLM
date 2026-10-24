"""Regression tests for answer leakage, EOS trimming, padding and resume input identity."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from scripts.common import WORKSPACE

from scripts.linux_assistant.baseline import trim_generation,generate_batch,prepare,resume_rows,atomic_rows,validate_batch_candidates
from scripts.linux_assistant.modeling import evaluation_messages


class Tokenizer:
    pad_token_id=0
    def apply_chat_template(self,messages,**kwargs):
        return '|'.join(m['role']+':'+m['content'] for m in messages)+'|assistant:'
    def __call__(self,text,**kwargs):return {'input_ids':[ord(c) for c in text]}


class BaselineTests(unittest.TestCase):
    def test_batch_parity_reference_must_be_single_example(self):
        self.assertEqual(validate_batch_candidates({'baseline':{'batch_candidates':[1,2,4]}}),[1,2,4])
        for invalid in ([4,2,1],[2,4],[1,1,2],[True,2],[],[1,0]):
            with self.subTest(invalid=invalid),self.assertRaises(ValueError):
                validate_batch_candidates({'baseline':{'batch_candidates':invalid}})

    def test_target_excluded_and_previous_assistant_retained(self):
        row={'id':'case','messages':[{'role':'user','content':'old'},
             {'role':'assistant','content':'history evidence'}, {'role':'user','content':'question'},
             {'role':'assistant','content':'UNIQUE_SECRET_TARGET'}]}
        c={'system_prompt':'system','context_length':2048,'max_new_tokens':256}
        p=prepare(c,Tokenizer(),[row])[0]
        self.assertNotIn('UNIQUE_SECRET_TARGET',p['prompt'])
        self.assertIn('history evidence',p['prompt'])
        self.assertEqual(p['context'][-1]['role'],'user')
        self.assertEqual(row['messages'][-1]['content'],'UNIQUE_SECRET_TARGET')
        with self.assertRaises(ValueError):evaluation_messages(c,{'id':'bad','messages':[]})

    def test_trim_stops_at_first_native_end_and_keeps_length_limit(self):
        self.assertEqual(trim_generation([4,9,0,0],[9,10]),[4,9])
        self.assertEqual(trim_generation([4,10,9],[9,10]),[4,10])
        self.assertEqual(trim_generation([4,5],[9]),[4,5])

    def test_generate_batch_left_padding_and_per_row_stop(self):
        import torch
        captured={}
        class Model:
            device='cpu'
            def generate(self,**kwargs):
                captured.update(kwargs)
                return torch.cat([kwargs['input_ids'],torch.tensor([[7,9,0],[8,8,9]])],dim=1)
        p=[{'input_ids':[1,2]},{'input_ids':[3]}]
        with patch('torch.cuda.synchronize'):
            ids,_=generate_batch(Model(),Tokenizer(),p,3,[9])
        self.assertEqual(captured['input_ids'].tolist(),[[1,2],[0,3]])
        self.assertEqual(captured['attention_mask'].tolist(),[[1,1],[0,1]])
        self.assertEqual(ids,[[7,9],[8,8,9]])
        self.assertFalse(captured['generation_config'].do_sample)

    def test_resume_rejects_changed_prompt_and_duplicate_ids(self):
        p={'row':{'id':'x','messages':[{'role':'user','content':'q'}]},'context':[
            {'role':'system','content':'s'},{'role':'user','content':'q'}], 'prompt':'p','input_ids':[1]}
        row={'id':'x','messages':p['row']['messages'],'input_messages':p['context'][1:], 'prompt':'p','input_ids':[1]}
        with tempfile.TemporaryDirectory(dir=WORKSPACE/'.cache') as tmp:
            path=Path(tmp)/'answers.jsonl';atomic_rows(path,[row])
            self.assertEqual(list(resume_rows(path,[p])),['x'])
            changed=copy.deepcopy(row);changed['input_ids']=[2];atomic_rows(path,[changed])
            with self.assertRaises(ValueError):resume_rows(path,[p])
            atomic_rows(path,[row,row])
            with self.assertRaises(ValueError):resume_rows(path,[p])


if __name__=='__main__':unittest.main()
