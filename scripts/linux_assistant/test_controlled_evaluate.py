"""No-GPU regression checks for exact-input batched checkpoint evaluation."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from scripts.common import WORKSPACE
from scripts.linux_assistant.evaluate import evaluate
from scripts.linux_assistant.common import jsonl


class Tokenizer:
    def apply_chat_template(self,messages,**kwargs):
        return '|'.join(m['role']+':'+m['content'] for m in messages)+'|assistant:'
    def __call__(self,text,**kwargs):return {'input_ids':[ord(c) for c in text]}
    def decode(self,ids,**kwargs):return str(ids[0])


class ControlledEvaluateTests(unittest.TestCase):
    def fixture(self):
        rows=[{'id':str(i),'group':str(i),'kind':'task','category':'linux_operation',
               'messages':[{'role':'user','content':'question '+str(i)},
                           {'role':'assistant','content':'SECRET_TARGET_'+str(i)}],
               'reference':'SECRET_TARGET_'+str(i),'rubric':'.+'} for i in range(2)]
        text=SimpleNamespace(use_cache=False)
        model=SimpleNamespace(eval=lambda:None,config=SimpleNamespace(get_text_config=lambda:text))
        c={'system_prompt':'system','context_length':2048,'max_new_tokens':256,
           'experiment_scope':'controlled_1000_skill_rebuild'}
        return rows,text,model,c

    def test_each_adapter_gets_own_parity_check_and_complete_outputs_resume(self):
        rows,text,model,c=self.fixture()
        with tempfile.TemporaryDirectory(dir=WORKSPACE/'.cache') as tmp:
            path=Path(tmp)/'dev-63.jsonl'
            with patch('scripts.linux_assistant.evaluate.generation_eos_ids',return_value=[9]), \
                 patch('scripts.linux_assistant.evaluate.autotune',return_value={
                     'selected_batch_size':2,'eligible_fallback_batches':[1,2]}) as tune, \
                 patch('scripts.linux_assistant.evaluate.generate_batch',return_value=([[7,9],[8,9]],.2)) as generate:
                result=evaluate(c,model,Tokenizer(),rows,path,{'adapter':'hash-63','step':63})
                self.assertEqual(tune.call_args.args[6]['adapter'],'hash-63')
                self.assertEqual(generate.call_count,1)
                self.assertEqual([r['id'] for r in result],['0','1'])
                self.assertTrue(all('SECRET_TARGET' not in r['prompt'] for r in result))
                self.assertTrue(all(r['batch_size']==2 for r in result))
                self.assertFalse(text.use_cache)
                tune.reset_mock();generate.reset_mock()
                evaluate(c,model,Tokenizer(),rows,path,{'adapter':'hash-63','step':63})
                tune.assert_not_called();generate.assert_not_called()

    def test_oom_only_falls_back_to_a_parity_verified_batch(self):
        import torch
        rows,text,model,c=self.fixture()
        with tempfile.TemporaryDirectory(dir=WORKSPACE/'.cache') as tmp:
            path=Path(tmp)/'dev-125.jsonl'
            with patch('scripts.linux_assistant.evaluate.generation_eos_ids',return_value=[9]), \
                 patch('scripts.linux_assistant.evaluate.autotune',return_value={
                     'selected_batch_size':2,'eligible_fallback_batches':[1,2]}), \
                 patch('scripts.linux_assistant.evaluate.generate_batch',side_effect=[
                     torch.cuda.OutOfMemoryError('fixture'),([[7,9]],.1),([[8,9]],.1)]), \
                 patch('torch.cuda.empty_cache'):
                result=evaluate(c,model,Tokenizer(),rows,path,{'adapter':'hash-125','step':125})
            self.assertEqual(len(result),2)
            self.assertTrue(all(r['batch_size']==1 for r in result))
            summary=json.loads(path.with_suffix('.summary.json').read_text('utf8'))
            self.assertEqual(summary['oom_events'][0]['to'],1)
            self.assertFalse(text.use_cache)

    def test_sealed_final_round_never_runs_batch_tuning_on_test_questions(self):
        rows,text,model,c=self.fixture()
        with tempfile.TemporaryDirectory(dir=WORKSPACE/'.cache') as tmp:
            with patch('scripts.linux_assistant.evaluate.generation_eos_ids',return_value=[9]), \
                 patch('scripts.linux_assistant.evaluate.autotune') as tune, \
                 patch('scripts.linux_assistant.evaluate.generate_batch',side_effect=[
                     ([[7,9]],.1),([[8,9]],.1)]) as generate:
                result=evaluate(c,model,Tokenizer(),rows,Path(tmp)/'final.jsonl',{'final_round':1})
            tune.assert_not_called()
            self.assertEqual(generate.call_count,2)
            self.assertTrue(all(r['batch_size']==1 for r in result))


if __name__=='__main__':unittest.main()
