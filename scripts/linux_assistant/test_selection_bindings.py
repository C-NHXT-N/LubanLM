"""Artifact-integrity and prerequisite regression tests using real bound files."""
from __future__ import annotations
import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.common import WORKSPACE, sha256
from scripts.linux_assistant.common import fingerprint
from scripts.linux_assistant import selection_bindings as bindings
from scripts.linux_assistant.select_controlled import _identity, select

SCRATCH = WORKSPACE / '.cache' / 'selection-bindings-tests'


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False)+'\n', encoding='utf8')


def plain(value):
    if isinstance(value, Path): return str(value.resolve())
    if isinstance(value, dict): return {k:plain(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [plain(v) for v in value]
    return value


def build_fixture(root, run_name='lr-1e-4'):
    """Create all actual inputs needed to bind four checkpoints; no GPU/model load."""
    root = Path(root)
    c = {'data':root/'data', 'report':root/'report', 'runs':root/'runs',
         'model':root/'model', 'baseline_report':str(root/'baseline'),
         'revision':'fixed-model-revision', 'system_prompt':'system', 'max_new_tokens':256,
         'context_length':2048, 'seed':20261019, 'selection_protocol_id':'axes-protocol',
         'experiment_scope':'controlled_1000_skill_rebuild',
         'training':{'checkpoint_steps':[63,125,188,250], 'learning_rate':1e-4,
                     'second_learning_rate':5e-5, 'optimizer_mode':'adamw_fused'},
         'dataset_contract':{'evaluation':{'total':3}},
         'acceptance':{'task_min':.8,'command_min':.8,'troubleshooting_min':.8,
                       'task_gain':.1,'error_reduction':.3,'general_max_drop':.05}}
    manifest = {'data_splits':['dev.jsonl','general-dev.jsonl']}
    write_json(c['data']/'manifest.json', manifest)
    questions = []
    for i,capability in enumerate(('command_generation','log_troubleshooting','general')):
        questions.append({'id':f'q{i}','group':f'g{i}','kind':'general' if capability=='general' else 'task',
             'category':'general' if capability=='general' else 'linux_operation',
             'messages':[{'role':'user','content':f'question-{i}'},{'role':'assistant','content':'answer'}],
             'reference':'answer','rubric':{},'grading':{'format':{'kind':'free'},'content':{}},
             'evaluation_capability':capability})
    def write_rows(path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf8')
    write_rows(c['data']/'dev.jsonl', questions[:2])
    write_rows(c['data']/'general-dev.jsonl', questions[2:])
    rows = [{**q,'prompt':f'prompt-{i}','input_ids':[i+1], 'input_messages':q['messages'][:-1],
             'answer':'answer','generated_ids':[42], 'stop_reason':'eos'} for i,q in enumerate(questions)]
    reviewed = [{**r,'axes':{'content_correct':True,'completeness':True,'format_compliant':True},
                  'accepted':True,'review_reason':'Actual answer is correct for visible question',
                  'reviewer':'independent-model-reviewer','scoring_protocol_id':'axes-protocol',
                  'final_conclusion_correct':True} for r in rows]
    baseline = Path(c['baseline_report'])
    baseline_identity = {'model':str(c['model']), 'revision':c['revision'], 'rows':fingerprint(questions),
         'system_prompt':c['system_prompt'], 'max_new_tokens':256,'context_length':2048,
         'greedy':True,'thinking':False,'prompt_policy':'context_through_last_user_exclude_target_assistant',
         'code_sha256':{name:sha256(Path(bindings.__file__).with_name(name)) for name in ('baseline.py','modeling.py')}}
    write_json(baseline/'identity.json', baseline_identity)
    write_rows(baseline/'answers.jsonl', rows)
    write_rows(baseline/'answers.reviewed.jsonl', reviewed)
    train_identity = {'lr':c['training']['second_learning_rate'] if run_name=='lr-5e-5' else c['training']['learning_rate'],
         'revision':c['revision'], 'manifest':fingerprint(manifest),
         'config':fingerprint({k:v for k,v in c.items() if k not in ('report','export','runs','data','model','comparison_model')})}
    # train's config fingerprint includes baseline_report as a string; other retained values are JSON values.
    run = c['runs']/run_name
    write_json(run/'configuration.json', {'identity':train_identity,'resolved_configuration':plain(c),
                                         'dataset_manifest_sha256':sha256(c['data']/'manifest.json')})
    candidates=[]
    for step in c['training']['checkpoint_steps']:
        cp=run/f'checkpoint-{step}'
        cp.mkdir(parents=True, exist_ok=True)
        (cp/'adapter_model.safetensors').write_bytes(f'adapter-{step}'.encode())
        write_json(cp/'adapter_config.json', {'base_model_name_or_path':str(c['model']), 'r':16,'lora_alpha':32})
        write_json(cp/'run-state.json', {'step':step,'identity':train_identity})
        raw_path=run/f'dev-{step}.jsonl'; review_path=run/f'dev-{step}.reviewed.jsonl'
        write_rows(raw_path,rows);write_rows(review_path,reviewed)
        identity={'step':step,'adapter':sha256(cp/'adapter_model.safetensors'),
            'adapter_config_sha256':sha256(cp/'adapter_config.json'), 'data_hash':fingerprint(questions),
            'model':str(c['model']),'revision':c['revision'],'context_length':2048,
            'system':'system','prompt_policy':'context_through_last_user_exclude_target_assistant',
            'generation':{'max_new_tokens':256,'greedy':True,'thinking':False,
                          'eos_token_ids':[42],'stop_policy':'native_assistant_end_or_text_end_v2'}}
        write_json(run/f'dev-{step}.identity.json',identity)
        candidates.append({'step':step,'passed':False,'adapter':str(cp),
            'adapter_sha256':identity['adapter'],'adapter_config_sha256':identity['adapter_config_sha256'],
            'raw_sha256':sha256(raw_path),'review_sha256':sha256(review_path),
            'identity_sha256':sha256(run/f'dev-{step}.identity.json')})
    c['report'].mkdir(parents=True,exist_ok=True)
    return c, candidates, fingerprint(questions)


def failed_first_decision(c, candidates, data_hash):
    decision={'passed':False,'selected':None,'candidates':copy.deepcopy(candidates),
              'data_hash':data_hash,'generation_budget':256,'protocol_id':c['selection_protocol_id'],
              'input_bindings':bindings.selection_inputs(c,'lr-1e-4')}
    write_json(c['report']/'first-run-decision.json',decision)
    return decision


class SelectionBindingsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): SCRATCH.mkdir(parents=True,exist_ok=True)

    def temporary(self): return tempfile.TemporaryDirectory(dir=SCRATCH)

    def test_complete_four_checkpoint_inputs_are_bound(self):
        with self.temporary() as folder:
            c,candidates,data_hash=build_fixture(folder)
            decision=failed_first_decision(c,candidates,data_hash)
            bindings.validate_selection_inputs(c,decision,'lr-1e-4')
            files=decision['input_bindings']['files']
            self.assertEqual([r['step'] for r in candidates],[63,125,188,250])
            for step in (63,125,188,250):
                self.assertIn(f'checkpoint_{step}_config',files)
                self.assertIn(f'checkpoint_{step}_weights',files)

    def test_baseline_or_unselected_checkpoint_changes_invalidate_decision(self):
        for target in ('baseline_raw','baseline_review','checkpoint_125_raw','checkpoint_250_review',
                       'checkpoint_188_weights','checkpoint_125_config'):
            with self.subTest(target=target),self.temporary() as folder:
                c,candidates,data_hash=build_fixture(folder)
                decision=failed_first_decision(c,candidates,data_hash)
                path=Path(decision['input_bindings']['files'][target]['path'])
                path.write_bytes(path.read_bytes()+b'\n')
                with self.assertRaises((ValueError,json.JSONDecodeError)):
                    bindings.validate_selection_inputs(c,decision,'lr-1e-4')

    def test_current_configuration_change_invalidates_old_training_artifacts(self):
        with self.temporary() as folder:
            c,candidates,data_hash=build_fixture(folder)
            decision=failed_first_decision(c,candidates,data_hash)
            c['seed']+=1
            with self.assertRaises(ValueError): bindings.validate_selection_inputs(c,decision,'lr-1e-4')

    def test_saved_training_configuration_and_checkpoint_identity_must_match(self):
        for target in ('resolved_configuration','lr','revision','checkpoint_identity'):
            with self.subTest(target=target),self.temporary() as folder:
                c,candidates,_=build_fixture(folder)
                path=c['runs']/'lr-1e-4/configuration.json'
                value=json.loads(path.read_text(encoding='utf8'))
                if target=='resolved_configuration':value[target]['seed']+=1
                elif target=='checkpoint_identity':
                    path=c['runs']/'lr-1e-4/checkpoint-125/run-state.json'
                    value=json.loads(path.read_text(encoding='utf8'));value['identity']['revision']='wrong'
                else:value['identity'][target]='wrong'
                write_json(path,value)
                with self.assertRaises(ValueError):bindings.selection_inputs(c,'lr-1e-4')

    def test_rule_source_change_invalidates_existing_decision(self):
        with self.temporary() as folder:
            c,candidates,data_hash=build_fixture(folder)
            code=Path(folder)/'isolated-code';code.mkdir()
            for name in bindings.RULE_FILES:shutil.copy2(Path(bindings.__file__).with_name(name),code/name)
            with patch.object(bindings,'__file__',str(code/'selection_bindings.py')):
                decision=failed_first_decision(c,candidates,data_hash)
                path=code/'semantic_grading.py';path.write_bytes(path.read_bytes()+b'\n# changed scoring protocol\n')
                with self.assertRaises(ValueError):bindings.validate_selection_inputs(c,decision,'lr-1e-4')

    def test_second_rate_requires_complete_unchanged_failed_first_decision(self):
        with self.temporary() as folder:
            c,candidates,data_hash=build_fixture(folder)
            require=bindings.require_first_failure
            with self.assertRaises((ValueError,FileNotFoundError)):require(c)
            valid=failed_first_decision(c,candidates,data_hash)
            require(c)
            for mutation in ('passed','selected','missing_checkpoint','stale_baseline'):
                with self.subTest(mutation=mutation):
                    bad=copy.deepcopy(valid)
                    if mutation=='passed':bad['passed']=True
                    elif mutation=='selected':bad['selected']=copy.deepcopy(candidates[0])
                    elif mutation=='missing_checkpoint':bad['candidates']=bad['candidates'][:-1]
                    else:(Path(c['baseline_report'])/'answers.jsonl').write_text('changed\n',encoding='utf8')
                    write_json(c['report']/'first-run-decision.json',bad)
                    with self.assertRaises(ValueError):require(c)

    def test_second_rate_selection_cannot_precede_first_failure(self):
        with self.temporary() as folder:
            c,_,_=build_fixture(folder,'lr-5e-5')
            with self.assertRaises((ValueError,FileNotFoundError)):
                select(c,'lr-5e-5')
            self.assertFalse((c['report']/'second-run-decision.json').exists())

    def test_second_rate_training_rejects_missing_first_failure_before_cuda(self):
        from scripts.linux_assistant.train import execute
        with self.temporary() as folder:
            c,_,_=build_fixture(folder)
            c['training_review']='independent-approval.json'
            with patch('scripts.linux_assistant.train.verify_data',return_value={}), \
                 patch('scripts.linux_assistant.train.verify_model'), \
                 patch('scripts.linux_assistant.review_training_data.validate_training_review',return_value={}), \
                 patch('scripts.linux_assistant.retrain.preflight',return_value={}), \
                 patch('torch.cuda.reset_peak_memory_stats',side_effect=AssertionError('CUDA must not be initialized')) as cuda:
                with self.assertRaises(FileNotFoundError):
                    execute(c,c['runs']/'lr-5e-5',second=True)
                cuda.assert_not_called()

    def test_select_rejects_adapter_configuration_changed_since_generation(self):
        with self.temporary() as folder:
            c,_,_=build_fixture(folder)
            # First prove all unchanged artifact identities support actual selection.
            result=select(c,'lr-1e-4')
            self.assertTrue(result['passed'])
            path=c['runs']/'lr-1e-4/checkpoint-125/adapter_config.json'
            value=json.loads(path.read_text(encoding='utf8'));value['r']=8;write_json(path,value)
            with self.assertRaisesRegex(ValueError,'checkpoint|config|configuration'):
                select(c,'lr-1e-4')

    def test_baseline_revision_and_context_must_match_current_experiment(self):
        for key,value in (('revision','other-model-revision'),('context_length',4096),('model','other-model')):
            with self.subTest(key=key),self.temporary() as folder:
                c,_,data_hash=build_fixture(folder)
                path=Path(c['baseline_report'])/'identity.json'
                identity=json.loads(path.read_text(encoding='utf8'));identity[key]=value;write_json(path,identity)
                with self.assertRaises(ValueError):_identity(path,data_hash,256,'system',c=c,baseline=True)


if __name__=='__main__':unittest.main()
