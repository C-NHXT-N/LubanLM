"""Regression guards for review approval, frozen evidence and fair baseline reuse."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts.common import WORKSPACE,write_json,sha256
from scripts.linux_assistant.common import write_jsonl,jsonl,fingerprint,verify_data
from scripts.linux_assistant import review_training_data as review

class ReviewGateTests(unittest.TestCase):
    def fixture(self,root):
        data=root/'data';report=root/'review';data.mkdir();report.mkdir()
        row={'id':'x','group':'linux/example','created_by':'builder-agent','messages':[{'role':'user','content':'question'},
            {'role':'assistant','content':'answer'}],'reference':'answer'}
        write_jsonl(data/'train.jsonl',[row]);write_jsonl(data/'source-review.jsonl',[{
            'id':'x','decision':'accept','reviewer_id':'independent-source-reviewer-agent',
            'reason':'Pinned fixture source and license were checked.','final_row_sha256':fingerprint(row),
            'source_verified':True,'source_hash_verified':True,'license_reviewed':True,'license':'fixture-test-only',
            'safety_reviewed':True,'safety_decision':'clear'}])
        write_json(data/'source-review-lock.json',{'review_schema':'lubanlm.source-review.v2',
            'status':'complete','review_count':1,'sha256':sha256(data/'source-review.jsonl')})
        write_jsonl(data/'content-review.jsonl',[{'id':'x','decision':'accept',
            'reviewer_id':'independent-reviewer-agent','reason':'Reviewed actual prompt and reference.',
            'final_row_sha256':fingerprint(row)}])
        manifest={'dataset_schema':'lubanlm.linux-assistant-dataset.v2','release_status':'complete','content_review_status':'complete',
            'content_review_file':'content-review.jsonl','content_review_sha256':sha256(data/'content-review.jsonl'),
            'data_splits':['train.jsonl'],'files':{'train.jsonl':sha256(data/'train.jsonl'),
            'source-review.jsonl':sha256(data/'source-review.jsonl'),
            'source-review-lock.json':sha256(data/'source-review-lock.json'),
            'content-review.jsonl':sha256(data/'content-review.jsonl')}}
        write_json(data/'manifest.json',manifest)
        write_jsonl(report/'approved-training-records.jsonl',[{'id':'x','decision':'accept',
            'reviewer_id':'independent-reviewer-agent','record_sha256':fingerprint(row),
            'reason':'Independently checked actual question, context, reference and constraints.'}])
        write_json(report/'evidence.json',{'passed':True})
        approval={'approval_schema':'lubanlm.dataset-content-review.v2',
            'review_mode':'independent_agent_cross_review','passed':True,
            'reviewer':'independent reviewer agent fixture','train_sha256':sha256(data/'train.jsonl'),
            'manifest_sha256':sha256(data/'manifest.json'),'source_review_sha256':sha256(data/'source-review.jsonl'),
            'ledger_sha256':sha256(report/'approved-training-records.jsonl'),
            'evidence':{'review/evidence.json':sha256(report/'evidence.json')}}
        write_json(report/'training-approval.json',approval)
        return {'data':data,'training_review':'review/training-approval.json'},row

    def test_review_gate_covers_actual_records(self):
        with tempfile.TemporaryDirectory(dir=WORKSPACE/'.cache') as tmp:
            root=Path(tmp);c,_=self.fixture(root)
            with patch.object(review,'WORKSPACE',root):
                self.assertEqual(review.validate_training_review(c)['approved_records'],1)

    def test_small_experiment_cannot_reuse_unscoped_approval(self):
        with tempfile.TemporaryDirectory(dir=WORKSPACE/'.cache') as tmp:
            root=Path(tmp);c,_=self.fixture(root)
            c['experiment_scope']='reviewed_small_scale_exploration'
            with patch.object(review,'WORKSPACE',root),self.assertRaisesRegex(ValueError,'approval scope'):
                review.validate_training_review(c)

    def test_training_reads_general_examples_and_excludes_all_holdouts(self):
        from scripts.linux_assistant.common import training_rows
        with tempfile.TemporaryDirectory(dir=WORKSPACE/'.cache') as tmp:
            root=Path(tmp); c,_=self.fixture(root)
            write_jsonl(c['data']/'general-train.jsonl',[{'id':'general-train'}])
            write_jsonl(c['data']/'general-dev.jsonl',[{'id':'sealed-out'}])
            path=c['data']/'manifest.json'; manifest=json.loads(path.read_text('utf8'))
            manifest['data_splits'] += ['general-train.jsonl','general-dev.jsonl']
            write_json(path,manifest)
            self.assertEqual([r['id'] for r in training_rows(c)],['x','general-train'])

    def test_task_only_approval_cannot_silently_cover_general_training(self):
        with tempfile.TemporaryDirectory(dir=WORKSPACE/'.cache') as tmp:
            root=Path(tmp);c,_=self.fixture(root)
            write_jsonl(c['data']/'general-train.jsonl',[{'id':'unreviewed-general'}])
            path=c['data']/'manifest.json';manifest=json.loads(path.read_text('utf8'))
            manifest['data_splits'].append('general-train.jsonl')
            write_json(path,manifest)
            approval_path=root/'review/training-approval.json';a=json.loads(approval_path.read_text('utf8'))
            a['manifest_sha256']=sha256(path);write_json(approval_path,a)
            with patch.object(review,'WORKSPACE',root),self.assertRaisesRegex(ValueError,'missing general training approval'):
                review.validate_training_review(c)

    def test_changed_prompt_rejected_even_when_ids_and_reference_match(self):
        with tempfile.TemporaryDirectory(dir=WORKSPACE/'.cache') as tmp:
            root=Path(tmp);c,row=self.fixture(root)
            row['messages'][0]['content']='different constraint';write_jsonl(c['data']/'train.jsonl',[row])
            with patch.object(review,'WORKSPACE',root),self.assertRaisesRegex(ValueError,'Approved data changed'):
                review.validate_training_review(c)

    def test_changed_review_evidence_rejected(self):
        with tempfile.TemporaryDirectory(dir=WORKSPACE/'.cache') as tmp:
            root=Path(tmp);c,_=self.fixture(root);write_json(root/'review/evidence.json',{'passed':False})
            with patch.object(review,'WORKSPACE',root),self.assertRaisesRegex(ValueError,'Review evidence changed'):
                review.validate_training_review(c)

    def test_approved_system_prompt_is_bound_and_required(self):
        with tempfile.TemporaryDirectory(dir=WORKSPACE/'.cache') as tmp:
            root=Path(tmp);c,_=self.fixture(root);c['system_prompt']='同一系统提示'
            path=root/'review/training-approval.json';a=json.loads(path.read_text('utf8'))
            a['system_prompt_sha256']=hashlib.sha256(c['system_prompt'].encode()).hexdigest();write_json(path,a)
            with patch.object(review,'WORKSPACE',root):
                self.assertEqual(review.validate_training_review(c)['approved_records'],1)
                for prompt in ('','另一系统提示'):
                    with self.assertRaisesRegex(ValueError,'System prompt differs'):
                        review.validate_training_review({**c,'system_prompt':prompt})

    def test_missing_decision_rejected_even_after_resigning_outer_hash(self):
        with tempfile.TemporaryDirectory(dir=WORKSPACE/'.cache') as tmp:
            root=Path(tmp);c,_=self.fixture(root);ledger=root/'review/approved-training-records.jsonl'
            write_jsonl(ledger,[]);path=root/'review/training-approval.json';a=json.loads(path.read_text('utf8'))
            a['ledger_sha256']=sha256(ledger);write_json(path,a)
            with patch.object(review,'WORKSPACE',root),self.assertRaisesRegex(ValueError,'Incomplete per-record'):
                review.validate_training_review(c)

    def test_legacy_dataset_manifest_is_rejected_even_when_rehashed(self):
        with tempfile.TemporaryDirectory(dir=WORKSPACE/'.cache') as tmp:
            root=Path(tmp);c,_=self.fixture(root)
            manifest_path=c['data']/'manifest.json';manifest=json.loads(manifest_path.read_text('utf8'))
            manifest['dataset_schema']='legacy-untyped';write_json(manifest_path,manifest)
            approval_path=root/'review/training-approval.json';approval=json.loads(approval_path.read_text('utf8'))
            approval['manifest_sha256']=sha256(manifest_path);write_json(approval_path,approval)
            with patch.object(review,'WORKSPACE',root),self.assertRaisesRegex(ValueError,'Legacy or untyped'):
                review.validate_training_review(c)

    def test_source_ledger_must_cover_current_rows_and_pin_hashes(self):
        with tempfile.TemporaryDirectory(dir=WORKSPACE/'.cache') as tmp:
            root=Path(tmp);c,_=self.fixture(root);data=c['data']
            write_jsonl(data/'source-review.jsonl',[])
            lock=data/'source-review-lock.json';value=json.loads(lock.read_text('utf8'))
            value.update(status='complete',review_count=0,sha256=sha256(data/'source-review.jsonl'));write_json(lock,value)
            manifest_path=data/'manifest.json';manifest=json.loads(manifest_path.read_text('utf8'))
            manifest['files']['source-review.jsonl']=sha256(data/'source-review.jsonl')
            manifest['files']['source-review-lock.json']=sha256(lock);write_json(manifest_path,manifest)
            with self.assertRaisesRegex(ValueError,'source-review ledger does not exactly cover'):
                verify_data({'data':data})

    def test_source_review_fingerprint_change_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=WORKSPACE/'.cache') as tmp:
            root=Path(tmp);c,_=self.fixture(root);data=c['data']
            ledger=jsonl(data/'source-review.jsonl');ledger[0]['final_row_sha256']='0'*64
            write_jsonl(data/'source-review.jsonl',ledger)
            lock=data/'source-review-lock.json';value=json.loads(lock.read_text('utf8'))
            value['sha256']=sha256(data/'source-review.jsonl');write_json(lock,value)
            manifest_path=data/'manifest.json';manifest=json.loads(manifest_path.read_text('utf8'))
            manifest['files']['source-review.jsonl']=sha256(data/'source-review.jsonl')
            manifest['files']['source-review-lock.json']=sha256(lock);write_json(manifest_path,manifest)
            with self.assertRaisesRegex(ValueError,'source/license/safety review missing or stale'):
                verify_data({'data':data})

    def test_partial_release_is_blocked_from_baseline_and_training(self):
        with tempfile.TemporaryDirectory(dir=WORKSPACE/'.cache') as tmp:
            root=Path(tmp);c,_=self.fixture(root);data=c['data']
            manifest_path=data/'manifest.json';manifest=json.loads(manifest_path.read_text('utf8'))
            manifest['release_status']='partial';write_json(manifest_path,manifest)
            with self.assertRaisesRegex(ValueError,'Partial Linux-assistant dataset release'):
                verify_data({'data':data})

    def test_explicit_small_experiment_still_requires_every_row_review(self):
        with tempfile.TemporaryDirectory(dir=WORKSPACE/'.cache') as tmp:
            root=Path(tmp);c,_=self.fixture(root);data=c['data']
            path=data/'manifest.json';manifest=json.loads(path.read_text('utf8'))
            manifest.update(release_status='partial',training_approval=True,exploratory_training={
                'scope':'reviewed_small_scale_exploration','user_choice':2,
                'coverage_incomplete':True,'pending_records_excluded':True})
            write_json(path,manifest)
            scoped={'data':data,'experiment_scope':'reviewed_small_scale_exploration'}
            self.assertEqual(verify_data(scoped)['release_status'],'partial')
            with self.assertRaisesRegex(ValueError,'Partial Linux-assistant'):
                verify_data({'data':data})
            ledger=jsonl(data/'content-review.jsonl');ledger[0]['decision']='needs_review'
            write_jsonl(data/'content-review.jsonl',ledger)
            manifest['files']['content-review.jsonl']=sha256(data/'content-review.jsonl')
            manifest['content_review_sha256']=sha256(data/'content-review.jsonl');write_json(path,manifest)
            with self.assertRaisesRegex(ValueError,'Independent content review missing'):
                verify_data(scoped)

    def test_bounded_side_effect_requires_explicit_scope_in_source_review(self):
        with tempfile.TemporaryDirectory(dir=WORKSPACE/'.cache') as tmp:
            root=Path(tmp);c,_=self.fixture(root);data=c['data']
            ledger=jsonl(data/'source-review.jsonl');ledger[0]['safety_decision']='bounded'
            write_jsonl(data/'source-review.jsonl',ledger)
            lock=data/'source-review-lock.json';value=json.loads(lock.read_text('utf8'))
            value['sha256']=sha256(data/'source-review.jsonl');write_json(lock,value)
            manifest_path=data/'manifest.json';manifest=json.loads(manifest_path.read_text('utf8'))
            manifest['files']['source-review.jsonl']=sha256(data/'source-review.jsonl')
            manifest['files']['source-review-lock.json']=sha256(lock);write_json(manifest_path,manifest)
            with self.assertRaisesRegex(ValueError,'source/license/safety review missing or stale'):
                verify_data({'data':data})

    def test_python_content_fixtures_preserve_unicode_and_json_types(self):
        import codecs
        decoder=codecs.getincrementaldecoder('utf-8')(errors='strict')
        self.assertEqual(decoder.decode(bytes.fromhex('e4b8'),final=False),'')
        self.assertEqual(decoder.decode(bytes.fromhex('ad'),final=True),'中')
        with self.assertRaises(UnicodeDecodeError):bytes.fromhex('e4b8').decode('utf8')
        with self.assertRaises(json.JSONDecodeError):json.loads('{"a":1}\n{"b":2}')
        value=json.loads('{"runtime":null,"verified":false}')
        self.assertIsNone(value['runtime']);self.assertIs(value['verified'],False)

    def test_evaluation_does_not_send_final_reference_as_input(self):
        from scripts.linux_assistant.modeling import evaluation_messages
        row={'id':'x','messages':[{'role':'user','content':'question'},
            {'role':'assistant','content':'prior context'}, {'role':'user','content':'follow-up'},
            {'role':'assistant','content':'DO_NOT_SEND_TARGET'}]}
        rendered=evaluation_messages({'system_prompt':'policy'},row)
        self.assertEqual(rendered[-1],{'role':'user','content':'follow-up'})
        self.assertNotIn('DO_NOT_SEND_TARGET',str(rendered))

if __name__=='__main__':unittest.main()
