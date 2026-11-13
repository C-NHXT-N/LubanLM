import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.common import WORKSPACE,sha256
from scripts.linux_assistant.freeze_release import checked_workspace_path,verify_pilot_qualification,freeze,SPLITS,LEDGERS


class FreezeReleaseTests(unittest.TestCase):
    def test_direct_training_requires_same_baseline_gate_before_cuda(self):
        from scripts.linux_assistant.train import execute
        c={'training':{},'model':Path('base'),'revision':'fixed',
           'training_review':'review.json','experiment_scope':'controlled_1000_skill_rebuild'}
        with tempfile.TemporaryDirectory(dir=WORKSPACE/'.cache') as directory:
            with patch('scripts.linux_assistant.train.verify_data',return_value={}),patch('scripts.linux_assistant.train.verify_model'),patch('scripts.linux_assistant.review_training_data.validate_training_review',return_value={}),patch('scripts.linux_assistant.retrain.preflight',side_effect=ValueError('baseline review missing')) as preflight,patch('scripts.linux_assistant.train.seed_everything') as seed,patch('scripts.linux_assistant.train.make_model') as model,self.assertRaisesRegex(ValueError,'baseline review missing'):
                execute(c,Path(directory)/'new-run',smoke=True)
            preflight.assert_called_once_with(c)
            seed.assert_not_called()
            model.assert_not_called()

    def test_cannot_write_outside_workspace(self):
        with self.assertRaisesRegex(ValueError,'below the workspace'):
            checked_workspace_path(WORKSPACE.parent/'outside-release')

    def test_passing_flag_without_evidence_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=WORKSPACE/'.cache') as directory:
            proof=Path(directory)/'proof.json'
            proof.write_text(json.dumps({'pilot_method_qualified':True,'expansion_allowed':True}),'utf8')
            with self.assertRaisesRegex(ValueError,'evidence is missing or stale'):
                verify_pilot_qualification(proof)

    def test_changed_qualification_evidence_invalidates_release(self):
        with tempfile.TemporaryDirectory(dir=WORKSPACE/'.cache') as directory:
            root=Path(directory);paths={};digests={}
            for name in ('data','primary','source','cross','engineering'):
                path=root/(name+'.json');path.write_text('[]','utf8')
                paths[name]=str(path);digests[name]=sha256(path)
            proof=root/'proof.json'
            proof.write_text(json.dumps({'pilot_method_qualified':True,'expansion_allowed':True,
                                        'input_paths':paths,'input_sha256':digests}),'utf8')
            (root/'source.json').write_text('[{}]','utf8')
            with self.assertRaisesRegex(ValueError,'stale: source'):
                verify_pilot_qualification(proof)

    def test_missing_independent_ledger_cannot_create_release(self):
        with tempfile.TemporaryDirectory(dir=WORKSPACE/'.cache') as directory:
            root=Path(directory);source=root/'candidate';source.mkdir()
            for name in SPLITS:(source/name).write_text('','utf8')
            with patch('scripts.linux_assistant.freeze_release.verify_pilot_qualification',return_value={}),self.assertRaisesRegex(ValueError,'content-review'):
                freeze({'experiment_scope':'controlled_1000_skill_rebuild'},source,root/'release',root/'report',root/'proof.json')
            self.assertFalse((root/'release').exists())

    def test_failed_content_validation_leaves_only_blocked_copy(self):
        with tempfile.TemporaryDirectory(dir=WORKSPACE/'.cache') as directory:
            root=Path(directory);source=root/'candidate';source.mkdir()
            for name in SPLITS+LEDGERS:(source/name).write_text('','utf8')
            (root/'proof.json').write_text('{}','utf8')
            with patch('scripts.linux_assistant.freeze_release.verify_pilot_qualification',return_value={}),patch('scripts.linux_assistant.freeze_release.verify_data',side_effect=ValueError('unreviewed record')),self.assertRaisesRegex(ValueError,'unreviewed record'):
                freeze({'experiment_scope':'controlled_1000_skill_rebuild'},source,root/'release',root/'report',root/'proof.json')
            manifest=json.loads((root/'release/manifest.json').read_text('utf8'))
            self.assertEqual(manifest['release_status'],'failed')
            self.assertFalse(manifest['training_approval'])
            self.assertFalse((root/'report/training-approval.json').exists())
            self.assertFalse(json.loads((root/'report/freeze-result.json').read_text('utf8'))['training_started'])

    def test_missing_or_changed_scope_cannot_skip_crossreview(self):
        with tempfile.TemporaryDirectory(dir=WORKSPACE/'.cache') as directory:
            root=Path(directory)
            for scope in (None,'anything_else'):
                with self.subTest(scope=scope),self.assertRaisesRegex(ValueError,'only controlled'):
                    freeze({'experiment_scope':scope},root/'source',root/'release',root/'report',root/'proof.json')
            self.assertFalse((root/'release').exists())

    def test_overlapping_destinations_do_not_pollute_input(self):
        with tempfile.TemporaryDirectory(dir=WORKSPACE/'.cache') as directory:
            root=Path(directory);source=root/'candidate';source.mkdir()
            for output,report in ((source/'new',root/'report'),(root/'same',root/'same'),(root/'new',source/'report')):
                with self.subTest(output=output,report=report),self.assertRaisesRegex(ValueError,'must not overlap'):
                    freeze({'experiment_scope':'controlled_1000_skill_rebuild'},source,output,report,root/'proof.json')
            self.assertEqual(list(source.iterdir()),[])


if __name__=='__main__':unittest.main()
