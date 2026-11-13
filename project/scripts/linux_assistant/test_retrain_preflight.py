"""CPU tests for no-GPU retrain preflight and all training-split step accounting."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import ANY, patch

from scripts.common import WORKSPACE
from scripts.linux_assistant import retrain


class RetrainPreflightTests(unittest.TestCase):
    def test_invalid_batch_config_stops_before_data_review(self):
        with patch.object(retrain,'validate_training_review') as review:
            with self.assertRaisesRegex(ValueError,'starting at 1'):
                retrain.preflight({'baseline':{'batch_candidates':[4,2,1]}})
        review.assert_not_called()

    def test_failed_preflight_writes_attempt_without_changing_previous_results(self):
        with tempfile.TemporaryDirectory(dir=WORKSPACE/'.cache') as tmp:
            root=Path(tmp); c,config_path=self._fixture(root)
            c['report'].mkdir()
            previous=c['report']/'status.json'
            previous.write_text('{"status":"historical"}',encoding='utf8')
            with patch.object(retrain,'preflight',side_effect=ValueError('unreviewed data')):
                with self.assertRaisesRegex(ValueError,'unreviewed data'):
                    retrain.record_preflight(c,config_path)
            self.assertEqual(previous.read_text('utf8'),'{"status":"historical"}')
            attempts=list((c['report']/'preflight-attempts').glob('*.json'))
            self.assertEqual(len(attempts),1)
            decision=json.loads(attempts[0].read_text('utf8'))
            self.assertFalse(decision['passed'])
            self.assertFalse(decision['model_loaded'])
            self.assertFalse(decision['training_started'])

    def _fixture(self, root):
        cleanup = root / 'cleanup.json'
        cleanup.write_text(json.dumps({'completed': True, 'deleted_targets': []}), encoding='utf8')
        config_path = root / 'config.json'
        config_path.write_text('{}', encoding='utf8')
        c = {
            'report': root / 'report',
            'cleanup_record': 'cleanup.json',
            'baseline_report': 'reports/baseline',
            'model': root / 'model',
            'data': root / 'data',
            'max_length': 1024,
            'training': {'epochs': 2, 'gradient_accumulation': 4},
        }
        return c, config_path

    def test_preflight_reuses_all_semantic_release_gates_without_model_load(self):
        with tempfile.TemporaryDirectory(dir=WORKSPACE / '.cache') as tmp:
            root = Path(tmp)
            c, config_path = self._fixture(root)
            with patch.object(retrain, 'WORKSPACE', root), \
                 patch.object(retrain, 'validate_training_review', return_value={'approved_records': 3}) as review, \
                 patch.object(retrain, 'verify_data', return_value={'release_status': 'complete'}) as data_gate, \
                 patch.object(retrain, 'verify_baseline', return_value={'semantic_review': {'scores': {'task': 1}}}) as baseline_gate, \
                 patch('scripts.linux_assistant.modeling.training_data', return_value=(
                     [{'id': 'long-case'}], [{'input_ids': [1, 2, 3], 'labels': [-100, 2, 3]}]
                 )) as encode_training:
                result = retrain.preflight(c, config_path, tokenizer=object())
            self.assertTrue(result['passed'])
            self.assertEqual(result['data_review']['approved_records'], 3)
            self.assertEqual(result['baseline']['semantic_review']['scores']['task'], 1)
            self.assertEqual(result['training_encoding']['max_sequence_length'], 3)
            self.assertEqual(result['training_encoding']['max_sequence_record_id'], 'long-case')
            self.assertFalse(result['training_encoding']['gpu_fit_proven'])
            review.assert_called_once_with(c)
            data_gate.assert_called_once_with(c)
            baseline_gate.assert_called_once_with(c, ANY)
            encode_training.assert_called_once_with(c, unittest.mock.ANY)

    def test_preflight_stops_at_partial_data_gate_before_baseline(self):
        with tempfile.TemporaryDirectory(dir=WORKSPACE / '.cache') as tmp:
            root = Path(tmp)
            c, config_path = self._fixture(root)
            with patch.object(retrain, 'WORKSPACE', root), \
                 patch.object(retrain, 'validate_training_review', return_value={'approved_records': 3}), \
                 patch.object(retrain, 'verify_data', side_effect=ValueError('partial dataset blocked')), \
                 patch.object(retrain, 'verify_baseline') as baseline_gate:
                with self.assertRaisesRegex(ValueError, 'partial dataset blocked'):
                    retrain.preflight(c, config_path, tokenizer=object())
            baseline_gate.assert_not_called()

    def test_comparison_step_count_includes_general_training_split(self):
        with tempfile.TemporaryDirectory(dir=WORKSPACE / '.cache') as tmp:
            data = Path(tmp)
            (data / 'manifest.json').write_text(json.dumps({
                'data_splits': ['train.jsonl', 'general-train.jsonl', 'dev.jsonl', 'test.jsonl']
            }), encoding='utf8')
            (data / 'train.jsonl').write_text('{"id":"t1"}\n{"id":"t2"}\n{"id":"t3"}\n', encoding='utf8')
            (data / 'general-train.jsonl').write_text('{"id":"g1"}\n{"id":"g2"}\n', encoding='utf8')
            (data / 'dev.jsonl').write_text('{"id":"d1"}\n', encoding='utf8')
            (data / 'test.jsonl').write_text('{"id":"sealed"}\n', encoding='utf8')
            from scripts.linux_assistant.common import training_rows
            c = {'data': data, 'training': {'epochs': 2, 'gradient_accumulation': 4}}
            self.assertEqual(len(training_rows(c)), 5)
            self.assertEqual(retrain.expected_update_count(c), 3)
            self.assertNotIn('sealed', {row['id'] for row in training_rows(c)})


if __name__ == '__main__':
    unittest.main()
