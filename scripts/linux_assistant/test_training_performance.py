"""CPU regressions for training-loop performance options and batch semantics."""
import unittest

from scripts.linux_assistant.train import (
    bucket_update_chunks,
    gradients_are_finite,
    optimizer_options,
    smoke_indices,
    micro_batches,
    checkpoint_policy,
)
from scripts.linux_assistant.benchmark_training_pipeline import (
    allocator_fraction_for_budget,
    memory_headroom,
)


class TrainingPerformanceTests(unittest.TestCase):
    def test_controlled_smoke_uses_resume_steps_instead_of_formal_steps(self):
        training={'checkpoint_steps':[63,125,188,250],
                  'full_development_at_each_checkpoint':True}
        self.assertEqual(checkpoint_policy(training,4,smoke=True),({2,4},False))
        self.assertEqual(checkpoint_policy(training,250),({63,125,188,250},True))
        with self.assertRaisesRegex(ValueError,'four updates'):
            checkpoint_policy(training,8,smoke=True)

    def test_explicit_checkpoint_policy_requires_full_final_step(self):
        steps,full=checkpoint_policy({'checkpoint_steps':[63,125,188,250],
                                      'full_development_at_each_checkpoint':True},250)
        self.assertEqual(steps,{63,125,188,250})
        self.assertTrue(full)
        for invalid in ([63,125,188], [125,63,250], [63,63,250], [0,250], [251]):
            with self.assertRaisesRegex(ValueError,'checkpoint_steps'):
                checkpoint_policy({'checkpoint_steps':invalid},250)
        self.assertEqual(checkpoint_policy({'save_steps':16},61),({16,32,48,61},False))

    def test_token_budget_batches_preserve_examples_and_update_membership(self):
        data=[{'input_ids':[0]*n} for n in (100,200,700,1024,90)]
        groups=list(micro_batches([0,1,2,3,4],data,4,1600))
        self.assertEqual([i for group in groups for i in group],[0,1,2,3,4])
        for group in groups:
            self.assertLessEqual(len(group),4)
            self.assertLessEqual(max(len(data[i]['input_ids']) for i in group)*len(group),1600)
        self.assertEqual(list(micro_batches([0,1,2,3,4],data,4)),[[0,1,2,3],[4]])
        with self.assertRaisesRegex(ValueError,'do not truncate'):
            list(micro_batches([3],data,4,512))

    def test_optimizer_modes_are_explicit_and_default_preserves_adamw(self):
        self.assertEqual(optimizer_options({}), ('adamw', {}))
        self.assertEqual(optimizer_options({'optimizer_mode': 'adamw'}), ('adamw', {}))
        self.assertEqual(optimizer_options({'optimizer_mode': 'adamw_fused'}),
                         ('adamw_fused', {'fused': True}))
        with self.assertRaisesRegex(ValueError, 'optimizer_mode'):
            optimizer_options({'optimizer_mode': 'unknown'})

    def test_gradient_check_modes_have_identical_finite_semantics(self):
        import torch

        params = [torch.nn.Parameter(torch.tensor([1.0, 2.0])),
                  torch.nn.Parameter(torch.tensor([-3.0]))]
        for p in params:
            p.grad = torch.ones_like(p)
        for mode in ('aggregated', 'per_parameter'):
            self.assertTrue(gradients_are_finite(params, mode, torch))
        params[1].grad[0] = float('inf')
        for mode in ('aggregated', 'per_parameter'):
            self.assertFalse(gradients_are_finite(params, mode, torch))
        self.assertFalse(gradients_are_finite([], 'aggregated', torch))
        with self.assertRaisesRegex(ValueError, 'gradient_check_mode'):
            gradients_are_finite(params, 'unknown', torch)

    def test_length_bucketing_only_reorders_within_existing_updates(self):
        data = [
            {'input_ids': [0] * 7},
            {'input_ids': [0] * 2},
            {'input_ids': [0] * 5},
            {'input_ids': [0] * 1},
            {'input_ids': [0] * 3},
        ]
        chunks = [[0, 1, 2], [3, 4]]
        self.assertEqual(bucket_update_chunks(chunks, data, False), chunks)
        bucketed = bucket_update_chunks(chunks, data, True)
        self.assertEqual(bucketed, [[1, 2, 0], [3, 4]])
        self.assertEqual([set(chunk) for chunk in bucketed], [set(chunk) for chunk in chunks])
        with self.assertRaisesRegex(ValueError, 'boolean'):
            bucket_update_chunks(chunks, data, 'true')

    def test_benchmark_memory_budget_fails_closed_and_respects_reserve(self):
        gib = 2**30
        fraction = allocator_fraction_for_budget(14 * gib, 16 * gib, 2 * 1024, 0.9)
        self.assertAlmostEqual(fraction, 0.75)
        with self.assertRaisesRegex(RuntimeError, 'below the configured reserve'):
            allocator_fraction_for_budget(1 * gib, 16 * gib, 2 * 1024, 0.9)
        enough = memory_headroom(3072, 12000, 16384, 2048)
        self.assertTrue(enough['passed'])
        tight = memory_headroom(1024, 15000, 16384, 2048)
        self.assertFalse(tight['passed'])

    def test_smoke_sample_always_contains_the_actual_longest_training_record(self):
        data = [{'input_ids': [0] * length} for length in (2, 7, 3, 9, 4)]
        chosen = smoke_indices(data, seed=7, count=3)
        first_pass = chosen[:3]
        self.assertEqual(len(set(first_pass)), 3)
        self.assertIn(3, first_pass)
        self.assertEqual(chosen, first_pass * 2)
        with self.assertRaisesRegex(ValueError, 'distinct training records'):
            smoke_indices(data, seed=7, count=6)


if __name__ == '__main__':
    unittest.main()
