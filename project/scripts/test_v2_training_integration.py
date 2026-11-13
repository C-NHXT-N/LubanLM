"""Small deterministic checks for the V2-manifest to V1-training adapter."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from scripts.common import ROOT
from scripts.switch_training_performance import checkpoint_is_newer
from training.batches import FullPassTokens
from training.v2_manifest import V2ManifestAdapter


EXPERIMENT = ROOT / "experiments" / "LubanLM-V2-P2-Implementation-20261013"
MANIFEST = EXPERIMENT / "data" / "processed" / "fineweb_4_5_smoke" / "dataset_manifest.json"
TOKENIZER = EXPERIMENT / "tokenizer" / "tokenizer.json"


class V2ManifestIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = V2ManifestAdapter(MANIFEST, tokenizer_path=TOKENIZER)

    def test_train_dev_manifest_and_hash_identity(self):
        train = self.adapter.for_v1_reader("train")
        dev = self.adapter.for_v1_reader("dev")
        self.assertEqual(train["dtype"], "<u2")
        self.assertNotEqual(train["shards"][0]["sha256"], dev["shards"][0]["sha256"])
        self.assertEqual(train["v2_identity"]["split"], "train")
        self.assertEqual(dev["v2_identity"]["split"], "dev")

    def test_full_pass_covers_exact_target_limit_and_resumes_sampler(self):
        sampler = FullPassTokens(
            MANIFEST.parent, length=256, seed=1234, token_limit=513,
            manifest=self.adapter.for_v1_reader("train"),
        )
        first = sampler.batch(1, torch.Generator().manual_seed(1))
        state = sampler.state()
        expected_next = sampler.batch(1, torch.Generator().manual_seed(1))
        sampler.restore(state)
        restored_next = sampler.batch(1, torch.Generator().manual_seed(1))
        for left, right in zip(expected_next[:2], restored_next[:2]):
            self.assertTrue(torch.equal(left, right))
        count = int((first[1] != -100).sum()) + int((restored_next[1] != -100).sum())
        count += sum(int((sampler.batch(min(1, sampler.remaining), torch.Generator().manual_seed(1))[1] != -100).sum()) for _ in range(sampler.remaining))
        self.assertEqual(count, 513)

    def test_frozen_test_split_is_not_exposed_to_training_reader(self):
        with self.assertRaisesRegex(ValueError, "frozen"):
            self.adapter.for_v1_reader("test")

    def test_split_collision_metadata_is_required(self):
        value = json.loads(MANIFEST.read_text(encoding="utf-8"))
        value["split"]["hash_collisions"]["train:dev"] = 1
        with tempfile.TemporaryDirectory(dir=ROOT / "experiments" / "LubanLM-V2-P2-Implementation-20261013" / "tests") as temp:
            path = Path(temp) / "dataset_manifest.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "disjoint"):
                V2ManifestAdapter(path)

    def test_checkpoint_watch_rejects_old_file_in_same_second(self):
        second = 1_800_000_000
        previous_file_ns = second * 1_000_000_000 + 700_000_000
        request_second_ns = second * 1_000_000_000
        baseline_ns = max(previous_file_ns, request_second_ns)
        self.assertFalse(checkpoint_is_newer(previous_file_ns, baseline_ns))
        self.assertTrue(checkpoint_is_newer(previous_file_ns + 100, baseline_ns))


if __name__ == "__main__":
    unittest.main(verbosity=2)
