import json
import tempfile
import unittest
from pathlib import Path

from scripts.linux_assistant.checkpoint_calibration import (
    row_generation_status,
    summarize_outputs,
    validate_raw_against_prepared,
    write_review_queue,
    scoring_protocol,
)


class CheckpointCalibrationTests(unittest.TestCase):
    def test_generation_status_is_budget_specific_and_not_semantic(self):
        eos = [10, 11]
        assistant_end = 10
        self.assertEqual(row_generation_status({"generated_ids": [3, 10], "stop_reason": "eos"}, eos, assistant_end, 256), "complete_assistant_end")
        self.assertEqual(row_generation_status({"generated_ids": [3, 11], "stop_reason": "eos"}, eos, assistant_end, 1024), "complete_text_end")
        self.assertEqual(row_generation_status({"generated_ids": [3] * 256, "stop_reason": "length_limit"}, eos, assistant_end, 256), "length_limit")
        self.assertEqual(row_generation_status({"generated_ids": [], "stop_reason": "eos"}, eos, assistant_end, 256), "empty_generation")
        self.assertEqual(row_generation_status({"generated_ids": [3], "stop_reason": "other"}, eos, assistant_end, 256), "unexpected_stop")

    def test_review_queue_maps_legacy_axis_and_does_not_fail_length_limit(self):
        question = {
            "id": "q1", "group": "g1", "kind": "task", "category": "Linux操作",
            "dataset_source": "NL2Bash", "messages": [{"role": "user", "content": "给出命令"}],
            "reference": "ls", "grading": {"format": {"kind": "free"}},
        }
        raw = {"id": "q1", "answer": "ls" * 200, "generated_ids": [3] * 256,
               "stop_reason": "length_limit", "messages": question["messages"]}
        legacy = {"id": "q1", "axes": {"content_correct": True, "steps_complete": True,
                                          "format_compliant": True}, "accepted": True,
                   "review_reason": "legacy", "reviewer": "prior"}
        scratch_root = Path(__file__).resolve().parents[3] / ".cache" / "checkpoint-calibration-unit-tests"
        scratch_root.mkdir(parents=True, exist_ok=True)
        c = {"report": scratch_root / "review-queue", "calibration": {
            "rk3576_sources": [], "document_command_sources": ["NL2Bash"],
            "diagnostic_max_new_tokens": 1024,
        }}
        path, queue = write_review_queue(c, [question], {"step-16": [raw]}, {"step-16": [legacy]}, [10, 11], 10, 256, "q.jsonl")
        self.assertTrue(path.exists())
        self.assertEqual(queue[0]["axes"]["completeness"], True)
        self.assertEqual(queue[0]["axes"]["generation_status"], "length_limit")
        self.assertFalse(queue[0]["product_complete"])
        self.assertTrue(queue[0]["accepted"])
        self.assertEqual(queue[0]["source_family"], "document_command")

    def test_raw_binding_rejects_prompt_or_order_change(self):
        prepared = [{"row": {"id": "q1", "reference": "ok", "grading": {}, "messages": []},
                     "prompt": "prompt", "input_ids": [1]}]
        identity = {"bound": "same"}
        scratch_root = Path(__file__).resolve().parents[3] / ".cache" / "checkpoint-calibration-unit-tests"
        scratch_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch_root) as tmp:
            p = Path(tmp) / "answers.jsonl"
            ident = p.with_suffix(".identity.json")
            p.write_text(json.dumps({"id": "q1", "prompt": "wrong", "input_ids": [1],
                                     "reference": "ok", "grading": {}, "messages": []}) + "\n", encoding="utf-8")
            ident.write_text(json.dumps(identity), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Raw output does not match"):
                validate_raw_against_prepared(p, ident, identity, prepared)

    def test_protocol_acceptance_excludes_generation_status(self):
        protocol = scoring_protocol({"max_new_tokens": 256, "calibration": {"diagnostic_max_new_tokens": 1024}, "context_length": 2048})
        self.assertIn("generation_status", protocol["axes"])
        self.assertIn("does not make the answer wrong", protocol["axes"]["generation_status"])
        self.assertIn("only when content_correct, completeness, and format_compliant are all true", protocol["accepted_definition"])
        self.assertIn("all 36 rows", protocol["budgets"]["wide_scope"])

    def test_summary_reports_stop_length_and_groups(self):
        row = {"model": "step-16", "generation_budget": 256, "generation_status": "length_limit",
               "generated_tokens_including_stop": 256, "product_complete": False,
               "accepted": True, "axes": {"content_correct": True, "completeness": True,
                                             "format_compliant": True},
               "source_family": "document_command", "category": "Linux操作"}
        summary = summarize_outputs([row])["step-16/256"]
        self.assertEqual(summary["generation_status_counts"], {"length_limit": 1})
        self.assertEqual(summary["by_source_family"]["document_command"]["accepted"], 1)
        self.assertEqual(summary["by_category"]["Linux操作"]["reviewed"], 1)


if __name__ == "__main__":
    unittest.main()
