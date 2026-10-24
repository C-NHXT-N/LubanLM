import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.common import WORKSPACE, sha256
from scripts.linux_assistant.controlled_final import (
    _expected_identity,
    _validate_output_identity,
    _validate_raw_prefix,
    _validate_reviewer_identity,
    _verify_review_unchanged,
    assessment_payload,
    collect_frozen_inputs,
    validate_selection,
)
from scripts.linux_assistant.common import fingerprint


SCRATCH = WORKSPACE / ".cache" / "controlled-final-tests"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")


def metric(values):
    return {"passed": sum(values), "total": len(values), "rate": sum(values) / len(values),
            "wilson95": [0.0, 1.0], "values": values}


class ControlledFinalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        SCRATCH.mkdir(parents=True, exist_ok=True)

    def temporary(self):
        return tempfile.TemporaryDirectory(dir=SCRATCH)

    def test_selection_requires_matching_eligible_run_decision_and_dev_hashes(self):
        with self.temporary() as folder:
            root = Path(folder)
            report, runs, model = root / "report", root / "runs", root / "model"
            adapter = runs / "lr-1e-4" / "checkpoint-16"
            adapter.mkdir(parents=True)
            model.mkdir()
            (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
            write_json(adapter / "adapter_config.json", {"base_model_name_or_path": str(model)})
            run_dir = adapter.parent
            (run_dir / "dev-16.jsonl").write_text("raw\n", encoding="utf-8")
            (run_dir / "dev-16.reviewed.jsonl").write_text("review\n", encoding="utf-8")
            selected = {"step": 16, "adapter": str(adapter), "passed": True,
                        "adapter_sha256": sha256(adapter / "adapter_model.safetensors"),
                        "adapter_config_sha256": sha256(adapter / "adapter_config.json"),
                        "raw_sha256": sha256(run_dir / "dev-16.jsonl"),
                        "review_sha256": sha256(run_dir / "dev-16.reviewed.jsonl")}
            report.mkdir()
            write_json(report / "candidate-selection.json", selected)
            write_json(report / "first-run-decision.json", {
                "passed": True, "data_hash": "dev-question-hash", "generation_budget": 256,
                "protocol_id": "protocol-v1", "selected": selected, "candidates": [selected],
            })
            c = {"report": report, "runs": runs, "model": model, "max_new_tokens": 256,
                 "selection_protocol_id": "protocol-v1"}
            # Full four-checkpoint artifact bindings are exercised separately in
            # test_selection_bindings; this fixture isolates final selection matching.
            with patch("scripts.linux_assistant.selection_bindings.validate_selection_inputs"):
                found, decision_name, decisions = validate_selection(c, "dev-question-hash")
            self.assertEqual(found, selected)
            self.assertEqual(decision_name, "first")
            self.assertIn("first", decisions)
            with self.assertRaisesRegex(ValueError, "different data hash"):
                validate_selection(c, "sealed-test-question-hash")

    def test_collect_frozen_inputs_binds_development_selection_separately_from_sealed_test(self):
        with self.temporary() as folder:
            root = Path(folder)
            data, report, runs, model = root / "data", root / "report", root / "runs", root / "model"
            data.mkdir()
            report.mkdir()
            model.mkdir()
            (model / "download_manifest.json").write_text("{}\n", encoding="utf-8")
            config_file = root / "config.json"
            config_file.write_text("{}\n", encoding="utf-8")
            manifest = {"dataset_schema": "lubanlm.linux-assistant-dataset.v2",
                        "data_splits": ["dev.jsonl", "general-dev.jsonl", "test.jsonl", "general-test.jsonl"]}
            write_json(data / "manifest.json", manifest)

            def make_rows(prefix):
                return [{"id": f"{prefix}-{i}", "messages": [{"role": "user", "content": f"question {prefix} {i}"}],
                         "grading": {"content": {"required_facts": ["answer"]}}}
                        for i in range(250)]

            dev_rows = make_rows("dev")
            test_rows = make_rows("test")
            (data / "dev.jsonl").write_text("".join(json.dumps(row) + "\n" for row in dev_rows[:125]), encoding="utf-8")
            (data / "general-dev.jsonl").write_text("".join(json.dumps(row) + "\n" for row in dev_rows[125:]), encoding="utf-8")
            (data / "test.jsonl").write_text("".join(json.dumps(row) + "\n" for row in test_rows[:125]), encoding="utf-8")
            (data / "general-test.jsonl").write_text("".join(json.dumps(row) + "\n" for row in test_rows[125:]), encoding="utf-8")

            adapter = runs / "lr-1e-4" / "checkpoint-16"
            adapter.mkdir(parents=True)
            (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
            write_json(adapter / "adapter_config.json", {"base_model_name_or_path": str(model)})
            run_dir = adapter.parent
            raw_path = run_dir / "dev-16.jsonl"
            reviewed_path = run_dir / "dev-16.reviewed.jsonl"
            raw_path.write_text("raw output\n", encoding="utf-8")
            reviewed_path.write_text("reviewed output\n", encoding="utf-8")
            selected = {"step": 16, "adapter": str(adapter), "passed": True,
                        "adapter_sha256": sha256(adapter / "adapter_model.safetensors"),
                        "adapter_config_sha256": sha256(adapter / "adapter_config.json"),
                        "raw_sha256": sha256(raw_path), "review_sha256": sha256(reviewed_path)}
            write_json(report / "candidate-selection.json", selected)
            write_json(report / "first-run-decision.json", {
                "passed": True, "data_hash": fingerprint(dev_rows), "generation_budget": 256,
                "protocol_id": "protocol-v1", "selected": selected, "candidates": [selected],
            })
            c = {"data": data, "report": report, "runs": runs, "model": model,
                 "max_new_tokens": 256, "selection_protocol_id": "protocol-v1",
                 "system_prompt": "system", "context_length": 2048, "revision": "revision-1",
                 "dataset_contract": {"evaluation": {"total": 250}}}

            with patch("scripts.linux_assistant.controlled_final.verify_data", return_value=manifest), \
                 patch("scripts.linux_assistant.controlled_final.verify_model", return_value="revision-1"), \
                 patch("scripts.linux_assistant.selection_bindings.validate_selection_inputs"):
                frozen, sealed_rows = collect_frozen_inputs(c, config_file)

            self.assertEqual(len(sealed_rows), 250)
            self.assertEqual(frozen["development_fingerprint_sha256"], fingerprint(dev_rows))
            self.assertEqual(frozen["question_fingerprint_sha256"], fingerprint(test_rows))
            self.assertNotEqual(frozen["development_fingerprint_sha256"], frozen["question_fingerprint_sha256"])
            self.assertEqual(frozen["selected_candidate"]["development_raw_sha256"], sha256(raw_path))

    def test_output_identity_is_bound_to_lock_and_generation_policy(self):
        c = {"model": Path("C:/model/Qwen3.5-2B"), "system_prompt": "system"}
        frozen = {"selected_candidate": {"selection_sha256": "selection-hash",
                                          "adapter_sha256": "adapter-hash",
                                          "adapter_config_sha256": "adapter-config-hash",
                                          "adapter_path": "C:/runs/checkpoint-16"},
                  "base_model": {"revision": "revision"},
                  "dataset_manifest_sha256": "manifest-hash", "selection_protocol_id": "protocol-v1",
                  "question_fingerprint_sha256": "questions-hash",
                  "generation": {"context_length": 2048}}
        expected = _expected_identity(c, frozen, "lock-hash", "adapter")
        identity = {**expected, "data_hash": "questions-hash", "system": "system",
                    "prompt_policy": "context_through_last_user_exclude_target_assistant",
                    "generation": {"greedy": True, "thinking": False, "max_new_tokens": 256,
                                   "stop_policy": "native_assistant_end_or_text_end_v2",
                                   "eos_token_ids": [1, 2]}}
        _validate_output_identity(identity, expected, c)
        identity["lock_sha256"] = "other-lock"
        with self.assertRaisesRegex(ValueError, "lock_sha256"):
            _validate_output_identity(identity, expected, c)

    def test_raw_output_must_be_a_prompt_exact_prefix_of_locked_questions(self):
        question = {"id": "q1", "messages": [{"role": "user", "content": "hi"}],
                    "reference": "hello", "group": "g", "kind": "task", "category": "general",
                    "grading": {"format": {"kind": "free"}}}
        prepared = [{"row": question, "prompt": "prompt", "input_ids": [1, 2],
                     "context": [{"role": "system", "content": "sys"}, *question["messages"]]}]
        raw = {"id": "q1", "messages": question["messages"], "reference": "hello",
               "group": "g", "kind": "task", "category": "general", "grading": question["grading"],
               "prompt": "prompt", "input_ids": [1, 2], "input_messages": question["messages"],
               "answer": "hello", "generated_ids": [3]}
        _validate_raw_prefix([raw], prepared)
        raw["prompt"] = "changed"
        with self.assertRaisesRegex(ValueError, "prompt"):
            _validate_raw_prefix([raw], prepared)

    def test_review_must_preserve_every_raw_answer_and_prompt_field(self):
        question = {"id": "q1", "messages": [{"role": "user", "content": "hi"}],
                    "reference": "hello", "grading": {}, "group": "g", "kind": "task", "category": "general"}
        raw = {"id": "q1", "answer": "hello", "prompt": "p", "input_ids": [1], "generated_ids": [2],
               "messages": question["messages"], "input_messages": question["messages"],
               "reference": "hello", "grading": {}, "group": "g", "kind": "task", "category": "general"}
        _verify_review_unchanged([question], [raw], [dict(raw)])
        changed = dict(raw, answer="changed")
        with self.assertRaisesRegex(ValueError, "answer"):
            _verify_review_unchanged([question], [raw], [changed])

    def test_review_identity_cannot_be_the_evaluated_base_model(self):
        c = {"model": Path("models/Qwen/Qwen3.5-2B")}
        self.assertEqual(_validate_reviewer_identity([{"reviewer": "gpt-6.1-sol"}], "original", c, "rev"),
                         "gpt-6.1-sol")
        with self.assertRaisesRegex(ValueError, "not independent"):
            _validate_reviewer_identity([{"reviewer": "Qwen3.5-2B judge"}], "adapter", c, "rev")

    def test_failed_assessment_stops_and_requires_a_new_sealed_test_set(self):
        with self.temporary() as folder:
            root = Path(folder)
            report = root / "report"
            report.mkdir()
            c = {"report": report, "seed": 20261019,
                 "acceptance": {"task_min": 0.8, "command_min": 0.8, "troubleshooting_min": 0.8,
                                "task_gain": 0.1, "error_reduction": 0.3, "general_max_drop": 0.05}}
            base = {"task": metric([1, 1, 1, 1, 1, 0, 0, 0, 0, 0]),
                    "command_generation": metric([1, 1, 0, 0]),
                    "log_troubleshooting": metric([1, 1, 0, 0]),
                    "general_full": metric([1] * 9 + [0]),
                    "general_final": metric([1] * 9 + [0])}
            candidate = {"task": metric([1] * 7 + [0] * 3),
                         "command_generation": metric([1, 1, 0, 0]),
                         "log_troubleshooting": metric([1, 1, 0, 0]),
                         "general_full": metric([1] * 9 + [0]),
                         "general_final": metric([1] * 9 + [0])}
            for stage in ("original", "adapter"):
                (report / f"controlled-final-{stage}.jsonl").write_text("raw\n", encoding="utf-8")
                (report / f"controlled-final-{stage}.reviewed.jsonl").write_text("review\n", encoding="utf-8")
            lock = {"frozen": {"question_fingerprint_sha256": "questions",
                               "selection_protocol_id": "protocol",
                               "generation": {"budget": 256, "greedy": True, "thinking": False}}}
            result = assessment_payload(c, lock, "lock", [{"id": f"q{i}"} for i in range(250)],
                                        {"original": base, "adapter": candidate},
                                        {"original": "gpt-judge", "adapter": "gpt-judge"},
                                        {"original": [], "adapter": []},
                                        {"original": [], "adapter": []})
            self.assertFalse(result["passed"])
            self.assertEqual(result["status"], "failed_stopped")
            self.assertIn("new sealed test set is required", result["next_action"])
            self.assertIn("command_generation", result["scores"]["adapter"])


if __name__ == "__main__":
    unittest.main()
