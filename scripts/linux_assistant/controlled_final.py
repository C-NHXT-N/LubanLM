"""One locked, paired final evaluation of the base model and selected adapter."""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
from pathlib import Path

from scripts.common import WORKSPACE, sha256, write_json
from scripts.linux_assistant.common import config, fingerprint, jsonl, verify_data, verify_model
from scripts.linux_assistant.select_controlled import _bind_review, _score, eligible, summarize


ROUND = 1
EXPECTED_TEST_ROWS = 250
GENERATION_BUDGET = 256
PROMPT_POLICY = "context_through_last_user_exclude_target_assistant"
OUTPUT_PREFIX = "controlled-final"
CODE_FILES = (
    "LubanLM/scripts/linux_assistant/controlled_final.py",
    "LubanLM/scripts/linux_assistant/select_controlled.py",
    "LubanLM/scripts/linux_assistant/selection_bindings.py",
    "LubanLM/scripts/linux_assistant/semantic_grading.py",
    "LubanLM/scripts/linux_assistant/common.py",
    "LubanLM/scripts/linux_assistant/evaluate.py",
    "LubanLM/scripts/linux_assistant/modeling.py",
    "LubanLM/scripts/linux_assistant/baseline.py",
)


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def lock_path(c):
    return c["report"] / f"{OUTPUT_PREFIX}-lock.json"


def output_path(c, stage):
    if stage not in ("original", "adapter"):
        raise ValueError("Unknown final evaluation stage")
    return c["report"] / f"{OUTPUT_PREFIX}-{stage}.jsonl"


def review_path(c, stage):
    return c["report"] / f"{OUTPUT_PREFIX}-{stage}.reviewed.jsonl"


def _same_candidate(left, right):
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False
    required = ("step", "adapter_sha256", "adapter")
    if any(left.get(key) != right.get(key) for key in required):
        return False
    return Path(left["adapter"]).resolve() == Path(right["adapter"]).resolve()


def validate_selection(c, development_data_hash):
    """Require a candidate decision bound to the complete development set."""
    report = c["report"]
    selection_file = report / "candidate-selection.json"
    if not selection_file.is_file():
        raise ValueError("No development-selected candidate-selection.json")
    selection = read_json(selection_file)
    if selection.get("passed") is not True:
        raise ValueError("Candidate selection is not marked eligible")
    if type(selection.get("step")) is not int or not selection.get("adapter_sha256"):
        raise ValueError("Candidate selection is missing step or adapter hash")
    budget = c.get("max_new_tokens")
    protocol = c.get("selection_protocol_id", "controlled-eval-axes-v1")
    decision_paths = {
        "first": report / "first-run-decision.json",
        "second": report / "second-run-decision.json",
    }
    decisions = {}
    for name, path in decision_paths.items():
        if not path.is_file():
            continue
        decision = read_json(path)
        if (decision.get("data_hash") != development_data_hash or
                decision.get("generation_budget") != budget or
                decision.get("protocol_id") != protocol):
            raise ValueError(f"{name} run decision has a different data hash, budget, or protocol")
        decisions[name] = {"path": str(path.resolve()), "sha256": sha256(path), "document": decision}
        from scripts.linux_assistant.selection_bindings import validate_selection_inputs
        validate_selection_inputs(c,decision,'lr-1e-4' if name=='first' else 'lr-5e-5')
    if not decisions:
        raise ValueError("Neither first-run nor second-run selection decision exists")

    matches = []
    other_selected = []
    for name, entry in decisions.items():
        decision = entry["document"]
        candidate = decision.get("selected")
        if candidate is None:
            continue
        if decision.get("passed") is not True or candidate.get("passed") is not True:
            raise ValueError(f"{name} decision contains a non-eligible selected candidate")
        candidates = [row for row in decision.get("candidates", [])
                      if row.get("step") == candidate.get("step")]
        if len(candidates) != 1 or not _same_candidate(candidates[0], candidate):
            raise ValueError(f"{name} decision selected entry is not present in its candidate list")
        if _same_candidate(candidate, selection):
            matches.append((name, candidate, candidates[0]))
        else:
            other_selected.append(name)
    if len(matches) != 1 or other_selected:
        raise ValueError("Candidate selection is ambiguous or does not match exactly one eligible run decision")
    decision_name, candidate, selected_record = matches[0]
    if decision_name=='second' and ('first' not in decisions or
            decisions['first']['document'].get('passed') is not False or
            decisions['first']['document'].get('selected') is not None):
        raise ValueError('Second learning rate requires a complete failed first-run decision')
    if candidate != selection or selected_record != candidate:
        raise ValueError("Candidate selection differs from the exact eligible run-decision record")

    adapter_dir = Path(selection["adapter"]).resolve()
    runs_root = Path(c["runs"]).resolve()
    if not adapter_dir.is_relative_to(runs_root):
        raise ValueError("Selected adapter is outside the configured run directory")
    adapter_file = adapter_dir / "adapter_model.safetensors"
    adapter_config_file = adapter_dir / "adapter_config.json"
    if not adapter_file.is_file() or not adapter_config_file.is_file():
        raise ValueError("Selected adapter files are incomplete")
    actual_adapter_sha = sha256(adapter_file)
    if actual_adapter_sha != selection["adapter_sha256"]:
        raise ValueError("Selected adapter weights changed after development selection")
    if sha256(adapter_config_file)!=selection.get('adapter_config_sha256'):
        raise ValueError('Selected adapter configuration changed after development selection')
    adapter_config = read_json(adapter_config_file)
    if Path(adapter_config.get("base_model_name_or_path", "")).resolve() != Path(c["model"]).resolve():
        raise ValueError("Selected adapter does not target the configured base model")

    run_dir = adapter_dir.parent
    step = selection["step"]
    for field, path in (("raw_sha256", run_dir / f"dev-{step}.jsonl"),
                        ("review_sha256", run_dir / f"dev-{step}.reviewed.jsonl")):
        if not path.is_file() or sha256(path) != selected_record.get(field):
            raise ValueError(f"Selected development {field} is missing or changed")
    return selection, decision_name, decisions


def _evaluation_rows(c, manifest):
    declared = set(manifest.get("data_splits", []))
    names = ("test.jsonl", "general-test.jsonl")
    if not set(names).issubset(declared):
        raise ValueError("Manifest does not declare both sealed test splits")
    contract = c.get("dataset_contract", {}).get("evaluation", {})
    if contract.get("total") != EXPECTED_TEST_ROWS:
        raise ValueError("Controlled final evaluation requires the pinned 250-question test contract")
    rows = jsonl(c["data"] / names[0]) + jsonl(c["data"] / names[1])
    if len(rows) != EXPECTED_TEST_ROWS or len({row.get("id") for row in rows}) != EXPECTED_TEST_ROWS:
        raise ValueError("Sealed evaluation set must contain 250 unique rows")
    expected_capabilities = contract.get("capabilities")
    if expected_capabilities:
        actual = dict(Counter(row.get("evaluation_capability") for row in rows))
        if actual != expected_capabilities:
            raise ValueError("Sealed evaluation capability counts differ from the dataset contract")
    for row in rows:
        if not isinstance(row.get("messages"), list) or not row["messages"]:
            raise ValueError("Sealed evaluation row has no messages: " + str(row.get("id")))
        if not row.get("grading"):
            raise ValueError("Sealed evaluation row has no frozen grading rules: " + str(row.get("id")))
    return rows


def _development_rows(c, manifest):
    declared = set(manifest.get("data_splits", []))
    names = ("dev.jsonl", "general-dev.jsonl")
    if not set(names).issubset(declared):
        raise ValueError("Manifest does not declare both development splits")
    contract = c.get("dataset_contract", {}).get("evaluation", {})
    if contract.get("total") != EXPECTED_TEST_ROWS:
        raise ValueError("Controlled candidate selection requires the pinned 250-question development contract")
    rows = jsonl(c["data"] / names[0]) + jsonl(c["data"] / names[1])
    if len(rows) != EXPECTED_TEST_ROWS or len({row.get("id") for row in rows}) != EXPECTED_TEST_ROWS:
        raise ValueError("Development selection set must contain 250 unique rows")
    return rows


def _code_hashes():
    return {name: sha256(WORKSPACE / name) for name in CODE_FILES}


def collect_frozen_inputs(c, config_file):
    if c.get("max_new_tokens") != GENERATION_BUDGET:
        raise ValueError("Controlled final generation budget must be exactly 256 tokens")
    if not c.get("system_prompt") or not c.get("selection_protocol_id"):
        raise ValueError("System prompt and selection protocol must be explicit")
    manifest = verify_data(c)
    manifest_file = c["data"] / "manifest.json"
    development_rows = _development_rows(c, manifest)
    development_data_hash = fingerprint(development_rows)
    questions = _evaluation_rows(c, manifest)
    question_hash = fingerprint(questions)
    selection, decision_name, decisions = validate_selection(c, development_data_hash)

    revision = verify_model(c["model"], c["revision"])
    if revision != c["revision"]:
        raise ValueError("Configured base-model revision does not match local model files")
    model_manifest_path = c["model"] / "download_manifest.json"
    frozen = {
        "round": ROUND,
        "config_sha256": sha256(config_file),
        "dataset_manifest_sha256": sha256(manifest_file),
        "development_files_sha256": {
            name: sha256(c["data"] / name) for name in ("dev.jsonl", "general-dev.jsonl")
        },
        "development_question_count": len(development_rows),
        "development_fingerprint_sha256": development_data_hash,
        "test_files_sha256": {
            name: sha256(c["data"] / name) for name in ("test.jsonl", "general-test.jsonl")
        },
        "question_count": len(questions),
        "question_fingerprint_sha256": question_hash,
        "base_model": {
            "path": str(Path(c["model"]).resolve()),
            "revision": revision,
            "download_manifest_sha256": sha256(model_manifest_path),
        },
        "selected_candidate": {
            "step": selection["step"],
            "adapter_path": str(Path(selection["adapter"]).resolve()),
            "adapter_sha256": selection["adapter_sha256"],
            "adapter_config_sha256": selection['adapter_config_sha256'],
              "development_raw_sha256": selection["raw_sha256"],
              "development_review_sha256": selection["review_sha256"],
            "selection_sha256": sha256(c["report"] / "candidate-selection.json"),
            "decision_name": decision_name,
            "decision_sha256": decisions[decision_name]["sha256"],
        },
        "selection_decisions_sha256": {
            name: entry["sha256"] for name, entry in sorted(decisions.items())
        },
        "selection_protocol_id": c["selection_protocol_id"],
        "generation": {
            "budget": GENERATION_BUDGET,
            "greedy": True,
            "thinking": False,
            "context_length": c["context_length"],
            "system_prompt_sha256": __import__("hashlib").sha256(
                c["system_prompt"].encode("utf-8")).hexdigest(),
            "prompt_policy": PROMPT_POLICY,
        },
        "review_rule_code_sha256": _code_hashes(),
    }
    return frozen, questions


def create_lock(c, config_file):
    destination = lock_path(c)
    if destination.exists():
        raise FileExistsError("Final round is already locked; this one-round lock cannot be replaced")
    for stage in ("original", "adapter"):
        if output_path(c, stage).exists() or output_path(c, stage).with_suffix(".identity.json").exists():
            raise FileExistsError("Final output exists without this round's lock; preserve it and investigate")
    for name in (f"{OUTPUT_PREFIX}-generation.json", f"{OUTPUT_PREFIX}-assessment.json"):
        if (c["report"] / name).exists():
            raise FileExistsError("Final-round report exists without this round's lock; preserve it and investigate")
    frozen, _ = collect_frozen_inputs(c, config_file)
    lock = {
        "schema": "lubanlm.controlled-final-lock.v1",
        "status": "locked",
        "scope": "paired_original_and_selected_adapter_only",
        "test_set_opened_for_generation": False,
        "frozen": frozen,
    }
    write_json(destination, lock)
    return {"lock_path": str(destination), "lock_sha256": sha256(destination),
            "question_count": frozen["question_count"],
            "question_fingerprint_sha256": frozen["question_fingerprint_sha256"],
            "selected_step": frozen["selected_candidate"]["step"]}


def verify_current_lock(c, config_file):
    path = lock_path(c)
    if not path.is_file():
        raise FileNotFoundError("Run lock before generating or assessing the final round")
    lock = read_json(path)
    if lock.get("schema") != "lubanlm.controlled-final-lock.v1" or lock.get("status") != "locked":
        raise ValueError("Unsupported or invalid final lock")
    frozen, questions = collect_frozen_inputs(c, config_file)
    if lock.get("frozen") != frozen:
        raise ValueError("Configuration, data, model, adapter, selection, or rule code changed after lock")
    return lock, sha256(path), questions


def _expected_identity(c, frozen, lock_sha, stage):
    adapter_sha = frozen["selected_candidate"]["adapter_sha256"] if stage == "adapter" else None
    return {
        "final_round": ROUND,
        "lock_sha256": lock_sha,
        "stage": stage,
        "model": str(Path(c["model"]).resolve()),
        "revision": frozen["base_model"]["revision"],
        "adapter_path": frozen["selected_candidate"]["adapter_path"] if stage == "adapter" else None,
        "adapter_sha256": adapter_sha,
        "candidate_selection_sha256": frozen["selected_candidate"]["selection_sha256"],
        "dataset_manifest_sha256": frozen["dataset_manifest_sha256"],
        "selection_protocol_id": frozen["selection_protocol_id"],
        "question_fingerprint_sha256": frozen["question_fingerprint_sha256"],
        "context_length": frozen["generation"]["context_length"],
        "max_new_tokens": GENERATION_BUDGET,
        "greedy": True,
        "thinking": False,
    }


def _validate_output_identity(identity, expected, c):
    for key, value in expected.items():
        if identity.get(key) != value:
            raise ValueError("Existing final output identity differs at " + key)
    if identity.get("data_hash") != expected["question_fingerprint_sha256"]:
        raise ValueError("Existing final output is bound to another question set")
    if identity.get("system") != c["system_prompt"] or identity.get("prompt_policy") != PROMPT_POLICY:
        raise ValueError("Existing final output system prompt or prompt policy changed")
    generation = identity.get("generation")
    if (not isinstance(generation, dict) or generation.get("greedy") is not True or
            generation.get("thinking") is not False or
            generation.get("max_new_tokens") != GENERATION_BUDGET or
            generation.get("stop_policy") != "native_assistant_end_or_text_end_v2" or
            not isinstance(generation.get("eos_token_ids"), list) or
            not generation.get("eos_token_ids")):
        raise ValueError("Existing final output generation identity is incomplete or differs")


def _validate_raw_prefix(raw_rows, prepared):
    if len(raw_rows) > len(prepared):
        raise ValueError("Existing final output contains more rows than the locked question set")
    for raw, item in zip(raw_rows, prepared):
        question = item["row"]
        expected_fields = {
            "id": question["id"],
            "messages": question["messages"],
            "reference": question["reference"],
            "prompt": item["prompt"],
            "input_ids": item["input_ids"],
            "input_messages": item["context"][1:],
            "group": question["group"],
            "kind": question["kind"],
            "category": question["category"],
        }
        if "grading" in question:
            expected_fields["grading"] = question["grading"]
        for key, value in expected_fields.items():
            if raw.get(key) != value:
                raise ValueError(f"Existing final output changed locked question field {key}: {question['id']}")
        if not isinstance(raw.get("answer"), str) or not isinstance(raw.get("generated_ids"), list):
            raise ValueError("Existing final output row is missing generated content: " + question["id"])


def inspect_existing_output(c, frozen, lock_sha, stage, tokenizer, questions):
    """Return True only for a complete, identity- and prompt-verified reusable output."""
    from scripts.linux_assistant.baseline import prepare

    output = output_path(c, stage)
    identity_file = output.with_suffix(".identity.json")
    if not output.exists() and not identity_file.exists():
        return False
    if output.exists() and not identity_file.exists():
        raise ValueError("Existing final output has no lock identity and cannot be reused")
    if identity_file.exists():
        _validate_output_identity(read_json(identity_file), _expected_identity(c, frozen, lock_sha, stage), c)
    prepared = prepare(c, tokenizer, questions)
    raw_rows = jsonl(output) if output.exists() else []
    _validate_raw_prefix(raw_rows, prepared)
    if output.exists() and len(raw_rows) == len(questions):
        return True
    return False


def _generate(c, config_file):
    lock, lock_sha, questions = verify_current_lock(c, config_file)
    frozen = lock["frozen"]
    if c["context_length"] <= GENERATION_BUDGET:
        raise ValueError("Context length must leave room for the 256-token generation budget")
    from transformers import AutoTokenizer
    from scripts.linux_assistant.modeling import load
    from scripts.linux_assistant.baseline import exclusive_job

    tokenizer = AutoTokenizer.from_pretrained(c["model"], local_files_only=True, trust_remote_code=False)
    reusable = {stage: inspect_existing_output(c, frozen, lock_sha, stage, tokenizer, questions)
                for stage in ("original", "adapter")}
    if all(reusable.values()):
        return {"status": "reused_complete_outputs", "lock_sha256": lock_sha,
                "outputs": {stage: {"path": str(output_path(c, stage)), "sha256": sha256(output_path(c, stage))}
                            for stage in reusable}}

    import torch
    from peft import PeftModel
    from scripts.linux_assistant.evaluate import evaluate

    output_status = {}
    with exclusive_job(c["runs"] / "controlled-final-worker.lock"), \
            exclusive_job(WORKSPACE / "runs/linux-assistant-gpu.lock"):
        torch.set_num_threads(c.get("baseline", {}).get("cpu_threads", 4))
        torch.manual_seed(c.get("seed", 0))
        torch.cuda.manual_seed_all(c.get("seed", 0))
        for stage in ("original", "adapter"):
            if reusable[stage]:
                output_status[stage] = "verified_reuse"
                continue
            model, stage_tokenizer = load(c["model"])
            if stage == "adapter":
                model = PeftModel.from_pretrained(
                    model, frozen["selected_candidate"]["adapter_path"], is_trainable=False)
            identity = _expected_identity(c, frozen, lock_sha, stage)
            evaluate(c, model, stage_tokenizer, questions, output_path(c, stage), identity)
            output_status[stage] = "completed"
            del model, stage_tokenizer
            gc.collect()
            torch.cuda.empty_cache()
    result = {"status": "generation_complete", "lock_sha256": lock_sha,
              "outputs": {stage: {"status": output_status[stage], "path": str(output_path(c, stage)),
                                  "sha256": sha256(output_path(c, stage))}
                          for stage in output_status},
              "next_action": "Obtain independent model review for both complete outputs, then run assess."}
    write_json(c["report"] / f"{OUTPUT_PREFIX}-generation.json", result)
    return result


def _validate_reviewer_identity(reviewed, stage, c, revision):
    identities = {str(row.get("reviewer", "")).strip() for row in reviewed}
    if "" in identities or len(identities) != 1:
        raise ValueError(f"{stage} review must identify one consistent reviewer on every row")
    reviewer = next(iter(identities))
    folded = reviewer.casefold()
    base_label = Path(c["model"]).name.casefold()
    if reviewer in (stage, str(Path(c["model"]).resolve()), revision) or revision.casefold() in folded or base_label in folded:
        raise ValueError(f"{stage} semantic review is not independent of the evaluated base model")
    return reviewer


def _verify_review_unchanged(questions, raw, reviewed):
    if len(raw) != len(questions) or len(reviewed) != len(questions):
        raise ValueError("Both raw and reviewed outputs must contain all 250 locked questions")
    immutable = ("id", "answer", "prompt", "input_ids", "generated_ids", "messages",
                 "input_messages", "reference", "grading", "group", "kind", "category")
    for question, result, decision in zip(questions, raw, reviewed):
        if question["id"] != result.get("id") or question["id"] != decision.get("id"):
            raise ValueError("Raw or reviewed output order differs from the locked questions")
        for key in immutable:
            if key not in result or key not in decision or result.get(key) != decision.get(key):
                raise ValueError(f"Reviewer changed or omitted original output field {key}: {question['id']}")
        for key in ("messages", "reference", "grading", "group", "kind", "category"):
            if key in question and result.get(key) != question.get(key):
                raise ValueError(f"Raw final output differs from locked question field {key}: {question['id']}")


def assessment_payload(c, lock, lock_sha, questions, scores, reviewers, raw_by_stage, reviewed_by_stage):
    passed = eligible(scores["original"], scores["adapter"], c["acceptance"])
    comparison = summarize(scores["original"], scores["adapter"], c.get("seed", 20261019))
    return {
        "status": "passed" if passed else "failed_stopped",
        "passed": passed,
        "round": ROUND,
        "lock_sha256": lock_sha,
        "question_count": len(questions),
        "question_fingerprint_sha256": lock["frozen"]["question_fingerprint_sha256"],
        "selection_protocol_id": lock["frozen"]["selection_protocol_id"],
        "generation_policy": lock["frozen"]["generation"],
        "independent_reviewers": reviewers,
        "scores": {stage: {name: {key: value for key, value in metric.items() if key != "values"}
                            for name, metric in stage_scores.items()}
                   for stage, stage_scores in scores.items()},
        "adapter_comparison": comparison,
        "raw_output_sha256": {stage: sha256(output_path(c, stage)) for stage in raw_by_stage},
        "review_file_sha256": {stage: sha256(review_path(c, stage)) for stage in reviewed_by_stage},
        "merge_evaluated": False,
        "board_evaluated": False,
        "next_action": ("Final original-versus-adapter evaluation passed." if passed else
                        "Stop. Do not tune against these sealed answers; a new sealed test set is required before another final round."),
    }


def assess_outputs(c, lock, lock_sha, questions):
    from scripts.linux_assistant.common import reviewed_scores
    from transformers import AutoTokenizer
    from scripts.linux_assistant.baseline import prepare

    protocol = lock["frozen"]["selection_protocol_id"]
    raw_by_stage, reviewed_by_stage, scores, reviewers = {}, {}, {}, {}
    tokenizer = AutoTokenizer.from_pretrained(c["model"], local_files_only=True, trust_remote_code=False)
    prepared = prepare(c, tokenizer, questions)
    for stage in ("original", "adapter"):
        output = output_path(c, stage)
        identity_file = output.with_suffix(".identity.json")
        if not output.is_file() or not identity_file.is_file():
            raise ValueError(f"Missing complete {stage} final output or identity")
        identity = read_json(identity_file)
        _validate_output_identity(identity, _expected_identity(c, lock["frozen"], lock_sha, stage), c)
        raw = jsonl(output)
        reviewed_file = review_path(c, stage)
        if not reviewed_file.is_file():
            raise FileNotFoundError(f"Missing independent {stage} review: {reviewed_file}")
        reviewed = jsonl(reviewed_file)
        _validate_raw_prefix(raw, prepared)
        _verify_review_unchanged(questions, raw, reviewed)
        reviewers[stage] = _validate_reviewer_identity(reviewed, stage, c, lock["frozen"]["base_model"]["revision"])
        if any(row.get("scoring_protocol_id") != protocol for row in reviewed):
            raise ValueError(f"{stage} review uses a different scoring protocol")
        # Validate each independent judgment against the shared semantic/format contract.
        reviewed_scores(reviewed)
        _bind_review(questions, raw, reviewed, protocol)
        scores[stage] = _score(_bind_review(questions, raw, reviewed, protocol))
        raw_by_stage[stage] = raw
        reviewed_by_stage[stage] = reviewed
    if reviewers["original"] != reviewers["adapter"]:
        raise ValueError("Original and adapter outputs must use the same independent reviewer")

    result = assessment_payload(c, lock, lock_sha, questions, scores, reviewers, raw_by_stage, reviewed_by_stage)
    destination = c["report"] / f"{OUTPUT_PREFIX}-assessment.json"
    write_json(destination, result)
    return result


def run(action, config_file):
    c = config(config_file)
    if action == "lock":
        return create_lock(c, config_file)
    if action == "generate":
        return _generate(c, config_file)
    if action == "assess":
        lock, lock_sha, questions = verify_current_lock(c, config_file)
        return assess_outputs(c, lock, lock_sha, questions)
    raise ValueError("Unknown action: " + action)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("action", choices=("lock", "generate", "assess"))
    args = parser.parse_args()
    result = run(args.action, args.config.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
