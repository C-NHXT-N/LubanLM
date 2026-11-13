"""Full development comparison for existing LoRA checkpoints, with strict identity binding."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path

from scripts.common import sha256, write_json
from scripts.linux_assistant.common import config, fingerprint, jsonl, write_jsonl
from scripts.linux_assistant.modeling import load, evaluation_messages, render, generation_eos_ids
from scripts.linux_assistant.evaluate import evaluate


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def calibration_config(path: Path):
    c = config(path)
    workspace = Path(__file__).resolve().parents[3]
    c["config_path"] = Path(path).resolve()
    c["calibration"]["training_run_configuration"] = (
        workspace / c["calibration"]["training_run_configuration"]
    ).resolve()
    c["calibration"]["review_protocol"] = (
        workspace / c["calibration"]["review_protocol"]
    ).resolve()
    for key, value in list(c["calibration"]["reuse_existing"].items()):
        c["calibration"]["reuse_existing"][key] = (workspace / value).resolve()
    return c


def read_evaluation_rows(c):
    """Load only the 27 task + 9 general development rows; never open test rows."""
    manifest_path = c["data"] / "manifest.json"
    manifest = read_json(manifest_path)
    expected_manifest = c["calibration"]["evaluation_manifest_sha256"]
    if file_sha256(manifest_path) != expected_manifest:
        raise ValueError("Active dataset manifest differs from calibration lock")
    if manifest.get("dataset_schema") != "lubanlm.linux-assistant-dataset.v2":
        raise ValueError("Unexpected dataset schema")
    for name in ("dev.jsonl", "general-dev.jsonl"):
        expected = manifest.get("files", {}).get(name)
        if not expected or file_sha256(c["data"] / name) != expected:
            raise ValueError("Development split file hash mismatch: " + name)
    rows = jsonl(c["data"] / "dev.jsonl") + jsonl(c["data"] / "general-dev.jsonl")
    expected_data_hash = c["calibration"]["evaluation_data_hash"]
    if len(rows) != 36 or fingerprint(rows) != expected_data_hash:
        raise ValueError("Development rows do not match the pinned 36-row evaluation identity")
    content_path = c["data"] / manifest["content_review_file"]
    if file_sha256(content_path) != manifest["content_review_sha256"]:
        raise ValueError("Content review ledger changed")
    content_by_id = {row["id"]: row for row in jsonl(content_path)}
    if len(content_by_id) != len(jsonl(content_path)):
        raise ValueError("Duplicate content review IDs")
    for row in rows:
        decision = content_by_id.get(row["id"])
        if (decision is None or decision.get("decision") != "accept" or
                decision.get("final_row_sha256") != fingerprint(row)):
            raise ValueError("Development row is not bound to an accepted final review: " + row["id"])
    return manifest, rows


def verify_training_identity(c):
    train_cfg_path = c["calibration"]["training_run_configuration"]
    run_cfg = read_json(train_cfg_path)
    identity = run_cfg.get("identity")
    expected = c["calibration"]["training_identity"]
    if not isinstance(identity, dict):
        raise ValueError("Checkpoint training identity is missing")
    for key, value in expected.items():
        if identity.get(key) != value:
            raise ValueError("Training identity differs at " + key)
    return run_cfg, identity, file_sha256(train_cfg_path)


def checkpoint_identity(c, step: int, run_cfg, training_identity):
    from scripts.linux_assistant.common import verify_model

    run_folder = c["runs"].parent / "linux-assistant-qwen35-2b-exploration-20261022" / "lr-1e-4"
    # The source run is immutable and intentionally separate from these new evaluation outputs.
    checkpoint = run_folder / f"checkpoint-{step}"
    if not checkpoint.is_dir():
        raise FileNotFoundError("Missing existing checkpoint: " + str(checkpoint))
    state_path = checkpoint / "run-state.json"
    state = read_json(state_path)
    if state.get("step") != step or state.get("identity") != run_cfg.get("identity"):
        raise ValueError("Checkpoint run-state identity or step mismatch: " + str(step))
    if state["identity"].get("smoke") is not False:
        raise ValueError("Smoke checkpoint cannot enter product comparison")
    if state["identity"].get("revision") != c["revision"]:
        raise ValueError("Checkpoint was trained from an unexpected model revision")
    adapter_path = checkpoint / "adapter_model.safetensors"
    adapter_cfg_path = checkpoint / "adapter_config.json"
    if not adapter_path.is_file() or not adapter_cfg_path.is_file():
        raise ValueError("Incomplete checkpoint adapter: " + str(step))
    adapter_cfg = read_json(adapter_cfg_path)
    if Path(adapter_cfg.get("base_model_name_or_path", "")).resolve() != c["model"]:
        raise ValueError("Adapter base-model path mismatch: " + str(step))
    if adapter_cfg.get("r") != 16 or adapter_cfg.get("lora_alpha") != 32:
        raise ValueError("Unexpected LoRA structure: " + str(step))
    base_revision = verify_model(c["model"], c["revision"])
    model_manifest = c["model"] / "download_manifest.json"
    component_hashes = {
        name: file_sha256(checkpoint / name)
        for name in ("adapter_model.safetensors", "adapter_config.json", "tokenizer.json",
                     "tokenizer_config.json", "chat_template.jinja")
        if (checkpoint / name).is_file()
    }
    return {
        "step": step,
        "checkpoint_path": str(checkpoint),
        "run_state_sha256": file_sha256(state_path),
        "training_run_configuration_sha256": file_sha256(c["calibration"]["training_run_configuration"]),
        "training_identity": training_identity,
        "training_manifest_sha256": training_identity["manifest"],
        "adapter_sha256": file_sha256(adapter_path),
        "adapter_component_sha256": component_hashes,
        "base_model_path": str(c["model"]),
        "base_model_revision": base_revision,
        "base_model_download_manifest_sha256": file_sha256(model_manifest),
    }


def prepare_prompts(c, tokenizer, rows):
    from scripts.linux_assistant.modeling import render

    prepared = []
    for row in rows:
        context = evaluation_messages(c, row)
        prompt = render(tokenizer, context, True)
        token_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if len(token_ids) + c["max_new_tokens"] > c["context_length"]:
            raise ValueError("Product prompt exceeds context budget: " + row["id"])
        prepared.append({"row": row, "context": context, "prompt": prompt, "input_ids": token_ids})
    return prepared


def validate_reused_outputs(c, manifest, rows, prepared, checkpoint_map):
    """Bind prior baseline/step-61 raw and reviewed rows to the current exact prompts."""
    import importlib.util

    reuse = c["calibration"]["reuse_existing"]
    semantic_path = c["report"].parent / "linux-assistant-qwen35-2b-exploration-20261022" / "semantic-result.json"
    semantic = read_json(semantic_path)
    comparison_path = semantic_path.with_name("development-comparison.json")
    comparison = read_json(comparison_path)

    specs = [
        ("baseline", reuse["baseline_answers"], reuse["baseline_identity"], reuse["baseline_reviewed"], None),
        ("step-61", reuse["checkpoint_61_answers"], reuse["checkpoint_61_identity"], reuse["checkpoint_61_reviewed"], 61),
    ]
    validated = {}
    for label, raw_path, identity_path, reviewed_path, step in specs:
        identity = read_json(identity_path)
        raw = jsonl(raw_path)
        reviewed = jsonl(reviewed_path)
        if len(raw) != 36 or len(reviewed) != 36:
            raise ValueError(label + " historical output/review is not full-36")
        if label == "baseline":
            if identity.get("rows") != fingerprint(rows) or identity.get("revision") != c["revision"]:
                raise ValueError("Historical baseline input/model identity mismatch")
            if identity.get("system_prompt") != c["system_prompt"]:
                raise ValueError("Historical baseline system prompt mismatch")
            if identity.get("max_new_tokens") != c["max_new_tokens"] or identity.get("greedy") is not True:
                raise ValueError("Historical baseline generation policy mismatch")
            expected_raw_hash = semantic["baseline_review_sha256"]
            actual_review_hash = file_sha256(reviewed_path)
            if actual_review_hash != expected_raw_hash:
                raise ValueError("Historical baseline review hash mismatch")
            if file_sha256(raw_path) != semantic.get("baseline", {}).get("raw_answers_sha256", identity.get("raw_answers_sha256")):
                # The raw file hash is also pinned in the baseline semantic review document.
                base_review = read_json(raw_path.parent / "semantic-review.json")
                if file_sha256(raw_path) != base_review.get("raw_answers_sha256"):
                    raise ValueError("Historical baseline raw output hash mismatch")
        else:
            cp = checkpoint_map[61]
            if (identity.get("step") != 61 or identity.get("adapter") != cp["adapter_sha256"] or
                    identity.get("data_hash") != fingerprint(rows) or identity.get("system") != c["system_prompt"]):
                raise ValueError("Historical step-61 identity mismatch")
            gen = identity.get("generation", {})
            if gen.get("max_new_tokens") != c["max_new_tokens"] or gen.get("greedy") is not True or gen.get("thinking") is not False:
                raise ValueError("Historical step-61 generation policy mismatch")
            if file_sha256(raw_path) != semantic.get("raw_candidate_sha256"):
                raise ValueError("Historical step-61 raw output hash mismatch")
            if file_sha256(reviewed_path) != semantic.get("candidate_review_sha256"):
                raise ValueError("Historical step-61 review hash mismatch")

        from scripts.linux_assistant.semantic_grading import validate_model_review
        for index, (question, prompt_item, raw_row, score) in enumerate(zip(rows, prepared, raw, reviewed)):
            if raw_row.get("id") != question["id"] or score.get("id") != question["id"]:
                raise ValueError(label + " row order/id mismatch at " + str(index))
            for key in ("messages", "reference", "grading", "kind", "category"):
                if raw_row.get(key) != question.get(key) or score.get(key) != question.get(key):
                    raise ValueError(label + " question/score differs from pinned current dev row: " + question["id"] + " " + key)
            if (raw_row.get("prompt") != prompt_item["prompt"] or
                    raw_row.get("input_ids") != prompt_item["input_ids"] or
                    raw_row.get("input_messages") != prompt_item["context"][1:]):
                raise ValueError(label + " exact prompt/token identity mismatch: " + question["id"])
            if raw_row.get("answer") != score.get("answer") or score.get("reference") != question["reference"]:
                raise ValueError(label + " reviewed answer/reference mismatch: " + question["id"])
            if not isinstance(score.get("axes"), dict):
                raise ValueError(label + " has no explicit content/completeness/format axes")
            validate_model_review({**question, "answer": raw_row["answer"]}, score)
        validated[label] = {
            "raw_path": str(raw_path), "raw_sha256": file_sha256(raw_path),
            "identity_path": str(identity_path), "identity_sha256": file_sha256(identity_path),
            "reviewed_path": str(reviewed_path), "reviewed_sha256": file_sha256(reviewed_path),
            "rows": 36,
        }
    if comparison.get("development_samples") != 36 or not comparison.get("input_token_parity"):
        raise ValueError("Historical comparison report did not bind a full identical 36-row input set")
    return validated, semantic


def row_generation_status(row, eos_ids, assistant_end_id, budget=256):
    generated = row.get("generated_ids", [])
    if not generated:
        if row.get("stop_reason") in ("length_limit", "max_new_tokens"):
            return "length_limit"
        return "empty_generation"
    last = generated[-1]
    if last == assistant_end_id:
        return "complete_assistant_end"
    if last in eos_ids:
        return "complete_text_end"
    if row.get("stop_reason") in ("length_limit", "max_new_tokens") or len(generated) >= budget:
        return "length_limit"
    return "unexpected_stop"


def make_identity(c, manifest, data_hash, checkpoint, step, eos_ids, system_prompt, budget, model_name):
    split_hashes = {name: file_sha256(c["data"] / name) for name in ("dev.jsonl", "general-dev.jsonl")}
    return {
        "step": step,
        "model_name": model_name,
        "selection_scope": c["calibration"]["selection_scope"],
        "budget_scope": "product_256" if budget == c["max_new_tokens"] else "wide_1024_diagnostic",
        "model": checkpoint["base_model_path"],
        "base_model_revision": checkpoint["base_model_revision"],
        "base_model_download_manifest_sha256": checkpoint["base_model_download_manifest_sha256"],
        "training_run_configuration_sha256": checkpoint["training_run_configuration_sha256"],
        "training_identity": checkpoint["training_identity"],
        "training_manifest_sha256": checkpoint["training_manifest_sha256"],
        "checkpoint_path": checkpoint["checkpoint_path"],
        "checkpoint_run_state_sha256": checkpoint["run_state_sha256"],
        "adapter_sha256": checkpoint["adapter_sha256"],
        "adapter_component_sha256": checkpoint["adapter_component_sha256"],
        "evaluation_manifest_sha256": file_sha256(c["data"] / "manifest.json"),
        "evaluation_split_sha256": split_hashes,
        "data_hash": data_hash,
        "system": system_prompt,
        "prompt_policy": c["calibration"]["prompt_policy"],
        "generation": {
            "greedy": True,
            "max_new_tokens": budget,
            "thinking": False,
            "eos_token_ids": eos_ids,
            "stop_policy": "native_assistant_end_or_text_end_v2",
        },
    }


def run_budget_evaluation(c, model, tokenizer, rows, identity, destination, budget):
    """Run a complete, resumable 36-row pass at exactly one generation budget."""
    eval_config = dict(c)
    eval_config["max_new_tokens"] = budget
    result = evaluate(eval_config, model, tokenizer, rows, destination, identity)
    raw_path = Path(destination)
    identity_path = raw_path.with_suffix(".identity.json")
    saved_identity = read_json(identity_path)
    if saved_identity.get("data_hash") != fingerprint(rows):
        raise ValueError("Evaluation identity has a different row fingerprint: " + str(raw_path))
    if saved_identity.get("system") != c["system_prompt"]:
        raise ValueError("Evaluation identity has a different system prompt: " + str(raw_path))
    if saved_identity.get("prompt_policy") != "context_through_last_user_exclude_target_assistant":
        raise ValueError("Evaluation identity has a different prompt policy: " + str(raw_path))
    generation = saved_identity.get("generation", {})
    if (generation.get("max_new_tokens") != budget or generation.get("greedy") is not True or
            generation.get("thinking") is not False or
            generation.get("stop_policy") != "native_assistant_end_or_text_end_v2" or
            set(generation.get("eos_token_ids", [])) != set(identity["generation"]["eos_token_ids"])):
        raise ValueError("Evaluation identity generation policy mismatch: " + str(raw_path))
    return result, saved_identity


def validate_raw_against_prepared(raw_path, identity_path, identity, prepared):
    raw = jsonl(raw_path)
    if len(raw) != len(prepared):
        raise ValueError("Full evaluation is incomplete: " + str(raw_path))
    if read_json(identity_path) != identity:
        raise ValueError("Evaluation identity mismatch: " + str(identity_path))
    for item, row in zip(prepared, raw):
        question = item["row"]
        if (row.get("id") != question["id"] or row.get("prompt") != item["prompt"] or
                row.get("input_ids") != item["input_ids"] or row.get("reference") != question["reference"] or
                row.get("grading") != question.get("grading") or row.get("messages") != question["messages"]):
            raise ValueError("Raw output does not match locked question/prompt: " + question["id"])
    return raw


def write_review_queue(c, rows, output_map, reviewed_map, eos_ids, assistant_end_id, budget, destination_name):
    report = c["report"]
    report.mkdir(parents=True, exist_ok=True)
    question_by_id = {r["id"]: r for r in rows}
    queue = []
    for model_name, raw_rows in output_map.items():
        review_by_id = {r["id"]: r for r in reviewed_map.get(model_name, [])}
        if reviewed_map.get(model_name) and set(review_by_id) != {r["id"] for r in raw_rows}:
            raise ValueError("Historical review does not cover the exact output set: " + model_name)
        for row in raw_rows:
            legacy = review_by_id.get(row["id"])
            question = question_by_id[row["id"]]
            task_kind = question["kind"]
            source = question.get("dataset_source") or question.get("evidence", {}).get("upstream_dataset") or ""
            if source in c["calibration"]["rk3576_sources"]:
                family = "rk3576"
            elif source in c["calibration"]["document_command_sources"]:
                family = "document_command"
            elif task_kind == "general":
                family = "general"
            else:
                family = "other_task"
            status = row_generation_status(row, eos_ids, assistant_end_id, budget)
            legacy_axes = legacy.get("axes", {}) if legacy else {}
            axes = ({
                "content_correct": legacy_axes.get("content_correct"),
                "completeness": legacy_axes.get("completeness", legacy_axes.get("steps_complete")),
                "format_compliant": legacy_axes.get("format_compliant"),
                "generation_status": status,
            } if legacy else {
                "content_correct": None, "completeness": None,
                "format_compliant": None, "generation_status": status,
            })
            product_complete = status in ("complete_assistant_end", "complete_text_end")
            accepted = (all(axes.get(key) is True for key in ("content_correct", "completeness", "format_compliant"))
                        if all(axes.get(key) is not None for key in ("content_correct", "completeness", "format_compliant")) else None)
            queue.append({
                "model": model_name, "step": (None if model_name == "baseline" else int(model_name.split("-")[1])), "id": row["id"],
                "group": question["group"], "kind": task_kind, "category": question["category"],
                "source_family": family, "dataset_source": source,
                "messages": question["messages"], "reference": question["reference"],
                "grading": question.get("grading"), "answer": row["answer"],
                "answer_sha256": hashlib.sha256(row["answer"].encode()).hexdigest(),
                "generated_tokens_including_stop": len(row.get("generated_ids", [])),
                "generated_tokens_excluding_stop": max(0, len(row.get("generated_ids", [])) - int(status in ("complete_assistant_end", "complete_text_end"))),
                "generation_budget": budget, "generation_status": status,
                "stop_reason_raw": row.get("stop_reason"),
                "axes": axes,
                "product_complete": product_complete,
                "accepted": accepted,
                "review_reason": legacy.get("review_reason") if legacy else "",
                "reviewer": legacy.get("reviewer") if legacy else "",
                "review_origin": model_name if legacy else "new_output_full36_content_review_required",
                "diagnostic_only": budget == c["calibration"]["diagnostic_max_new_tokens"],
            })
    path = report / destination_name
    write_jsonl(path, queue)
    return path, queue


def summarize_outputs(queue):
    grouped = {}
    for row in queue:
        model = row["model"]
        key = f"{model}/{row['generation_budget']}"
        grouped.setdefault(key, []).append(row)
    summaries = {}
    for key, items in grouped.items():
        lengths = [item["generated_tokens_including_stop"] for item in items]
        statuses = {}
        for item in items:
            statuses[item["generation_status"]] = statuses.get(item["generation_status"], 0) + 1
        by_family = {}
        by_category = {}
        for field, dest in (("source_family", by_family), ("category", by_category)):
            buckets = {}
            for item in items:
                buckets.setdefault(item[field], []).append(item)
            for name, rows_in_bucket in buckets.items():
                scores = [row["accepted"] for row in rows_in_bucket if row["accepted"] is not None]
                dest[name] = {
                    "rows": len(rows_in_bucket),
                    "reviewed": len(scores),
                    "accepted": sum(score is True for score in scores),
                    "content_correct": sum(row["axes"]["content_correct"] is True for row in rows_in_bucket),
                    "completeness": sum(row["axes"]["completeness"] is True for row in rows_in_bucket),
                    "format_compliant": sum(row["axes"]["format_compliant"] is True for row in rows_in_bucket),
                }
        ordered = sorted(lengths)
        p95 = ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)] if ordered else None
        summaries[key] = {
            "rows": len(items), "generation_status_counts": statuses,
            "generated_tokens_including_stop": {
                "min": min(lengths) if lengths else None,
                "median": statistics.median(lengths) if lengths else None,
                "p95": p95, "max": max(lengths) if lengths else None,
                "mean": statistics.mean(lengths) if lengths else None,
            },
            "product_complete": sum(item["product_complete"] for item in items),
            "semantic_reviewed": sum(item["accepted"] is not None for item in items),
            "semantic_accepted": sum(item["accepted"] is True for item in items),
            "by_source_family": by_family, "by_category": by_category,
        }
    return summaries


def run_evaluations(c, steps):
    import torch

    manifest, rows = read_evaluation_rows(c)
    train_cfg, training_identity, train_cfg_sha = verify_training_identity(c)
    from scripts.linux_assistant.common import verify_model
    from scripts.linux_assistant.modeling import generation_eos_ids
    base_revision = verify_model(c["model"], c["revision"])
    if fingerprint(rows) != c["calibration"]["evaluation_data_hash"]:
        raise ValueError("Evaluation row fingerprint changed")
    run_cfg_path = c["calibration"]["training_run_configuration"]
    training_run = read_json(run_cfg_path)
    if training_run.get("identity") != training_identity:
        raise ValueError("Checkpoint parent run configuration changed")
    checkpoint_steps = c["calibration"]["steps"]
    checkpoints = {step: checkpoint_identity(c, step, training_run, training_identity) for step in checkpoint_steps}
    if set(steps) - set(checkpoint_steps):
        raise ValueError("Requested a checkpoint absent from the locked existing run")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(c["model"], local_files_only=True, trust_remote_code=False)
    prepared = prepare_prompts(c, tokenizer, rows)
    assistant_end_id = tokenizer.get_vocab().get("<|im_end|>")
    if assistant_end_id is None:
        raise ValueError("Qwen tokenizer lacks the native assistant-end token")
    # All rows fit at either budget within the pinned 2048-token context window.
    if any(len(item["input_ids"]) + c["calibration"]["diagnostic_max_new_tokens"] > c["context_length"] for item in prepared):
        raise ValueError("At least one row cannot fit the full 1024-token pass in the locked context window")
    reuse, semantic = validate_reused_outputs(c, manifest, rows, prepared, checkpoints)
    c["report"].mkdir(parents=True, exist_ok=True)
    c["runs"].mkdir(parents=True, exist_ok=True)
    preflight = {
        "status": "passed", "scope": "development_only_full_36",
        "evaluation_manifest_sha256": file_sha256(c["data"] / "manifest.json"),
        "evaluation_data_hash": fingerprint(rows), "evaluation_rows": len(rows),
        "task_rows": sum(row["kind"] == "task" for row in rows),
        "general_rows": sum(row["kind"] == "general" for row in rows),
        "training_identity_manifest_sha256": training_identity["manifest"],
        "current_evaluation_manifest_differs_from_training_manifest": training_identity["manifest"] != file_sha256(c["data"] / "manifest.json"),
        "training_configuration_sha256": train_cfg_sha, "base_model_revision": base_revision,
        "checkpoint_identities": checkpoints, "reused_historical_outputs": reuse,
        "historical_semantic_result_sha256": file_sha256(c["report"].parent / "linux-assistant-qwen35-2b-exploration-20261022" / "semantic-result.json"),
        "generation_policy": {"greedy": True, "product_max_new_tokens": c["max_new_tokens"],
                              "wide_max_new_tokens": c["calibration"]["diagnostic_max_new_tokens"],
                              "wide_applies_to_all_models_and_rows": True,
                              "thinking": False, "context_length": c["context_length"]},
    }
    write_json(c["report"] / "preflight.json", preflight)
    write_json(c["report"] / "scoring-protocol.json", scoring_protocol(c))

    product_map = {
        "baseline": jsonl(c["calibration"]["reuse_existing"]["baseline_answers"]),
        "step-61": jsonl(c["calibration"]["reuse_existing"]["checkpoint_61_answers"]),
    }
    reviewed_map = {
        "baseline": jsonl(c["calibration"]["reuse_existing"]["baseline_reviewed"]),
        "step-61": jsonl(c["calibration"]["reuse_existing"]["checkpoint_61_reviewed"]),
    }
    wide_map = {}
    wide_identities = {}
    model_specs = [("baseline", None)] + [(f"step-{step}", step) for step in checkpoint_steps]

    # Baseline wide pass: fresh model run, bound to the same exact 36 prompts.
    base_model, base_tokenizer = load(c["model"])
    live_eos = generation_eos_ids(base_model, base_tokenizer)
    base_identity = {
        "step": None, "model_name": "baseline", "selection_scope": c["calibration"]["selection_scope"],
        "budget_scope": "wide_1024_diagnostic", "model": str(c["model"]),
        "base_model_revision": base_revision,
        "base_model_download_manifest_sha256": file_sha256(c["model"] / "download_manifest.json"),
        "evaluation_manifest_sha256": preflight["evaluation_manifest_sha256"],
        "evaluation_split_sha256": {name: file_sha256(c["data"] / name) for name in ("dev.jsonl", "general-dev.jsonl")},
        "system": c["system_prompt"], "prompt_policy": c["calibration"]["prompt_policy"],
        "generation": {"greedy": True, "max_new_tokens": 1024, "thinking": False,
                       "eos_token_ids": live_eos, "stop_policy": "native_assistant_end_or_text_end_v2"},
    }
    dest = c["runs"] / "baseline-1024.jsonl"
    _, saved = run_budget_evaluation(c, base_model, base_tokenizer, rows, base_identity, dest, 1024)
    wide_map["baseline"] = validate_raw_against_prepared(dest, dest.with_suffix(".identity.json"), saved, prepared)
    wide_identities["baseline"] = {"path": str(dest), "sha256": file_sha256(dest), "identity_sha256": file_sha256(dest.with_suffix(".identity.json"))}
    del base_model, base_tokenizer
    gc.collect(); torch.cuda.empty_cache()

    for step in checkpoint_steps:
        model_name = f"step-{step}"
        checkpoint = checkpoints[step]
        model, model_tokenizer = load(c["model"])
        if (model_tokenizer.get_vocab().get("<|im_end|>") != assistant_end_id or
                model_tokenizer.eos_token_id != tokenizer.eos_token_id):
            raise ValueError("Checkpoint-local tokenizer differs from the pinned base tokenizer")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, Path(checkpoint["checkpoint_path"]), is_trainable=False)
        live_eos = generation_eos_ids(model, model_tokenizer)
        identity_root = make_identity(c, manifest, fingerprint(rows), checkpoint, step,
                                      live_eos, c["system_prompt"], 1024, model_name)

        if step != 61:
            product_identity = make_identity(c, manifest, fingerprint(rows), checkpoint, step,
                                             live_eos, c["system_prompt"], c["max_new_tokens"], model_name)
            product_dest = c["runs"] / f"{model_name}-256.jsonl"
            _, product_saved = run_budget_evaluation(c, model, model_tokenizer, rows,
                                                     product_identity, product_dest, c["max_new_tokens"])
            product_map[model_name] = validate_raw_against_prepared(
                product_dest, product_dest.with_suffix(".identity.json"), product_saved, prepared)
            wide_identity = identity_root
        else:
            wide_identity = identity_root

        wide_dest = c["runs"] / f"{model_name}-1024.jsonl"
        _, wide_saved = run_budget_evaluation(c, model, model_tokenizer, rows,
                                              wide_identity, wide_dest, 1024)
        wide_map[model_name] = validate_raw_against_prepared(
            wide_dest, wide_dest.with_suffix(".identity.json"), wide_saved, prepared)
        wide_identities[model_name] = {"path": str(wide_dest), "sha256": file_sha256(wide_dest),
                                       "identity_sha256": file_sha256(wide_dest.with_suffix(".identity.json"))}
        if step != 61:
            product_dest = c["runs"] / f"{model_name}-256.jsonl"
            product_map[model_name + "_path"] = str(product_dest)
        print(f"CALIBRATION {model_name} product={'reused' if step == 61 else '36/36'} wide=36/36", flush=True)
        del model, model_tokenizer
        gc.collect(); torch.cuda.empty_cache()

    # Preserve model-keyed output maps; path metadata is tracked separately.
    product_map = {k: v for k, v in product_map.items() if isinstance(v, list)}
    for name, path in (("baseline", c["calibration"]["reuse_existing"]["baseline_answers"]),
                       ("step-61", c["calibration"]["reuse_existing"]["checkpoint_61_answers"])):
        for item, raw_row in zip(prepared, product_map[name]):
            if raw_row.get("id") != item["row"]["id"] or raw_row.get("prompt") != item["prompt"] or raw_row.get("input_ids") != item["input_ids"]:
                raise ValueError("Reused historical output is not prompt-identical: " + name + " " + item["row"]["id"])

    product_queue_path, product_queue = write_review_queue(
        c, rows, product_map, reviewed_map, live_eos, assistant_end_id, 256, "product-review-queue.jsonl")
    wide_queue_path, wide_queue = write_review_queue(
        c, rows, wide_map, {}, live_eos, assistant_end_id, 1024, "wide-review-queue.jsonl")
    identity_files = {}
    for name in product_map:
        if name == "baseline":
            path = c["calibration"]["reuse_existing"]["baseline_answers"]
            identity_path = c["calibration"]["reuse_existing"]["baseline_identity"]
        elif name == "step-61":
            path = c["calibration"]["reuse_existing"]["checkpoint_61_answers"]
            identity_path = c["calibration"]["reuse_existing"]["checkpoint_61_identity"]
        else:
            path = c["runs"] / f"{name}-256.jsonl"
            identity_path = path.with_suffix(".identity.json")
        identity_files[name + "_256"] = {"path": str(path), "sha256": file_sha256(path),
                                          "identity_path": str(identity_path),
                                          "identity_sha256": file_sha256(identity_path)}
    identity_files.update({name + "_1024": item for name, item in wide_identities.items()})
    lock = {
        "data_hash": fingerprint(rows), "evaluation_manifest_sha256": preflight["evaluation_manifest_sha256"],
        "evaluation_rows": len(rows), "product_outputs": identity_files,
        "product_review_queue_sha256": file_sha256(product_queue_path),
        "wide_review_queue_sha256": file_sha256(wide_queue_path),
        "product_review_queue_rows": len(product_queue), "wide_review_queue_rows": len(wide_queue),
    }
    write_json(c["report"] / "evaluation-lock.json", lock)
    summary = {
        "status": "generation_complete_semantic_review_pending",
        "data_hash": fingerprint(rows), "evaluation_manifest_sha256": preflight["evaluation_manifest_sha256"],
        "models": ["baseline", "step-16", "step-32", "step-48", "step-61"],
        "rows_per_model_per_budget": 36, "budgets": [256, 1024],
        "product": summarize_outputs(product_queue), "wide": summarize_outputs(wide_queue),
        "identity_files": identity_files,
        "review_queues": {"product": str(product_queue_path), "wide": str(wide_queue_path)},
        "training_manifest_differs_from_evaluation_manifest": preflight["current_evaluation_manifest_differs_from_training_manifest"],
        "training_manifest_sha256": training_identity["manifest"],
        "current_evaluation_manifest_sha256": preflight["evaluation_manifest_sha256"],
        "historical_semantic_result_sha256": preflight["historical_semantic_result_sha256"],
        "limits": ["The sealed test set was not opened.", "New checkpoint semantic scores remain pending reviewer decisions.",
                   "The checkpoints were trained under the older manifest recorded above; this comparison binds to the exact unchanged 36-row development set."],
    }
    write_json(c["report"] / "generation-summary.json", summary)
    write_markdown_report(c, summary)
    return {"preflight": preflight, "product_queue": product_queue_path,
            "wide_queue": wide_queue_path, "summary": summary}


def scoring_protocol(c):
    return {
        "protocol_id": "checkpoint-calibration-axes-v1",
        "scope": "36-row development comparison only; never the sealed test set",
        "axes": {
            "content_correct": "True only when the answer's factual claims and requested operation/answer are correct and do not add unsupported source, board, version, or device claims. Ignore optional facts unless their inclusion creates an error.",
            "completeness": "True only when every applicable required fact and required step in the frozen grading contract is covered. Optional facts and unrequested intermediate calculations are not required. For source-excerpt questions, cover the stated purpose/example and every requested placeholder explanation. For arithmetic, require the final asked-for value/object/unit; intermediate work is optional unless explicitly required.",
            "format_compliant": "True only when the answer follows explicit format instructions in the user message and the frozen grading.format specification. Run format_check as a lower bound; passing a free-format check does not override explicit bullets, JSON, command-only, item count, or ordering requested in the prompt.",
            "generation_status": "Record complete_assistant_end, complete_text_end, length_limit, empty_generation, or unexpected_stop from generated token IDs and raw stop metadata. This is an outcome/status axis, not a semantic error flag. Record product_complete separately; reaching the token cap alone does not make the answer wrong.",
        },
        "accepted_definition": "accepted=true only when content_correct, completeness, and format_compliant are all true. generation_status and product_complete are reported separately; a length limit is not an automatic semantic failure.",
        "budgets": {"product_max_new_tokens": 256, "wide_max_new_tokens": 1024, "wide_scope": "all 36 rows for baseline and checkpoints 16, 32, 48, and 61; separate full pass, not only continuations of capped product answers", "context_length": c["context_length"]},
        "review_requirements": [
            "Score the displayed model answer, not the reference wording.",
            "Use only facts and constraints present in messages, grading, and fixed source evidence.",
            "Do not make a category pass from a keyword or regex match.",
            "Keep 1024-token outputs in a separate wide-budget record; they do not change the product-budget status or score.",
            "All rows require a reviewer reason and model/output identity; a blank or undecided axis is not a pass.",
        ],
    }


def write_markdown_report(c, summary):
    lines = [
        "# Checkpoint evaluation calibration",
        "",
        "This report compares the existing Qwen3.5-2B baseline and LoRA checkpoints on the same 36 development rows. It does not use the sealed test set or start training.",
        "",
        f"Evaluation data fingerprint: `{summary['data_hash']}`  ",
        f"Current evaluation manifest SHA256: `{summary['evaluation_manifest_sha256']}`  ",
        f"Training manifest SHA256: `{summary['training_manifest_sha256']}`  ",
        "",
        "The 36 development rows are identical to the earlier baseline and step-61 run. The checkpoint training manifest differs from the current active dataset manifest, so results describe these existing checkpoints against the current frozen development set; they are not evidence that training used the current manifest.",
        "",
        "## Product budget (256 tokens)",
        "",
        "| Model | Rows | Ended normally | Hit limit | Empty/unexpected | Reviewed | Accepted |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in ("baseline", "step-16", "step-32", "step-48", "step-61"):
        s = summary["product"].get(f"{model}/256", {})
        status = s.get("generation_status_counts", {})
        lines.append(f"| {model} | {s.get('rows', 0)} | {status.get('complete_assistant_end', 0) + status.get('complete_text_end', 0)} | {status.get('length_limit', 0)} | {status.get('empty_generation', 0) + status.get('unexpected_stop', 0)} | {s.get('semantic_reviewed', 0)} | {s.get('semantic_accepted', 0)} |")
    lines += ["", "## Wide budget (1024 tokens)", "", "Each model received a fresh complete 36-row pass at 1024 tokens, including baseline and step 61. These outputs are separate diagnostics and do not overwrite product-budget outputs.", "", "| Model | Rows | Ended normally | Hit limit | Empty/unexpected | Median tokens | P95 tokens | Max tokens |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for model in ("baseline", "step-16", "step-32", "step-48", "step-61"):
        s = summary["wide"].get(f"{model}/1024", {})
        status = s.get("generation_status_counts", {})
        lengths = s.get("generated_tokens_including_stop", {})
        lines.append(f"| {model} | {s.get('rows', 0)} | {status.get('complete_assistant_end', 0) + status.get('complete_text_end', 0)} | {status.get('length_limit', 0)} | {status.get('empty_generation', 0) + status.get('unexpected_stop', 0)} | {lengths.get('median')} | {lengths.get('p95')} | {lengths.get('max')} |")
    lines += ["", "## Semantic review", "", "New step-16/32/48 product outputs and all wide-budget outputs are in hash-bound review queues. Counts below distinguish category and source-family coverage; pending rows are not treated as failures or passes.", ""]
    for budget_label, bucket in (("Product 256", summary["product"]), ("Wide 1024", summary["wide"])):
        lines += [f"### {budget_label}", "", "| Model | Source family | Rows | Reviewed | Accepted |", "|---|---|---:|---:|---:|"]
        for model in ("baseline", "step-16", "step-32", "step-48", "step-61"):
            s = bucket.get(f"{model}/{256 if budget_label.startswith('Product') else 1024}", {})
            for family in ("document_command", "rk3576", "general", "other_task"):
                f = s.get("by_source_family", {}).get(family)
                if f:
                    lines.append(f"| {model} | {family} | {f['rows']} | {f['reviewed']} | {f['accepted']} |")
        lines.append("")
    lines += ["## Files", "", f"- Product scoring queue: `{c['report'] / 'product-review-queue.jsonl'}`", f"- Wide-budget scoring queue: `{c['report'] / 'wide-review-queue.jsonl'}`", f"- Generation summary: `{c['report'] / 'generation-summary.json'}`", f"- Identity lock: `{c['report'] / 'evaluation-lock.json'}`", "", "Content/completeness/format decisions for new outputs remain pending. The sealed test set was not read."]
    (c["report"] / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[2] / "configs/linux-assistant-checkpoint-calibration-20261022.json")
    parser.add_argument("--steps", type=int, nargs="+", default=[16, 32, 48, 61])
    args = parser.parse_args()
    c = calibration_config(args.config)
    result = run_evaluations(c, args.steps)
    print(json.dumps({"preflight": result["preflight"]["status"],
                      "product_review_queue": str(result["product_queue"]),
                      "wide_review_queue": str(result["wide_queue"]),
                      "product_rows": 180, "wide_rows": 180,
                      "status": result["summary"]["status"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
