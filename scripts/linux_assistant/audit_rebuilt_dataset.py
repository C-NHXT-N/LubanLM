"""Validate a source-reviewed partial/full dataset without running model inference."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from functools import lru_cache

from transformers import AutoTokenizer
from scripts.common import WORKSPACE, sha256, write_json
from scripts.linux_assistant.common import config, jsonl, fingerprint
from scripts.linux_assistant.dataset_quality import split_issues
from scripts.linux_assistant.modeling import encode_training, evaluation_messages


def input_token_count(encoded) -> int:
    """Count IDs for either a plain token list or Transformers BatchEncoding."""
    token_ids = encoded["input_ids"] if hasattr(encoded, "keys") and "input_ids" in encoded else encoded
    return len(token_ids)


@lru_cache(maxsize=8)
def imported_source_records(path_string,expected_sha256):
    path=Path(path_string)
    if sha256(path)!=expected_sha256:raise ValueError('Imported source file hash mismatch')
    if path.suffix=='.parquet':
        import pyarrow.parquet as pq
        return pq.read_table(path).to_pylist()
    if path.suffix=='.gz':
        import gzip,hashlib
        with gzip.open(path,'rb') as stream:
            return {r['message_id']:(r,hashlib.sha256(line).hexdigest()) for line in stream
                    for r in [json.loads(line)]}
    import hashlib
    return [(json.loads(line),hashlib.sha256(line).hexdigest()) for line in path.read_bytes().splitlines(keepends=True)]


def verify_source_bytes(row: dict, active_split: str | None = None) -> None:
    """Recompute pinned upstream/local evidence hashes; review flags alone are insufficient."""
    evidence = row.get("evidence", {})
    root = WORKSPACE / "provenance/linux-assistant-upstream"
    repository = evidence.get("upstream_repository", "")
    dataset=row.get('dataset_source')
    if dataset=='openai/gsm8k':
        records=imported_source_records(str(WORKSPACE/evidence['source_file']),evidence['source_file_sha256'])
        source=records[evidence['source_row_index_zero_based']]
        import hashlib
        if (hashlib.sha256(source['question'].encode()).hexdigest()!=evidence['source_question_sha256'] or
                hashlib.sha256(source['answer'].encode()).hexdigest()!=evidence['source_answer_sha256']):
            raise ValueError('GSM8K original row mismatch: '+row['id'])
        if evidence.get('upstream_split')!='train':raise ValueError('Official GSM8K heldout must not enter this experiment')
        return
    if dataset=='databricks/databricks-dolly-15k':
        records=imported_source_records(str(WORKSPACE/evidence['source_file']),evidence['source_file_sha256'])
        source,digest=records[evidence['source_row_index_0based']]
        if digest!=evidence['source_raw_line_sha256_including_lf']:
            raise ValueError('Dolly original row mismatch: '+row['id'])
        return
    if dataset=='OpenAssistant/oasst2':
        records=imported_source_records(str(WORKSPACE/evidence['source_file']),evidence['source_gzip_sha256'])
        for item in evidence['parent_chain']:
            source,digest=records[item['message_id']]
            if digest!=item['line_sha256_including_lf'] or source['parent_id']!=item['parent_id']:
                raise ValueError('OASST2 original conversation mismatch: '+row['id'])
        return
    if repository.endswith("/tldr"):
        if evidence.get("upstream_commit") != "d8bcc0a2cd2a6c02881b50031d998cae3dd35ed7":
            raise ValueError("Unexpected tldr commit: " + row["id"])
        source_root = root / "tldr-pages-tldr"
        source_path = source_root / evidence["source_page_path"]
        if not source_path.is_file() or sha256(source_path) != evidence.get("source_page_sha256"):
            raise ValueError("tldr source page hash mismatch: " + row["id"])
        license_path = source_root / evidence["license_file_path"]
        if not license_path.is_file() or sha256(license_path) != evidence.get("license_file_sha256"):
            raise ValueError("tldr license hash mismatch: " + row["id"])
        guide = evidence.get("placeholder_syntax_source")
        if guide:
            guide_path = source_root / guide
            if not guide_path.is_file() or sha256(guide_path) != evidence.get("placeholder_syntax_source_sha256"):
                raise ValueError("tldr placeholder guide hash mismatch: " + row["id"])
        return

    if repository.endswith("/nl2bash"):
        if evidence.get("upstream_commit") != "d6b9f5bdff45621d190134e31ab63b7bf7002190":
            raise ValueError("Unexpected NL2Bash commit: " + row["id"])
        source_root = root / "nl2bash"
        line_no = evidence.get("source_pair_line_number")
        if not isinstance(line_no, int) or line_no < 1:
            raise ValueError("Missing NL2Bash source line number: " + row["id"])
        nl_path = source_root / evidence["source_nl_path"]
        cm_path = source_root / evidence["source_cm_path"]
        if not nl_path.is_file() or not cm_path.is_file():
            raise ValueError("NL2Bash source files are missing: " + row["id"])
        nl_lines = nl_path.read_bytes().splitlines(keepends=True)
        cm_lines = cm_path.read_bytes().splitlines(keepends=True)
        if line_no > len(nl_lines) or line_no > len(cm_lines):
            raise ValueError("NL2Bash source line is out of range: " + row["id"])
        nl_line, cm_line = nl_lines[line_no - 1], cm_lines[line_no - 1]
        digest = lambda value: __import__("hashlib").sha256(value).hexdigest()
        if digest(nl_line) != evidence.get("source_nl_raw_line_sha256"):
            raise ValueError("NL2Bash raw NL line hash mismatch: " + row["id"])
        if digest(cm_line) != evidence.get("source_cm_raw_line_sha256"):
            raise ValueError("NL2Bash raw command line hash mismatch: " + row["id"])
        if digest(nl_line + b"\0" + cm_line) != evidence.get("source_raw_record_sha256"):
            raise ValueError("NL2Bash raw pair hash mismatch: " + row["id"])
        if row.get("reference") != evidence.get("source_command_exact"):
            raise ValueError("NL2Bash command differs from pinned source: " + row["id"])
        if active_split is not None and evidence.get("source_official_final_split") != active_split:
            raise ValueError("NL2Bash row is in the wrong active split: " + row["id"])
        if active_split == "train" and evidence.get("source_initial_split") != "train":
            raise ValueError("NL2Bash official held-out data entered training: " + row["id"])
        license_path = source_root / evidence["source_license_path"]
        if not license_path.is_file() or sha256(license_path) != evidence.get("source_license_sha256"):
            raise ValueError("NL2Bash data license hash mismatch: " + row["id"])
        return

    sources = row.get("sources", [])
    if not sources:
        raise ValueError("No local source evidence: " + row["id"])
    for source in sources:
        if not isinstance(source, dict) or not source.get("path") or not source.get("sha256"):
            raise ValueError("Malformed local source evidence: " + row["id"])
        path = (WORKSPACE / source["path"]).resolve()
        if not path.is_file() or sha256(path) != source["sha256"]:
            raise ValueError("Local evidence file hash mismatch: " + row["id"] + " " + source["path"])


def audit(data: Path | None = None, report_path: Path | None = None) -> dict:
    c = config()
    data = (data or c["data"]).resolve()
    report_path = (report_path or WORKSPACE / "reports/linux-assistant-rebuild-20261020/active-dataset-qa.json").resolve()
    manifest_path = data / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    if manifest.get("dataset_schema") != "lubanlm.linux-assistant-dataset.v2":
        raise ValueError("Unexpected rebuilt dataset schema")
    if manifest.get("content_review_status") != "complete":
        raise ValueError("Every included row must have a completed content review")
    for name, digest in manifest["files"].items():
        path = data / name
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"Manifest hash mismatch: {name}")

    split_names = manifest.get("data_splits")
    if not isinstance(split_names, list) or not split_names:
        raise ValueError("Dataset must list explicit split files")
    splits = {Path(name).stem.replace("-", "_"): jsonl(data / name) for name in split_names}
    rows = [row for values in splits.values() for row in values]
    ids = [row.get("id") for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate IDs across dataset splits")
    issues = split_issues(splits)
    if issues:
        raise ValueError("Structural or leakage checks failed: " + json.dumps(issues[:10], ensure_ascii=False))

    source_ledger_path = data / "source-review.jsonl"
    lock_path = data / "source-review-lock.json"
    content_path = data / manifest.get("content_review_file", "content-review.jsonl")
    if not source_ledger_path.is_file() or not lock_path.is_file() or not content_path.is_file():
        raise ValueError("Pinned source and content review ledgers are required")
    if manifest.get("content_review_sha256") != sha256(content_path):
        raise ValueError("Content-review digest is missing or stale")
    source_rows = jsonl(source_ledger_path)
    lock = json.loads(lock_path.read_text(encoding="utf8"))
    if (lock.get("review_schema") != "lubanlm.source-review.v2" or
            lock.get("status") != "complete" or lock.get("sha256") != sha256(source_ledger_path) or
            lock.get("review_count") != len(rows)):
        raise ValueError("Source-review lock is incomplete or stale")
    content_rows = jsonl(content_path)
    source_by_id = {item.get("id"): item for item in source_rows}
    content_by_id = {item.get("id"): item for item in content_rows}
    expected = set(ids)
    if len(source_by_id) != len(source_rows) or set(source_by_id) != expected:
        raise ValueError("Source-review ledger must cover every included row exactly once")
    if len(content_by_id) != len(content_rows) or set(content_by_id) != expected:
        raise ValueError("Content-review ledger must cover every included row exactly once")
    row_splits = {row["id"]: split for split, values in splits.items() for row in values}
    for row in rows:
        author = row.get("created_by") or row.get("evidence", {}).get("created_by")
        source = source_by_id[row["id"]]
        content = content_by_id[row["id"]]
        digest = fingerprint(row)
        if (source.get("decision") != "accept" or not source.get("reviewer_id") or
                source.get("reviewer_id") == author or not source.get("reason") or
                source.get("final_row_sha256") != digest or
                source.get("source_verified") is not True or source.get("source_hash_verified") is not True or
                source.get("license_reviewed") is not True or not source.get("license") or
                source.get("safety_reviewed") is not True or
                source.get("safety_decision") not in ("clear", "bounded") or
                (source.get("safety_decision") == "bounded" and not source.get("safety_scope"))):
            raise ValueError("Missing or stale source/license/safety review: " + row["id"])
        if (content.get("decision") != "accept" or not content.get("reviewer_id") or
                content.get("reviewer_id") == author or not content.get("reason") or
                content.get("final_row_sha256") != digest):
            raise ValueError("Missing or stale content review: " + row["id"])
        verify_source_bytes(row, row_splits[row["id"]])

    tokenizer = AutoTokenizer.from_pretrained(c["model"], local_files_only=True)
    train_rows = splits.get("train", []) + splits.get("general_train", [])
    lengths, target_lengths = [], []
    for row in train_rows:
        # Mask checks alone cannot detect a duplicated target in earlier chat turns.
        # Validate the actual training context with the same policy as generation.
        training_context = evaluation_messages(c, row)
        if training_context[-1]['role'] != 'user':
            raise ValueError('Training input does not end at the user question: ' + row['id'])
        encoded = encode_training(c, tokenizer, row)
        target = [token for token in encoded["labels"] if token != -100]
        if tokenizer.decode(target, skip_special_tokens=True) != row["reference"]:
            raise ValueError("Final assistant supervision mismatch: " + row["id"])
        lengths.append(len(encoded["input_ids"]))
        target_lengths.append(len(target))

    heldout_count = 0
    heldout_input_lengths = []
    for split_name in ("dev", "test", "general_dev", "general_test"):
        for row in splits.get(split_name, []):
            prompt = evaluation_messages(c, row)
            expected_context = row["messages"][:-1] if row["messages"][-1]["role"] == "assistant" else row["messages"]
            if prompt[1:] != expected_context or prompt[-1]["role"] != "user":
                raise ValueError("Held-out answer leaked into model input: " + row["id"])
            rendered = tokenizer.apply_chat_template(prompt, tokenize=True,
                add_generation_prompt=True, enable_thinking=False)
            # BatchEncoding.__len__ counts keys rather than encoded tokens.
            heldout_input_lengths.append(input_token_count(rendered))
            heldout_count += 1

    def percentile(values: list[int], q: float) -> int | None:
        if not values:
            return None
        ordered = sorted(values)
        return ordered[min(len(ordered) - 1, max(0, int((len(ordered) - 1) * q + 0.5)))]

    categories = Counter(row.get("category_label", row.get("category", "unknown")) for row in rows)
    result = {
        "status": "工程检查通过；此报告不代表训练批准或模型验收",
        "dataset_release_status": manifest.get("release_status"),
        "model_inference_run": False,
        "training_run": False,
        "manifest_sha256": sha256(manifest_path),
        "split_counts": {name: len(values) for name, values in splits.items()},
        "total_records": len(rows),
        "scenario_record_counts": dict(sorted(categories.items())),
        "training_multiturn_records": sum(len(row.get("messages", [])) > 2 for row in train_rows),
        "training_multiturn_fraction": (sum(len(row.get("messages", [])) > 2 for row in train_rows) / len(train_rows)) if train_rows else None,
        "training_unique_cases": len({row.get("case_id", row.get("group")) for row in train_rows}),
        "training_unique_groups": len({row.get("group") for row in train_rows}),
        "native_tokenizer": "Qwen3.5-2B local tokenizer",
        "training_input_tokens_max": max(lengths) if lengths else None,
        "training_input_tokens_mean": (sum(lengths) / len(lengths)) if lengths else None,
        "training_input_tokens_p50": percentile(lengths, 0.50),
        "training_input_tokens_p95": percentile(lengths, 0.95),
        "training_target_tokens_min": min(target_lengths) if target_lengths else None,
        "training_target_tokens_p50": percentile(target_lengths, 0.50),
        "training_target_tokens_p95": percentile(target_lengths, 0.95),
        "training_final_assistant_targets_match": len(target_lengths),
        "training_contexts_checked_without_target_assistant": len(train_rows),
        "heldout_inputs_checked_without_reference_answer": heldout_count,
        "heldout_input_tokens_max": max(heldout_input_lengths) if heldout_input_lengths else None,
        "structural_issues": 0,
        "source_review_rows": len(source_rows),
        "content_review_rows": len(content_rows),
        "limitations": manifest.get("release_limitations", []),
    }
    write_json(report_path, result)
    return result


if __name__ == "__main__":
    print(json.dumps(audit(), ensure_ascii=False, indent=2))
