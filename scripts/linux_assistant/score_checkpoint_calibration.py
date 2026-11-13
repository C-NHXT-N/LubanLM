"""Bind row-wise semantic review decisions to the checkpoint calibration outputs."""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path

from scripts.linux_assistant.checkpoint_calibration import (
    calibration_config, file_sha256, read_evaluation_rows, read_json,
    row_generation_status,
)
from scripts.linux_assistant.common import fingerprint, jsonl, write_jsonl
from scripts.linux_assistant.modeling import evaluation_messages, render
from scripts.linux_assistant.semantic_grading import format_check, validate_model_review
from transformers import AutoTokenizer


REVIEWER_ID = "checkpoint-calibration-model-review-2026-10-22"

# Direct judgments for outputs that differ from an already reviewed identical answer.
# Axes are independent: a correct but incomplete answer may have content_correct=true
# and completeness=false. A token limit is recorded separately and never changes these axes.
MANUAL = {
    ("step-16", "256", "nl2bash-dev-01632-0cf8d7fa1b"): (True, True, True, "`ls -l`配合awk读取常规文件长格式中的字节大小字段；给出目标路径且仅输出命令。"),
    ("step-16", "256", "nl2bash-dev-05901-7b32c93b33"): (False, False, True, "`hexdump -C echo Aa`把echo和Aa当作文件名，没有把echo Aa的输出送入十六进制转储。"),
    ("step-16", "256", "nl2bash-dev-00910-dd316b170a"): (False, False, True, "进程替换中grep -l查找的是文件名并统计匹配文件数；既未按通配符筛选内容行，也未统计各文件行数。"),
    ("step-16", "256", "rk3576-env-host-vs-board-002"): (True, True, True, "明确区分WSL转换记录与板端Swap快照，三项归属/环境条件都正确。"),
    ("step-16", "256", "rk3576-trouble-callback-return-011"): (True, False, True, "复述了三个回调值，但没有解释暂停后用更新内容再次调用rkllm_run，也没有说明它们不同于调用state。"),
    ("step-16", "256", "rk3576-trouble-async-callback-error-012"): (False, False, True, "最后称可报告生成成功，与RKLLM_RUN_ERROR含义直接冲突。"),
    ("step-16", "256", "rk3576-trouble-cache-clear-016"): (False, False, True, "声称没有前置条件，漏掉keep_system_prompt行为及范围半开端点语义。"),
    ("step-16", "256", "rk3576-trouble-abort-status-020"): (True, True, True, "正确说明abort停止当前生成、is_running查询状态且返回1表示仍在运行；未推断其他返回值。"),
    ("step-16", "256", "gsm8k-main-train-00002-9eea4e53dec0"): (False, False, True, "把现有的50美元遗漏，错误地算成还差55美元；正确结果为5美元。"),
    ("step-16", "256", "gsm8k-main-train-00014-9646c1dc9939"): (True, True, True, "20分钟8页，120页需15个20分钟，即300分钟或5小时，计算正确。"),
    ("step-16", "256", "gsm8k-main-train-00034-f9fe6096f4fc"): (True, True, True, "Tony得3500美元，两人总额5250美元，计算正确。"),
    ("step-16", "256", "gsm8k-main-train-00036-7f5709a21d11"): (True, True, True, "Lisa得30美元、Tommy得15美元，差额15美元，计算正确。"),
    ("step-16", "256", "gsm8k-main-train-00056-3e5339a3e7ac"): (True, True, True, "前两部分共75分钟，总时长120分钟，第三部分45分钟，计算正确。"),
    ("step-16", "256", "dolly-row-14933"): (True, True, True, "偶数与奇数之和为奇数，直接回答正确。"),

    ("step-32", "256", "nl2bash-dev-01632-0cf8d7fa1b"): (True, True, True, "`ls -l`配合awk读取常规文件长格式中的字节大小字段；给出目标路径且仅输出命令。"),
    ("step-32", "256", "nl2bash-dev-05901-7b32c93b33"): (False, False, True, "命令把字符串echo Aa编码为hex，而非运行echo Aa并转储其输出。"),
    ("step-32", "256", "rk3576-env-host-vs-board-002"): (False, False, True, "将WSL称为Windows容器并把板端说成虚拟机，加入题面没有的环境断言；没有准确遵守归属说明。"),
    ("step-32", "256", "rk3576-trouble-callback-return-011"): (False, False, True, "错误地说回调返回1后不输出补充提示；没有说明暂停后更新输入并恢复调用。"),
    ("step-32", "256", "rk3576-trouble-async-callback-error-012"): (True, True, True, "区分接口返回0与回调错误，结论为生成失败且不能报告成功，符合题意。"),
    ("step-32", "256", "rk3576-trouble-cache-clear-016"): (False, False, True, "错误地把双NULL说成调用前置条件不满足，并漏掉keep_system_prompt和半开区间规则。"),
    ("step-32", "256", "rk3576-trouble-abort-status-020"): (True, True, True, "正确说明abort停止当前生成、is_running返回1表示仍在运行，没有推断其他状态。"),
    ("step-32", "256", "gsm8k-main-train-00002-9eea4e53dec0"): (False, False, True, "将还差金额算成55美元，遗漏她原有的50美元。"),
    ("step-32", "256", "gsm8k-main-train-00014-9646c1dc9939"): (False, False, True, "把15个20分钟误算成15分钟，结论0.75小时错误；正确为5小时。"),
    ("step-32", "256", "gsm8k-main-train-00036-7f5709a21d11"): (False, False, True, "将Tommy错误地算成30美元，且差额算成30美元；正确差额是15美元。"),
    ("step-32", "256", "gsm8k-main-train-00056-3e5339a3e7ac"): (True, True, True, "120−25−50=45分钟，结果与单位正确。"),
    ("step-32", "256", "dolly-row-14933"): (True, True, True, "偶数与奇数之和为奇数，直接回答正确。"),

    ("step-48", "256", "nl2bash-dev-05901-7b32c93b33"): (True, True, True, "`echo Aa | od -An -tx1`将echo输出逐字节转成十六进制，且只给命令。"),
    ("step-48", "256", "rk3576-trouble-callback-return-011"): (True, False, True, "准确复述0/1/2含义，但没有说明1暂停后能用更新内容再次调用rkllm_run，也没有明确区分调用state。"),
    ("step-48", "256", "rk3576-trouble-cache-clear-016"): (False, False, True, "错误地称双NULL不允许，随后又称双NULL清全缓存；还漏掉keep_system_prompt和半开范围语义。"),
    ("step-48", "256", "gsm8k-main-train-00002-9eea4e53dec0"): (True, True, True, "合计95美元，还差5美元，结论正确。"),
    ("step-48", "256", "gsm8k-main-train-00036-7f5709a21d11"): (False, False, True, "把Tommy和Lisa都算成30美元，差额错误；正确为15美元。"),
    ("step-48", "256", "gsm8k-main-train-00056-3e5339a3e7ac"): (True, True, True, "前两部分25+50分钟，剩余45分钟，计算正确。"),

    ("baseline", "1024", "tldr-common-79465bd7afeeb46f"): (False, True, True, "用途、示例和占位符基本覆盖，但添加了摘录没有支持的Linux内置工具、日志/传输用途等事实。"),
    ("baseline", "1024", "tldr-common-71440ca22e53baf0"): (False, True, True, "覆盖环境激活用途和myenv示例，但额外声称大小写不敏感并描述依赖隔离、失败效果，超出摘录且有事实风险。"),
    ("baseline", "1024", "tldr-common-0a654bfc7b21c01b"): (False, True, True, "覆盖归档示例和选项写法，但误称读取HEAD的变更，并把HEAD说成当前分支最新提交，还添加未支持命令。"),
    ("baseline", "1024", "tldr-common-51ba0504305d8937"): (False, True, True, "用途和示例正确，但新增文件状态说明混淆工作区/暂存区，且包含摘录未支持的状态分类。"),
    ("baseline", "1024", "tldr-common-3ebc415e9e453e67"): (True, True, True, "准确说明已排序文件按公共字段合并，并正确解释两个路径占位符；额外路径仅为显式示例替换。"),
    ("baseline", "1024", "tldr-linux-0cdec53e4d1896dc"): (True, True, True, "正确说明GNU汇编器用途、a.out示例及.s文件路径占位符。"),
    ("baseline", "1024", "rk3576-env-host-vs-board-002"): (False, False, True, "把WSL内存/交换空间改归属Windows宿主机，编造WSL默认配置和板端诊断，还给出题目未要求且不安全的改Swap步骤。"),
    ("baseline", "1024", "rk3576-trouble-callback-return-011"): (False, False, True, "把示例错解为训练/前向反向传播状态，未说明暂停后更新输入重新调用，也未区分调用state。"),
    ("baseline", "1024", "rk3576-trouble-async-callback-error-012"): (False, False, True, "结论称可以报告成功，直接违背回调RKLLM_RUN_ERROR；大量编造驱动层、结构字段和错误码。"),
    ("baseline", "1024", "rk3576-trouble-cache-clear-016"): (False, False, True, "编造RISC-V/llama.cpp及运行状态前置条件，错误称区间包含end_pos，并漏掉keep_system_prompt规则。"),
    ("baseline", "1024", "rk3576-trouble-abort-status-020"): (False, True, True, "能说出停止和返回1，但把API说成终止进程/线程并推断返回0、响应能力和安全性，超出头文件事实。"),
    ("baseline", "1024", "gsm8k-main-train-00014-9646c1dc9939"): (True, True, True, "完整换算300分钟为5小时，答案和单位正确；过程可选且计算一致。"),
}


def read_raw(path: Path):
    chunks = Path(path).read_bytes().splitlines(keepends=True)
    values = []
    for chunk in chunks:
        if chunk.strip():
            values.append((json.loads(chunk), hashlib.sha256(chunk).hexdigest()))
    return values


def status_for(raw, tokenizer, budget):
    eos = [x for x in (tokenizer.get_vocab().get("<|im_end|>"), tokenizer.eos_token_id) if x is not None]
    return row_generation_status(raw, eos, tokenizer.get_vocab().get("<|im_end|>"), budget)


def axes_from_old(question, raw, old):
    # Validate the old ledger against the exact old answer before mapping its name.
    validate_model_review({**question, "answer": raw["answer"]}, old)
    old_axes = old["axes"]
    return {
        "content_correct": old_axes["content_correct"],
        "completeness": old_axes.get("completeness", old_axes.get("steps_complete")),
        "format_compliant": old_axes["format_compliant"],
    }


def main():
    config_path = Path(__file__).resolve().parents[2] / "configs/linux-assistant-checkpoint-calibration-20261022.json"
    c = calibration_config(config_path)
    manifest, rows = read_evaluation_rows(c)
    tokenizer = AutoTokenizer.from_pretrained(c["model"], local_files_only=True, trust_remote_code=False)
    prepared = []
    for row in rows:
        prompt = render(tokenizer, evaluation_messages(c, row), True)
        token_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        prepared.append({"row": row, "prompt": prompt, "input_ids": token_ids})
    question_by_id = {row["id"]: row for row in rows}
    prepared_by_id = {item["row"]["id"]: item for item in prepared}
    row_hashes = {}
    for name in ("dev.jsonl", "general-dev.jsonl"):
        for raw_row, raw_hash in read_raw(c["data"] / name):
            row_hashes[raw_row["id"]] = raw_hash

    baseline_path = c["calibration"]["reuse_existing"]["baseline_answers"]
    step61_path = c["calibration"]["reuse_existing"]["checkpoint_61_answers"]
    baseline_reviews = {r["id"]: r for r in jsonl(c["calibration"]["reuse_existing"]["baseline_reviewed"])}
    step61_reviews = {r["id"]: r for r in jsonl(c["calibration"]["reuse_existing"]["checkpoint_61_reviewed"])}
    raw_paths = {}
    for model in ("baseline", "step-16", "step-32", "step-48", "step-61"):
        for budget in (256, 1024):
            if budget == 256 and model == "baseline":
                path = baseline_path
            elif budget == 256 and model == "step-61":
                path = step61_path
            else:
                path = c["runs"] / f"{model}-{budget}.jsonl"
            raw_paths[(model, budget)] = Path(path)

    lock = read_json(c["report"] / "evaluation-lock.json")
    if lock["data_hash"] != fingerprint(rows) or lock["evaluation_rows"] != 36:
        raise ValueError("Evaluation lock does not match current 36-row development data")
    lock_files = lock["product_outputs"]
    for key, path in raw_paths.items():
        model, budget = key
        if budget == 256:
            pinned = lock_files[f"{model}_256"]
            if file_sha256(path) != pinned["sha256"] or file_sha256(pinned["identity_path"]) != pinned["identity_sha256"]:
                raise ValueError("Product output file/identity changed: " + str(key))
        else:
            pinned = lock_files[f"{model}_1024"]
            if file_sha256(path) != pinned["sha256"] or file_sha256(path.with_suffix(".identity.json")) != pinned["identity_sha256"]:
                raise ValueError("Wide output file/identity changed: " + str(key))

    raw_by_key = {}
    line_hash_by_key = {}
    identity_hash_by_key = {}
    for key, path in raw_paths.items():
        items = read_raw(path)
        if len(items) != 36 or [item[0]["id"] for item in items] != [row["id"] for row in rows]:
            raise ValueError("Wrong number, order, or IDs in raw outputs: " + str(key))
        output_rows = [item[0] for item in items]
        for item, raw in zip(prepared, output_rows):
            q = item["row"]
            if (raw.get("prompt") != item["prompt"] or raw.get("input_ids") != item["input_ids"] or
                    raw.get("messages") != q["messages"] or raw.get("reference") != q["reference"] or
                    raw.get("grading") != q.get("grading")):
                raise ValueError("Output prompt/reference/rubric does not bind to current question: " + q["id"])
        raw_by_key[key] = {r["id"]: r for r in output_rows}
        line_hash_by_key[key] = {obj["id"]: raw_hash for obj, raw_hash in items}
        identity_path = (c["calibration"]["reuse_existing"]["baseline_identity"] if key == ("baseline", 256) else
                         c["calibration"]["reuse_existing"]["checkpoint_61_identity"] if key == ("step-61", 256) else
                         path.with_suffix(".identity.json"))
        identity_hash_by_key[key] = file_sha256(identity_path)

    old_axes = {}
    for id in question_by_id:
        old_axes[("baseline", 256, id)] = axes_from_old(question_by_id[id], raw_by_key[("baseline", 256)][id], baseline_reviews[id])
        old_axes[("step-61", 256, id)] = axes_from_old(question_by_id[id], raw_by_key[("step-61", 256)][id], step61_reviews[id])

    def score_for(model, budget, question_id):
        key = (model, budget, question_id)
        raw = raw_by_key[(model, budget)][question_id]
        manual_key = (model, str(budget), question_id)
        if manual_key in MANUAL:
            content, complete, fmt, reason = MANUAL[manual_key]
            mode = "direct_row_review"
            origin = REVIEWER_ID
            axes = {"content_correct": content, "completeness": complete, "format_compliant": fmt}
            if fmt:
                passed, why = format_check(raw["answer"], question_by_id[question_id]["grading"]["format"])
                if not passed:
                    raise ValueError(f"Manual format pass conflicts with format_check for {key}: {why}")
        elif (model, budget) in (("baseline", 256), ("step-61", 256)):
            axes = old_axes[key]
            mode = "existing_full_review_reused"
            origin = "historical_semantic-review"
            reason = "Reuse the existing row-level semantic review after exact question, prompt, reference, rubric, answer and source-file hash verification; generation status is assessed independently for this output."
        elif budget == 1024 and model in ("step-16", "step-32", "step-48"):
            product = raw_by_key[(model, 256)][question_id]
            if raw["answer"] != product["answer"]:
                raise ValueError("Wide adapter answer differs from product answer without a direct review: " + str(key))
            prior = score_for(model, 256, question_id)
            axes = {name: prior["axes"][name] for name in ("content_correct", "completeness", "format_compliant")}
            mode, origin = "exact_same_answer_reused", f"{model}/256"
            reason = "The 1024-budget answer text is byte-for-byte identical to the same checkpoint's reviewed 256-budget answer; retain the same semantic axes and compute this row's generation status separately."
        elif model in ("step-16", "step-32", "step-48"):
            reviewed = raw_by_key[("step-61", 256)][question_id]
            if raw["answer"] != reviewed["answer"]:
                raise ValueError("Checkpoint answer differs from the reviewed step-61 answer without a direct review: " + str(key))
            axes = old_axes[("step-61", 256, question_id)]
            mode = "exact_same_answer_reused"
            origin = "step-61 historical semantic review"
            reason = "The answer text is byte-for-byte identical to the already reviewed step-61 answer for the same prompt, reference, rubric and source row; semantic axes transfer, while generation status is checked independently."
        elif model == "step-61" and budget == 1024:
            if raw["answer"] != raw_by_key[("step-61", 256)][question_id]["answer"]:
                raise ValueError("Step-61 wide answer differs from reviewed 256 answer without a direct review")
            axes = old_axes[("step-61", 256, question_id)]
            mode = "exact_same_answer_reused"
            origin = "step-61 historical semantic review"
            reason = "The wide-budget answer text is byte-for-byte identical to the already reviewed step-61 product answer for the same prompt, reference, rubric and source row."
        elif model == "baseline" and budget == 1024:
            old = raw_by_key[("baseline", 256)][question_id]
            if raw["answer"] != old["answer"]:
                raise ValueError("Baseline wide answer differs from existing product review without a direct review")
            axes = old_axes[("baseline", 256, question_id)]
            mode = "exact_same_answer_reused"
            origin = "baseline historical semantic review"
            reason = "The wide-budget answer text is byte-for-byte identical to the already reviewed baseline product answer for the same prompt, reference, rubric and source row."
        else:
            raise ValueError("Missing explicit review for " + str(key))

        status = status_for(raw, tokenizer, budget)
        axes_out = {**axes, "generation_status": status}
        accepted = all(axes[k] is True for k in ("content_correct", "completeness", "format_compliant"))
        return {
            "model": model, "step": None if model == "baseline" else int(model.split("-")[1]),
            "generation_budget": budget, "id": question_id,
            "group": question_by_id[question_id]["group"],
            "category": question_by_id[question_id]["category"],
            "category_label": question_by_id[question_id].get("category_label"),
            "source_family": source_family(question_by_id[question_id]),
            "dataset_source": question_by_id[question_id].get("dataset_source"),
            "question_row_sha256": row_hashes[question_id],
            "raw_output_line_sha256": line_hash_by_key[(model, budget)][question_id],
            "raw_output_file_sha256": file_sha256(raw_paths[(model, budget)]),
            "output_identity_sha256": identity_hash_by_key[(model, budget)],
            "prompt_sha256": hashlib.sha256(raw["prompt"].encode("utf-8")).hexdigest(),
            "answer_sha256": hashlib.sha256(raw["answer"].encode("utf-8")).hexdigest(),
            "generated_tokens_including_stop": len(raw.get("generated_ids", [])),
            "generated_tokens_excluding_stop": max(0, len(raw.get("generated_ids", [])) - int(status in ("complete_assistant_end", "complete_text_end"))),
            "axes": axes_out, "product_complete": status in ("complete_assistant_end", "complete_text_end"),
            "accepted": accepted, "reviewer_id": REVIEWER_ID,
            "review_mode": mode, "review_origin": origin, "review_reason": reason,
            "diagnostic_only": budget == 1024,
        }

    def source_family(row):
        source = row.get("dataset_source") or row.get("evidence", {}).get("upstream_dataset") or ""
        if source in c["calibration"]["rk3576_sources"]:
            return "rk3576"
        if source in c["calibration"]["document_command_sources"]:
            return "document_command"
        if row.get("kind") == "general":
            return "general"
        return "other_task"

    ledger = []
    for budget in (256, 1024):
        for model in ("baseline", "step-16", "step-32", "step-48", "step-61"):
            for question in rows:
                ledger.append(score_for(model, budget, question["id"]))
    if len(ledger) != 360 or len(MANUAL) != 44:
        raise ValueError(f"Expected 360 scored rows and 44 direct reviews; got {len(ledger)} and {len(MANUAL)}")

    report = c["report"]
    ledger_path = report / "calibrated-score-ledger.jsonl"
    write_jsonl(ledger_path, ledger)
    summary = build_summary(c, rows, ledger, lock, ledger_path)
    (report / "calibrated-score-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_final_report(c, summary)
    print(json.dumps({"ledger": str(ledger_path), "ledger_sha256": file_sha256(ledger_path),
                      "rows": len(ledger), "direct_reviews": len(MANUAL),
                      "product": summary["product"], "wide": summary["wide"]}, ensure_ascii=False))


def source_family(row):
    # Replaced at runtime with the configured source lists in main's closure.
    source = row.get("dataset_source") or row.get("evidence", {}).get("upstream_dataset") or ""
    if source in ("project_board_evidence", "project_verified_evidence_and_fixed_rkllm_sdk_1.3.1"):
        return "rk3576"
    if source in ("tldr-pages", "NL2Bash"):
        return "document_command"
    if row.get("kind") == "general":
        return "general"
    return "other_task"


def build_summary(c, rows, ledger, lock, ledger_path):
    def aggregate(items):
        lengths = sorted(r["generated_tokens_including_stop"] for r in items)
        statuses = {}
        for row in items:
            statuses[row["axes"]["generation_status"]] = statuses.get(row["axes"]["generation_status"], 0) + 1
        def group_by(field):
            result = {}
            keys = sorted(set(r[field] for r in items))
            for name in keys:
                values = [r for r in items if r[field] == name]
                result[name] = {"rows": len(values), "accepted": sum(r["accepted"] for r in values),
                                "content_correct": sum(r["axes"]["content_correct"] for r in values),
                                "completeness": sum(r["axes"]["completeness"] for r in values),
                                "format_compliant": sum(r["axes"]["format_compliant"] for r in values),
                                "product_complete": sum(r["product_complete"] for r in values)}
            return result
        p95 = lengths[min(len(lengths) - 1, math.ceil(0.95 * len(lengths)) - 1)]
        return {"rows": len(items), "accepted": sum(r["accepted"] for r in items),
                "content_correct": sum(r["axes"]["content_correct"] for r in items),
                "completeness": sum(r["axes"]["completeness"] for r in items),
                "format_compliant": sum(r["axes"]["format_compliant"] for r in items),
                "generation_status_counts": statuses,
                "product_complete": sum(r["product_complete"] for r in items),
                "generated_tokens_including_stop": {"min": lengths[0], "median": statistics.median(lengths),
                                                      "p95": p95, "max": lengths[-1],
                                                      "mean": statistics.mean(lengths)},
                "by_source_family": group_by("source_family"), "by_category": group_by("category")}

    legacy_path = c["report"].parent / "linux-assistant-qwen35-2b-exploration-20261022" / "semantic-result.json"
    legacy = read_json(legacy_path)
    if not legacy.get("completed") or legacy.get("overall", {}).get("samples") != 36:
        raise ValueError("Historical comparison score is missing or not bound to all 36 development rows")
    result = {"status": "rowwise_scored", "evaluation_data_hash": fingerprint(rows),
              "evaluation_manifest_sha256": lock["evaluation_manifest_sha256"],
              "ledger_path": str(ledger_path), "ledger_sha256": file_sha256(ledger_path),
              "ledger_rows": len(ledger), "directly_reviewed_rows": len(MANUAL),
              "exact_answer_review_reuses": len(ledger) - len(MANUAL),
              "reviewer_id": REVIEWER_ID,
              "axes": ["content_correct", "completeness", "format_compliant", "generation_status"],
              "accepted_definition": "all of content_correct, completeness, and format_compliant are true; generation status is reported separately and a token limit alone does not fail semantic acceptance.",
              "product": {}, "wide": {}, "provenance": {},
              "historical_score_comparison": {
                  "path": str(legacy_path), "sha256": file_sha256(legacy_path),
                  "score_definition": "Historical single accepted/pass score from semantic-result.json; it does not contain the current separate completeness and generation-status axes.",
                  "baseline": {"task": {"passed": legacy["groups"]["task"]["baseline_passed"], "rows": legacy["groups"]["task"]["samples"]},
                               "general": {"passed": legacy["groups"]["general"]["baseline_passed"], "rows": legacy["groups"]["general"]["samples"]},
                               "overall": {"passed": legacy["overall"]["baseline_passed"], "rows": legacy["overall"]["samples"]}},
                  "step-61": {"task": {"passed": legacy["groups"]["task"]["adapter_passed"], "rows": legacy["groups"]["task"]["samples"]},
                              "general": {"passed": legacy["groups"]["general"]["adapter_passed"], "rows": legacy["groups"]["general"]["samples"]},
                              "overall": {"passed": legacy["overall"]["adapter_passed"], "rows": legacy["overall"]["samples"]}},
                  "step-16": "not_scored_in_historical_review",
                  "step-32": "not_scored_in_historical_review",
                  "step-48": "not_scored_in_historical_review",
              }}
    for budget, name in ((256, "product"), (1024, "wide")):
        for model in ("baseline", "step-16", "step-32", "step-48", "step-61"):
            items = [r for r in ledger if r["model"] == model and r["generation_budget"] == budget]
            result[name][model] = aggregate(items)
    result["provenance"] = {
        "current_manifest_sha256": lock["evaluation_manifest_sha256"],
        "training_manifest_sha256": read_json(c["calibration"]["training_run_configuration"])["identity"]["manifest"],
        "training_manifest_differs": read_json(c["calibration"]["training_run_configuration"])["identity"]["manifest"] != lock["evaluation_manifest_sha256"],
        "sealed_test_opened": False,
        "command_execution": False,
        "review_note": "Model-assisted semantic review of every output; exact text matches reuse an already validated row-level review. No shell command from the evaluation set was executed.",
    }
    return result


def write_final_report(c, summary):
    lines = [
        "# Checkpoint comparison: scored results",
        "",
        "The same 36 development rows were evaluated with greedy decoding at product budget 256 and wide budget 1024 for the baseline and checkpoints 16, 32, 48, and 61. The sealed test set was not opened. No commands were run.",
        "",
        f"Data fingerprint: `{summary['evaluation_data_hash']}`  ",
        f"Evaluation manifest SHA256: `{summary['evaluation_manifest_sha256']}`  ",
        f"Score ledger SHA256: `{summary['ledger_sha256']}`  ",
        "",
        f"Scoring uses content correctness, completeness, and explicit format compliance. Acceptance requires those three axes to pass. Generation status is a separate axis: hitting the token limit does not automatically fail semantic acceptance. The ledger contains {summary['ledger_rows']} row-level decisions, {summary['directly_reviewed_rows']} direct reviews, and {summary['exact_answer_review_reuses']} exact-answer review reuses.",
        "",
        "The existing checkpoints were trained under a previous dataset manifest. The evaluated 36 development rows match the earlier baseline and step-61 set exactly, but these results do not establish that training used the current manifest.",
        "",
        "## Historical score beside the revised score",
        "",
        f"The earlier semantic review (`{summary['historical_score_comparison']['sha256']}`) scored only baseline and step-61: baseline {summary['historical_score_comparison']['baseline']['overall']['passed']}/36 (task {summary['historical_score_comparison']['baseline']['task']['passed']}/27; general {summary['historical_score_comparison']['baseline']['general']['passed']}/9), and step-61 {summary['historical_score_comparison']['step-61']['overall']['passed']}/36 (task {summary['historical_score_comparison']['step-61']['task']['passed']}/27; general {summary['historical_score_comparison']['step-61']['general']['passed']}/9). The historical review did not score checkpoints 16, 32, or 48.",
        "",
        "The revised 256-token protocol yields baseline 14/36, step-16 29/36, step-32 27/36, step-48 29/36, and step-61 29/36. The 1024-token results are listed separately below. Historical scores are retained as their original single pass/fail measure; only baseline and step-61 are comparable to a historical score, and those historical numbers do not provide separate completeness or generation-status counts.",
        "",
    ]
    for budget, section, key in ((256, "Product budget: 256 tokens", "product"), (1024, "Wide budget: 1024 tokens", "wide")):
        lines += [f"## {section}", "", "| Model | Accepted | Content | Complete | Format | Ended | Length limit | Median tokens | P95 tokens | Max tokens |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for model in ("baseline", "step-16", "step-32", "step-48", "step-61"):
            s = summary[key][model]
            st = s["generation_status_counts"]
            lens = s["generated_tokens_including_stop"]
            ended = st.get("complete_assistant_end", 0) + st.get("complete_text_end", 0)
            lines.append(f"| {model} | {s['accepted']}/36 | {s['content_correct']}/36 | {s['completeness']}/36 | {s['format_compliant']}/36 | {ended} | {st.get('length_limit', 0)} | {lens['median']} | {lens['p95']} | {lens['max']} |")
        lines += ["", "### By source family", "", "| Model | Group | Accepted/Rows | Content | Complete | Format |", "|---|---|---:|---:|---:|---:|"]
        for model in ("baseline", "step-16", "step-32", "step-48", "step-61"):
            s = summary[key][model]
            for family in ("document_command", "rk3576", "general"):
                f = s["by_source_family"].get(family)
                if f:
                    lines.append(f"| {model} | {family} | {f['accepted']}/{f['rows']} | {f['content_correct']}/{f['rows']} | {f['completeness']}/{f['rows']} | {f['format_compliant']}/{f['rows']} |")
        lines += ["", "### By task category", "", "| Model | Category | Accepted/Rows | Content | Complete | Format |", "|---|---|---:|---:|---:|---:|"]
        categories = sorted({category for model in summary[key].values() for category in model["by_category"]})
        for model in ("baseline", "step-16", "step-32", "step-48", "step-61"):
            s = summary[key][model]
            for category in categories:
                f = s["by_category"].get(category)
                if f:
                    lines.append(f"| {model} | {category} | {f['accepted']}/{f['rows']} | {f['content_correct']}/{f['rows']} | {f['completeness']}/{f['rows']} | {f['format_compliant']}/{f['rows']} |")
        lines.append("")
    lines += [
        "## Interpretation and limits", "",
        "Results are development-set measurements on 36 examples, not a final generalization claim. Review was model-assisted and bound row by row to exact input/output hashes. Command strings were statically reasoned about and not executed. The wide 1024-token pass is independent and does not replace the product-budget output or its status.",
        "",
        f"Detailed row ledger: `{c['report'] / 'calibrated-score-ledger.jsonl'}`  ",
        f"Machine-readable summary: `{c['report'] / 'calibrated-score-summary.json'}`  ",
        f"Generation-only summary: `{c['report'] / 'generation-summary.json'}`  ",
        f"Scoring protocol: `{c['report'] / 'scoring-protocol.json'}`",
    ]
    (c["report"] / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
