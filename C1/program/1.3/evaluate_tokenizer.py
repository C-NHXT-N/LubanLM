from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from tokenizer_utils import SPECIAL_TOKENS, load_tokenizer, read_jsonl


def main() -> None:
    """在独立评估集上检查往返还原和不同类别的切分长度。"""
    parser = argparse.ArgumentParser(description="按语言类别评估 LubanLM Tokenizer")
    parser.add_argument("--tokenizer", required=True, help="含 tokenizer.json 的模型目录")
    parser.add_argument("--input", required=True, help="独立 UTF-8 JSONL 评估集")
    parser.add_argument("--report-dir", required=True, help="报告输出目录")
    parser.add_argument("--hidden-size", type=int, default=1024, help="Embedding 隐藏维度")
    args = parser.parse_args()
    if args.hidden_size <= 0:
        parser.error("hidden-size 必须为正整数")

    tokenizer = load_tokenizer(args.tokenizer)
    training_config_path = Path(args.tokenizer) / "training_config.json"
    # 旧产物可能没有训练记录；报告仍可完成，但会标注目标词表未记录。
    training_config = json.loads(training_config_path.read_text(encoding="utf-8")) if training_config_path.is_file() else {}
    records = read_jsonl(args.input)
    vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
    special_ids = {token: tokenizer.token_to_id(token) for token in SPECIAL_TOKENS}
    if any(token_id is None for token_id in special_ids.values()):
        raise ValueError(f"词表缺少特殊 Token：{special_ids}")
    if len(set(special_ids.values())) != len(SPECIAL_TOKENS):
        raise ValueError(f"特殊 Token ID 冲突：{special_ids}")

    # defaultdict 保证每个首次出现的类别都从同一组零计数开始。
    metrics: dict[str, dict[str, int | float | bool]] = defaultdict(lambda: {
        "records": 0,
        "characters": 0,
        "utf8_bytes": 0,
        "tokens": 0,
        "roundtrip_failures": 0,
    })
    special_id_set = set(special_ids.values())
    for record in records:
        text = record["text"]
        category = record["category"]
        # 评估原文切分，不额外添加 BOS/EOS，避免边界符影响长度统计。
        encoding = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(encoding.ids, skip_special_tokens=False)
        row = metrics[category]
        row["records"] = int(row["records"]) + 1
        row["characters"] = int(row["characters"]) + len(text)
        row["utf8_bytes"] = int(row["utf8_bytes"]) + len(text.encode("utf-8"))
        row["tokens"] = int(row["tokens"]) + len(encoding.ids)
        # 任何空白、换行或 Unicode 字符变化都计为往返失败。
        row["roundtrip_failures"] = int(row["roundtrip_failures"]) + (decoded != text)
        if special_id_set.intersection(encoding.ids):
            # 语料中出现了特殊 Token 的可见拼写，需在正式数据审查中处理。
            row["contains_special_token_spelling"] = True

    # 指标按类别总量相除，避免先对每条短文本取平均造成偏差。
    for row in metrics.values():
        tokens = int(row["tokens"])
        row["characters_per_token"] = round(int(row["characters"]) / tokens, 4) if tokens else 0.0
        row["utf8_bytes_per_token"] = round(int(row["utf8_bytes"]) / tokens, 4) if tokens else 0.0
        row["roundtrip_ok"] = int(row["roundtrip_failures"]) == 0
    # JSON 供程序读取；下方 Markdown 表供读者直接查看。
    report = {
        "tokenizer_dir": Path(args.tokenizer).name,
        "evaluation_file": Path(args.input).name,
        "hidden_size": args.hidden_size,
        "target_vocab_size": training_config.get("target_vocab_size"),
        "vocab_size_including_special_tokens": vocab_size,
        "embedding_parameters": vocab_size * args.hidden_size,
        "special_token_ids": special_ids,
        "by_category": dict(sorted(metrics.items())),
    }

    output = Path(args.report_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Tokenizer 评估报告", "",
        f"- 目标词表大小：{training_config['target_vocab_size']:,}" if training_config.get("target_vocab_size") else "- 目标词表大小：未记录",
        f"- 实际词表大小：{vocab_size:,}",
        f"- Embedding 参数量（hidden_size={args.hidden_size}）：{vocab_size * args.hidden_size:,}",
        f"- 特殊 Token ID：`{json.dumps(special_ids, ensure_ascii=False)}`", "",
        "|类别|样本数|字符数|UTF-8 字节数|Token 总数|字符数 / Token|UTF-8 字节 / Token|往返还原|",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for category, row in sorted(metrics.items()):
        lines.append(f"|{category}|{row['records']}|{row['characters']}|{row['utf8_bytes']}|{row['tokens']}|{row['characters_per_token']}|{row['utf8_bytes_per_token']}|{'通过' if row['roundtrip_ok'] else '失败'}|")
    (output / "metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    # 报告仍会落盘，随后以非零退出码让自动化流程感知失败。
    if any(int(row["roundtrip_failures"]) for row in metrics.values()):
        raise SystemExit("评估失败：存在无法往返还原的文本。")


if __name__ == "__main__":
    main()
