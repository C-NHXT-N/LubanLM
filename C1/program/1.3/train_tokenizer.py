from __future__ import annotations

import argparse
import hashlib
import json
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import tokenizers
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from tokenizer_utils import SPECIAL_TOKENS, read_jsonl


def main() -> None:
    """解析命令行参数，训练字节级 BPE，并写出可重载的交付目录。"""
    parser = argparse.ArgumentParser(description="训练 LubanLM 字节级 BPE Tokenizer")
    parser.add_argument("--input", required=True, help="UTF-8 JSONL 训练集")
    parser.add_argument("--vocab-size", type=int, required=True, help="包含特殊 Token 的目标词表大小")
    parser.add_argument("--output", required=True, help="输出目录")
    args = parser.parse_args()
    # 256 个字节基础符号和 3 个固定特殊 Token 已占满最小词表。
    if args.vocab_size < 259:
        parser.error("vocab-size 至少为 259（256 个字节符号和 3 个特殊 Token）")

    # 先完成输入格式验证，再将文本生成器交给训练器。
    records = read_jsonl(args.input)
    if not any(record["text"].strip() for record in records):
        raise ValueError("训练集不含可训练文本")

    # ByteLevel 提供 256 个基础字节，因而不使用未知字符 Token 回退。
    tokenizer = Tokenizer(models.BPE(unk_token=None))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = decoders.ByteLevel()
    # initial_alphabet 固定基础字母表，BPE 只在其上学习常见合并片段。
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=2,
        special_tokens=list(SPECIAL_TOKENS),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    # category 不参与训练；length 仅帮助训练器显示准确进度。
    tokenizer.train_from_iterator((record["text"] for record in records), trainer=trainer, length=len(records))

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer_path = output / "tokenizer.json"
    # tokenizer.json 同时保存模型、预分词器、解码器和 added tokens。
    tokenizer.save(str(tokenizer_path), pretty=True)
    vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
    special_ids = {token: tokenizer.token_to_id(token) for token in SPECIAL_TOKENS}
    # 为后续 Transformers fast tokenizer 加载补充本课程采用的边界符约定。
    tokenizer_config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "clean_up_tokenization_spaces": False,
        "add_bos_token": False,
        "add_eos_token": False,
        "pad_token": "<pad>",
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "vocab_size": vocab_size,
    }
    (output / "tokenizer_config.json").write_text(json.dumps(tokenizer_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # 单独写 special_tokens_map，便于通用工具读取特殊 Token 名称。
    special_map = {key: value for key, value in tokenizer_config.items() if key in ("pad_token", "bos_token", "eos_token")}
    (output / "special_tokens_map.json").write_text(json.dumps(special_map, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    source = Path(args.input)
    # 哈希记录这次训练确切使用的输入文件，后续可检测文件是否变动。
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    categories = Counter(record["category"] for record in records)
    # 记录训练设置和实际结果；实际词表可能因 min_frequency 而小于目标。
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": source.name,
        "input_sha256": digest,
        "record_count": len(records),
        "record_count_by_category": dict(sorted(categories.items())),
        "target_vocab_size": args.vocab_size,
        "actual_vocab_size": vocab_size,
        "special_token_ids": special_ids,
        "bytelevel_initial_alphabet_size": len(pre_tokenizers.ByteLevel.alphabet()),
        "min_frequency": 2,
        "tokenizers_version": tokenizers.__version__,
        "python_version": platform.python_version(),
        "normalizer": None,
        "pre_tokenizer": "ByteLevel(add_prefix_space=False, use_regex=True)",
    }
    (output / "training_config.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    # 不用虚构 Token 填满词表，明确提示语料不足的情况。
    if vocab_size < args.vocab_size:
        print(f"提示：语料不足以达到目标词表；实际 {vocab_size:,} / 目标 {args.vocab_size:,}。")


if __name__ == "__main__":
    main()
