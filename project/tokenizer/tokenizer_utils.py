from __future__ import annotations

import json
from pathlib import Path

from tokenizers import Tokenizer

# 顺序会决定训练器分配的特殊 Token ID，训练后不得随意改动。
SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>")
# category 只为评估报告分组，不会传给 BPE 训练器。
CATEGORIES = ("中文", "英文", "代码")


def read_jsonl(path: str | Path) -> list[dict[str, str]]:
    """读取并验证课程约定的 JSONL：每条记录只有 text 和 category 两个必需字段。"""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"找不到数据文件：{source}")
    records: list[dict[str, str]] = []
    # utf-8-sig 同时兼容普通 UTF-8 与带 BOM 的 Windows 文本。
    with source.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            # 空白行不算记录，允许数据文件中留下分隔空行。
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{line_number}: JSON 格式错误：{exc.msg}") from exc
            if not isinstance(item, dict) or not isinstance(item.get("text"), str):
                raise ValueError(f"{source}:{line_number}: 每条记录都必须包含字符串字段 text")
            if item.get("category") not in CATEGORIES:
                raise ValueError(f"{source}:{line_number}: category 必须是 {', '.join(CATEGORIES)} 之一")
            # 丢弃不参与本章流程的额外字段，保持下游输入稳定。
            records.append({"text": item["text"], "category": item["category"]})
    if not records:
        raise ValueError(f"{source}: 数据集为空")
    return records


def load_tokenizer(directory: str | Path) -> Tokenizer:
    """从交付目录加载 tokenizer.json，不猜测或重建训练规则。"""
    path = Path(directory) / "tokenizer.json"
    if not path.is_file():
        raise FileNotFoundError(f"Tokenizer 文件不存在：{path}")
    return Tokenizer.from_file(str(path))


def encode_text(tokenizer: Tokenizer, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
    """编码正文，并按需在两端各补一次边界符。"""
    # 正文默认不自动添加边界符；样本格式需要时由调用者明确选择。
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    if add_bos:
        if bos_id is None:
            raise ValueError("Tokenizer 未定义 <bos>")
        # 已经以 BOS 开头时保持原序列，避免重复添加。
        if not ids or ids[0] != bos_id:
            ids.insert(0, bos_id)
    if add_eos:
        if eos_id is None:
            raise ValueError("Tokenizer 未定义 <eos>")
        # EOS 与 PAD 语义不同，结束样本时只能追加 EOS。
        if not ids or ids[-1] != eos_id:
            ids.append(eos_id)
    return ids
