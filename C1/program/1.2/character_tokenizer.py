"""教学用字符级 Tokenizer：只依赖 Python 标准库。"""

from __future__ import annotations

import json
from pathlib import Path


class CharacterTokenizer:
    """将训练语料中出现过的每个字符映射为一个固定 ID。"""

    def __init__(self, vocabulary: list[str]):
        # vocabulary 的下标就是该字符的 Token ID。
        self.vocabulary = vocabulary
        self.token_to_id = {token: index for index, token in enumerate(vocabulary)}

    @classmethod
    def train(cls, texts: list[str]) -> "CharacterTokenizer":
        """收集全部字符并排序，使相同语料总能生成相同 ID 顺序。"""
        if not texts or not any(texts):
            raise ValueError("训练语料不能为空")
        return cls(sorted(set("".join(texts))))

    def encode(self, text: str) -> list[int]:
        """逐字符查表；不认识的字符直接报错，避免静默丢失信息。"""
        missing = sorted(set(text) - self.token_to_id.keys())
        if missing:
            raise ValueError(f"训练集中未出现字符：{missing!r}")
        return [self.token_to_id[char] for char in text]

    def decode(self, ids: list[int]) -> str:
        """按 ID 反查字符并拼回原文。"""
        # 先整体检查范围，使负数索引不会被 Python 当作从末尾取值。
        if any(index < 0 or index >= len(self.vocabulary) for index in ids):
            raise ValueError("Token ID 超出词表范围")
        try:
            return "".join(self.vocabulary[index] for index in ids)
        except IndexError as exc:
            raise ValueError("Token ID 超出词表范围") from exc

    def save(self, path: str | Path) -> None:
        """保存词表；教学格式显式记录 type，便于加载时区分格式。"""
        Path(path).write_text(
            json.dumps({"type": "character", "vocabulary": self.vocabulary}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "CharacterTokenizer":
        """读取并验证由 save 写出的字符级 Tokenizer 文件。"""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("type") != "character" or not isinstance(data.get("vocabulary"), list):
            raise ValueError("无效的字符 Tokenizer 文件")
        return cls(data["vocabulary"])
