"""便于逐步调试的简化 BPE；正式项目请使用 1.3 的字节级实现。"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


class TeachingBPE:
    """保留每轮合并历史的最小 BPE 实现，便于与正文表格逐项对照。"""

    def __init__(self, alphabet: list[str], merges: list[tuple[str, str]], history: list[dict] | None = None):
        self.alphabet = alphabet
        self.merges = merges
        self.history = history or []
        # 初始字符先占据词表前部，随后按合并发生的顺序追加新 Token。
        self.token_to_id = {token: index for index, token in enumerate(alphabet)}
        for left, right in merges:
            token = left + right
            if token not in self.token_to_id:
                self.token_to_id[token] = len(self.token_to_id)

    @classmethod
    def train(cls, texts: list[str], num_merges: int = 20) -> "TeachingBPE":
        """重复统计相邻对并合并最高频 pair，最多执行 num_merges 轮。"""
        if not texts or not any(texts):
            raise ValueError("训练语料不能为空")
        # 训练中的 sequences 会被逐轮替换；原始 texts 保持不变。
        sequences = [list(text) for text in texts]
        alphabet = sorted(set(char for sequence in sequences for char in sequence))
        merges: list[tuple[str, str]] = []
        history: list[dict] = []
        for _ in range(num_merges):
            # 每一轮都从“当前 Token 序列”重新计数，而非始终统计原字符序列。
            pairs: Counter[tuple[str, str]] = Counter()
            for sequence in sequences:
                pairs.update(zip(sequence, sequence[1:]))
            if not pairs:
                break
            # 相同频次时按左右 Token 的 Unicode 顺序决胜，保证结果可复现。
            best = min(pairs, key=lambda pair: (-pairs[pair], pair))
            merges.append(best)
            # history 仅服务教学展示，前五项足够与正文的手算表核对。
            ranked = sorted(pairs.items(), key=lambda item: (-item[1], item[0]))
            history.append({
                "round": len(merges),
                "top_pairs": [(left, right, count) for (left, right), count in ranked[:5]],
                "chosen": [best[0], best[1], pairs[best]],
            })
            sequences = [cls._merge(sequence, best) for sequence in sequences]
        return cls(alphabet, merges, history)

    @staticmethod
    def _merge(sequence: list[str], pair: tuple[str, str]) -> list[str]:
        """从左至右做不重叠合并；命中 pair 后跳过两个旧 Token。"""
        merged: list[str] = []
        index = 0
        while index < len(sequence):
            if index + 1 < len(sequence) and (sequence[index], sequence[index + 1]) == pair:
                merged.append(sequence[index] + sequence[index + 1])
                index += 2
            else:
                merged.append(sequence[index])
                index += 1
        return merged

    def encode_tokens(self, text: str) -> list[str]:
        """按训练时保存的合并顺序切分新文本，不会重新统计输入文本。"""
        missing = sorted(set(text) - set(self.alphabet))
        if missing:
            raise ValueError(f"训练集中未出现字符：{missing!r}")
        tokens = list(text)
        # 后续规则依赖前一轮产生的 Token，所以合并顺序不可交换。
        for pair in self.merges:
            tokens = self._merge(tokens, pair)
        return tokens

    def encode(self, text: str) -> list[int]:
        """先切分成 Token，再映射到词表 ID。"""
        return [self.token_to_id[token] for token in self.encode_tokens(text)]

    def decode(self, ids: list[int]) -> str:
        """建立一次反向映射并将 Token 按顺序拼回字符串。"""
        inverse = {index: token for token, index in self.token_to_id.items()}
        try:
            return "".join(inverse[index] for index in ids)
        except KeyError as exc:
            raise ValueError(f"无效 Token ID：{exc.args[0]}") from exc

    def save(self, path: str | Path) -> None:
        """保存字母表、合并规则和教学历史，使编码规则能够重载。"""
        Path(path).write_text(
            json.dumps({"type": "teaching_bpe", "alphabet": self.alphabet, "merges": self.merges, "history": self.history}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "TeachingBPE":
        """读取教学 BPE 文件；JSON 数组形式的 pair 还原为 tuple。"""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("type") != "teaching_bpe":
            raise ValueError("无效的教学 BPE 文件")
        return cls(data["alphabet"], [tuple(pair) for pair in data["merges"]], data.get("history", []))
