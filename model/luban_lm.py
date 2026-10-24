"""把 C2 的 LubanLM 配置组装为完整的 Decoder-only 语言模型。"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .blocks import LubanBlock, RMSNorm
from torch.utils.checkpoint import checkpoint

PROGRAM_ROOT = Path(__file__).resolve().parents[1] / "C3" / "program"

@dataclass
class LubanOutput:
    logits: torch.Tensor
    past_key_values: tuple


@dataclass
class LubanConfig:
    """C2 配置字段的类型化表示；特殊 Token ID 来自实际 Tokenizer。"""

    name: str
    status: str
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    context_length: int
    tie_word_embeddings: bool
    linear_bias: bool
    # C2 的候选 JSON 还没有这些字段；Tokenizer 确定后再填真实 ID。
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | None = None
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0

    @classmethod
    def from_json(cls, path: str | Path) -> "LubanConfig":
        """读取 C2 JSON，并在建模前检查维度关系。"""
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        # 只选当前模型类认识的键；额外说明字段不会误传给网络模块。
        allowed = set(cls.__dataclass_fields__)
        config = cls(**{key: value for key, value in data.items() if key in allowed})
        config.validate()
        return config

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LubanConfig":
        """从保存的 JSON 字典恢复配置，供转换脚本复用。"""
        allowed = set(cls.__dataclass_fields__)
        config = cls(**{key: value for key, value in data.items() if key in allowed})
        config.validate()
        return config

    def validate(self) -> None:
        """验证本课程 Decoder 和 GQA 所需的尺寸约束。"""
        dimensions = (
            "vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads",
            "num_key_value_heads", "head_dim", "intermediate_size", "context_length",
        )
        for key in dimensions:
            value = getattr(self, key)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{key} 必须是正整数，当前为 {value!r}")
        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError("要求 hidden_size = num_attention_heads × head_dim")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads 必须能被 num_key_value_heads 整除")
        if self.head_dim % 2:
            raise ValueError("RoPE 要求 head_dim 为偶数")
        if not isinstance(self.tie_word_embeddings, bool):
            raise ValueError("tie_word_embeddings 必须是布尔值")
        if not isinstance(self.linear_bias, bool):
            raise ValueError("linear_bias 必须是布尔值")
        for name in ("pad_token_id", "bos_token_id", "eos_token_id"):
            token_id = getattr(self, name)
            if token_id is not None and not 0 <= token_id < self.vocab_size:
                raise ValueError(f"{name}={token_id} 超出词表范围 [0,{self.vocab_size})")


class LubanBackbone(nn.Module):
    """Token Embedding、Block 堆叠和 Final RMSNorm。"""

    def __init__(self, config: LubanConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [
                LubanBlock(
                    hidden_size=config.hidden_size,
                    num_attention_heads=config.num_attention_heads,
                    num_key_value_heads=config.num_key_value_heads,
                    head_dim=config.head_dim,
                    intermediate_size=config.intermediate_size,
                    linear_bias=config.linear_bias,
                    rms_norm_eps=config.rms_norm_eps,
                    rope_theta=config.rope_theta,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, gradient_checkpointing=False):
        hidden_states = self.embed_tokens(input_ids)
        presents = []
        for index, layer in enumerate(self.layers):
            past = None if past_key_values is None else past_key_values[index]
            if gradient_checkpointing and self.training:
                if use_cache or past is not None:
                    raise ValueError("training checkpointing cannot use KV cache")
                hidden_states = checkpoint(layer, hidden_states, attention_mask, use_reentrant=False)
            else:
                result = layer(hidden_states, attention_mask, past, use_cache)
                if use_cache:
                    hidden_states, present = result
                    presents.append(present)
                else:
                    hidden_states = result
        return self.norm(hidden_states), tuple(presents)


class LubanLM(nn.Module):
    """输入 Token ID，输出每个位置对整个词表的 Logits。"""

    def __init__(self, config: LubanConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.gradient_checkpointing = False
        self.model = LubanBackbone(config)
        # Linear 的输入宽度 H、输出宽度 V，所以 logits 最后一维等于词表大小。
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 只初始化权重层；RMSNorm 的 gamma 保持其构造时设定的全 1。
        self.apply(self._initialize_weight)
        if config.tie_word_embeddings:
            # 真正共享同一个 Parameter，而不是复制一份数值相同的矩阵。
            self.lm_head.weight = self.model.embed_tokens.weight

    @staticmethod
    def _initialize_weight(module: nn.Module) -> None:
        """随机初始化 Embedding/Linear；实际训练会从这些随机参数开始。"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor, attention_mask=None, past_key_values=None, use_cache=False):
        """处理 [B,T] 的 LongTensor，并返回 [B,T,V] 的原始分数。"""
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids 必须是 [B,T] 二维张量，收到 {tuple(input_ids.shape)}")
        if input_ids.dtype != torch.long:
            raise TypeError(f"input_ids 必须使用 torch.long，收到 {input_ids.dtype}")
        batch_size, sequence_length = input_ids.shape
        if batch_size <= 0 or sequence_length <= 0:
            raise ValueError("Batch 和序列长度必须大于 0")
        past_length = 0 if past_key_values is None else past_key_values[0][0].shape[2]
        if past_key_values is not None and len(past_key_values) != len(self.model.layers):
            raise ValueError("KV cache layer count mismatch")
        if sequence_length + past_length > self.config.context_length:
            raise ValueError(
                f"序列长度 {sequence_length} 超过 context_length={self.config.context_length}"
            )

        hidden_states, presents = self.model(input_ids, attention_mask, past_key_values, use_cache, self.gradient_checkpointing)
        logits = self.lm_head(hidden_states)    # [B,T,H] -> [B,T,V]
        return LubanOutput(logits, presents) if use_cache else logits

    def config_dict(self) -> dict[str, Any]:
        """返回可序列化配置，权重与架构参数可一起保存。"""
        return asdict(self.config)


def main() -> None:
    """运行一条短序列的随机权重前向演示，不做训练或质量评估。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROGRAM_ROOT.parent.parent / "C2" / "program" / "2.4" / "config_50m.json",
        help="C2 的模型 JSON；默认使用 50M 候选配置",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--device", default="cpu", help="例如 cpu 或 cuda")
    args = parser.parse_args()

    config = LubanConfig.from_json(args.config)
    if not 1 <= args.sequence_length <= config.context_length:
        parser.error("--sequence-length 必须位于 1 与 context_length 之间")
    if args.batch_size <= 0:
        parser.error("--batch-size 必须大于 0")

    # 固定种子让随机输入与初始化可复现；Token ID 0、1、2 留给特殊 Token。
    torch.manual_seed(2026)
    model = LubanLM(config).to(args.device).eval()
    input_ids = torch.randint(
        low=3, high=config.vocab_size,
        size=(args.batch_size, args.sequence_length),
        dtype=torch.long, device=args.device,
    )
    with torch.no_grad():
        logits = model(input_ids)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"模型: {config.name}")
    print(f"input_ids.shape = {tuple(input_ids.shape)}")
    print(f"logits.shape    = {tuple(logits.shape)}")
    print(f"参数量          = {parameter_count:,}")
    print("输入和输出均为随机演示数据；这不表示模型已经训练或具备语言能力。")


if __name__ == "__main__":
    main()
