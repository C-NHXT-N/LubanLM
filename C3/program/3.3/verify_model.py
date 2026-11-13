"""检查 C2 配置参数量、LubanLM 张量形状、因果性、梯度和短时过拟合。"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
PROGRAM_ROOT = ROOT / "C3" / "program"
sys.path.insert(0, str(PROGRAM_ROOT / "3.2"))
from luban_lm import LubanConfig, LubanLM  # noqa: E402


EXPECTED_COUNTS = {
    "LubanLM-50M": 49_558_400,
    "LubanLM-0.5B": 488_964_864,
}


def formula_parameter_count(config: LubanConfig) -> int:
    """复算 C2 的无偏置、GQA、共享 LM Head 参数公式。"""
    vocab, hidden = config.vocab_size, config.hidden_size
    query_heads, kv_heads, head_dim = config.num_attention_heads, config.num_key_value_heads, config.head_dim
    attention = 2 * hidden * head_dim * (query_heads + kv_heads)
    ffn = 3 * hidden * config.intermediate_size
    norms = 2 * hidden + 2 * head_dim
    embedding = vocab * hidden
    lm_head_extra = 0 if config.tie_word_embeddings else vocab * hidden
    return embedding + config.num_hidden_layers * (attention + ffn + norms) + hidden + lm_head_extra


def parameter_count_without_allocating(config: LubanConfig) -> tuple[int, str]:
    """優先在 meta device 上构造模型，从模块实际参数形状求和而不分配权重内存。"""
    try:
        # meta 张量只保存形状信息，适合核对 0.5B 配置，不会占用数 GB 内存。
        with torch.device("meta"):
            model = LubanLM(config)
        count = sum(parameter.numel() for parameter in model.parameters())
        return count, "meta 模型参数形状求和"
    except (AttributeError, NotImplementedError, RuntimeError, TypeError) as exc:
        # 旧版 PyTorch 可能没有完整 meta 支持；此时仍用 C2 公式交叉核对。
        return formula_parameter_count(config), f"参数公式回退（meta 不可用：{type(exc).__name__}）"


def load_c2_config(path: Path) -> LubanConfig:
    """加载候选 JSON；名称、词表、尺寸均来自 C2 文件。"""
    config = LubanConfig.from_json(path)
    if config.name not in EXPECTED_COUNTS:
        raise ValueError(f"无法识别候选名称 {config.name!r}：{path}")
    return config


def check_c2_parameter_counts() -> None:
    """对两份 C2 配置验证公式结果和模型实际参数形状。"""
    config_dir = ROOT / "C2" / "program" / "2.4"
    files = (config_dir / "config_50m.json", config_dir / "config_0.5b.json")
    for path in files:
        config = load_c2_config(path)
        formula_count = formula_parameter_count(config)
        model_count, method = parameter_count_without_allocating(config)
        expected = EXPECTED_COUNTS[config.name]
        assert formula_count == expected, (config.name, formula_count, expected)
        assert model_count == expected, (config.name, model_count, expected)
        print(f"参数量 {config.name}: {model_count:,}；{method}；与 C2 一致")


def build_tiny_config(base: LubanConfig) -> LubanConfig:
    """缩小 V/H/层数做单元级验证，不改变张量运算和模块连接方式。"""
    tiny = replace(
        base,
        name="LubanLM-tiny-verification",
        status="correctness_demo_only",
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=64,
        context_length=16,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    tiny.validate()
    return tiny


def check_shapes_causality_loss_and_gradients(config: LubanConfig) -> None:
    """验证单/多序列形状、Attention/FFN 分支、因果性与反向传播。"""
    torch.manual_seed(2026)
    model = LubanLM(config).cpu().eval()
    assert model.lm_head.weight is model.model.embed_tokens.weight, "共享权重必须是同一 Parameter"

    # 两种 Batch 大小验证 [B,T] -> [B,T,V]，T 在小配置范围内。
    for batch_size in (1, 2):
        input_ids = torch.randint(3, config.vocab_size, (batch_size, 5), dtype=torch.long)
        with torch.no_grad():
            logits = model(input_ids)
        assert logits.shape == (batch_size, 5, config.vocab_size), logits.shape
        print(f"前向形状 B={batch_size}: input_ids={tuple(input_ids.shape)} -> logits={tuple(logits.shape)}")

    # 逐层检查两条分支外部宽度能与 Pre-Norm 残差相加。
    sample_ids = torch.randint(3, config.vocab_size, (2, 5), dtype=torch.long)
    block = model.model.layers[0]
    block_input = model.model.embed_tokens(sample_ids)  # [2,5,H]
    normalized = block.input_layernorm(block_input)
    attention_output = block.self_attn(normalized)
    mlp_output = block.mlp(block.post_attention_layernorm(block_input))
    assert attention_output.shape == block_input.shape == mlp_output.shape
    gate_shape = block.mlp.gate_proj(block_input).shape
    up_shape = block.mlp.up_proj(block_input).shape
    assert gate_shape == up_shape == (2, 5, config.intermediate_size)
    print(
        "分支形状: Attention ", tuple(attention_output.shape),
        "; SwiGLU gate/up ", tuple(gate_shape), " -> down ", tuple(mlp_output.shape), sep=""
    )

    # 只更改最后一个 Token；更早位置的 logits 不应受未来 Token 改变。
    first = torch.tensor([[7, 11, 13, 17]], dtype=torch.long)
    changed_future = torch.tensor([[7, 11, 13, 23]], dtype=torch.long)
    with torch.no_grad():
        original_logits = model(first)
        changed_logits = model(changed_future)
    torch.testing.assert_close(original_logits[:, :3], changed_logits[:, :3], rtol=1e-5, atol=1e-6)
    print("因果性：改变第 4 个 Token 后，第 1～3 个位置的 logits 保持不变")

    # next-token 目标右移一格；忽略最后一个没有后继目标的位置。
    model.train()
    logits = model(sample_ids)
    loss = F.cross_entropy(
        logits[:, :-1, :].reshape(-1, config.vocab_size),
        sample_ids[:, 1:].reshape(-1),
    )
    assert torch.isfinite(loss), f"Loss 非有限值：{loss.item()}"
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients, "没有任何参数获得梯度"
    assert all(torch.isfinite(gradient).all().item() for gradient in gradients), "Gradient 出现 NaN/Inf"
    assert all(parameter.grad is not None for parameter in model.parameters()), "有参数没有参与梯度计算"
    print(f"有限 Loss: {loss.item():.6f}；有限梯度张量: {len(gradients)}")


def overfit_repeated_batch(config: LubanConfig, steps: int) -> tuple[float, float]:
    """在重复的短 Token 序列上训练几步，只证明可学习，不衡量语言能力。"""
    torch.manual_seed(2027)
    torch.set_num_threads(1)
    model = LubanLM(config).cpu().train()
    # 四条相同的短序列；所有 ID 都在微型词表范围内且不使用特殊 Token。
    one_sequence = torch.tensor([5, 8, 13, 21, 34, 55], dtype=torch.long)
    input_ids = one_sequence.unsqueeze(0).repeat(4, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0)

    def batch_loss() -> torch.Tensor:
        predictions = model(input_ids)
        return F.cross_entropy(
            predictions[:, :-1, :].reshape(-1, config.vocab_size),
            input_ids[:, 1:].reshape(-1),
        )

    with torch.no_grad():
        initial_loss = batch_loss().item()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = batch_loss()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        final_loss = batch_loss().item()
    assert final_loss < initial_loss, (
        f"重复样本上的 Loss 没有下降：{initial_loss:.6f} -> {final_loss:.6f}；"
        "可尝试增加 --steps 或检查 PyTorch 数值环境"
    )
    return initial_loss, final_loss


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100, help="微型重复数据集训练步数")
    parser.add_argument("--skip-overfit", action="store_true", help="只做形状/梯度检查")
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps 必须是正整数")

    # 参数量只在无存储开销的 meta 张量上统计，不会初始化 0.5B 实际权重。
    check_c2_parameter_counts()
    base = load_c2_config(ROOT / "C2" / "program" / "2.4" / "config_50m.json")
    tiny = build_tiny_config(base)
    actual_tiny_count = sum(parameter.numel() for parameter in LubanLM(tiny).parameters())
    assert actual_tiny_count == formula_parameter_count(tiny)
    print(f"微型正确性配置参数量: {actual_tiny_count:,}（仅用于快速验证）")

    check_shapes_causality_loss_and_gradients(tiny)
    if not args.skip_overfit:
        initial, final = overfit_repeated_batch(tiny, args.steps)
        print(f"重复样本短训 Loss: {initial:.6f} -> {final:.6f}（正确性演示，不是正式评估）")

    print("所有已启用的模型正确性检查通过。短训结果不能说明模型学会了通用语言能力。")


if __name__ == "__main__":
    main()
