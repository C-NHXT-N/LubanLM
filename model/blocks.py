"""LubanLM 的手写 Transformer Block 核心部件。

默认教学路径优先呈现张量如何流动；训练可显式选择 SDPA 执行路径。
Qwen3 风格的模块命名便于后续把权重映射到
标准 Hugging Face Qwen3 模型；命名相同本身不代表 RKLLM 已验证兼容。
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    """沿最后一维计算 RMSNorm；输入 [B,T,D]，输出形状不变。"""

    def __init__(self, dimension: int, eps: float = 1e-6) -> None:
        super().__init__()
        # gamma 是逐特征的可训练缩放参数；C2 的参数统计不包含额外 bias。
        self.weight = nn.Parameter(torch.ones(dimension))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 用 float32 累加平方，避免半精度输入在求均方时过早溢出或下溢。
        input_dtype = hidden_states.dtype
        values = hidden_states.float()
        mean_square = values.pow(2).mean(dim=-1, keepdim=True)
        normalized = values * torch.rsqrt(mean_square + self.variance_epsilon)
        # gamma 也以 float32 参与乘法，最后再回到输入 dtype 供后续层使用。
        return (normalized * self.weight.float()).to(input_dtype)


class RotaryEmbedding(nn.Module):
    """生成 RoPE 的 cos/sin；每个头的前半与后半坐标成对旋转。"""

    def __init__(self, head_dim: int, theta: float = 1_000_000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE 要求 head_dim 为偶数，收到 {head_dim}")
        # inverse_frequency[r] = theta ** (-2r / head_dim)，不是可训练参数。
        indices = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (theta ** (indices / head_dim))
        # 非持久 buffer 会随模型移动设备，但不会写入 state_dict。
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self, sequence_length: int, device: torch.device, position_offset: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # position_ids=[0,1,...,T-1]；外积得到每个位置、每对坐标的旋转角。
        positions = torch.arange(position_offset, position_offset + sequence_length, device=device, dtype=torch.float32)
        angles = torch.outer(positions, self.inv_freq.to(device=device))
        # 返回 [1,1,T,d/2]，广播到 batch 与头数；稍后复制到输入 dtype。
        return angles.cos()[None, None, :, :], angles.sin()[None, None, :, :]


def rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    """将最后一维分为两半并返回 [-第二半, 第一半]。"""
    first_half, second_half = hidden_states.chunk(2, dim=-1)
    return torch.cat((-second_half, first_half), dim=-1)


def apply_rotary_pos_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """对 Q/K 应用半分组 RoPE；Q/K 为 [B,heads,T,d]。"""
    # cos/sin 原本是 [1,1,T,d/2]，各复制一份以匹配完整 head_dim。
    cos_full = torch.cat((cos, cos), dim=-1).to(dtype=query.dtype)
    sin_full = torch.cat((sin, sin), dim=-1).to(dtype=query.dtype)
    # x*cos + rotate_half(x)*sin 等价于对每一对坐标做二维旋转。
    query = query * cos_full + rotate_half(query) * sin_full
    key = key * cos_full.to(dtype=key.dtype) + rotate_half(key) * sin_full.to(dtype=key.dtype)
    return query, key


class LubanAttention(nn.Module):
    """手写 GQA：逐头 QK-Norm、RoPE、因果注意力与输出投影。"""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        linear_bias: bool = False,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1_000_000.0,
    ) -> None:
        super().__init__()
        if hidden_size != num_attention_heads * head_dim:
            raise ValueError("本课程配置要求 hidden_size = num_attention_heads × head_dim")
        if num_attention_heads % num_key_value_heads != 0:
            raise ValueError("num_attention_heads 必须能被 num_key_value_heads 整除")

        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.num_key_value_groups = num_attention_heads // num_key_value_heads
        # 执行方式不是模型结构，也不增加 state_dict 键。
        self.attention_backend = 'manual'

        # Q 有 n_q 个头；K/V 只投影 n_kv 个头，因此省下 K/V 权重和缓存。
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=linear_bias)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=linear_bias)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=linear_bias)
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=linear_bias)

        # 一条长度 d 的 gamma 在同类型的所有头之间共享，不为每个头复制参数。
        self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(head_dim, theta=rope_theta)

    def forward(self, hidden_states: torch.Tensor, attention_mask=None, past_key_value=None, use_cache=False):
        # 输入保留为 [B,T,H]；B 是批量大小，T 是序列长度，H 是隐藏维度。
        batch_size, sequence_length, _ = hidden_states.shape

        # 三个线性层逐 Token 投影，先得到未拆头的 Q/K/V 最后一维。
        query = self.q_proj(hidden_states)  # [B,T,n_q*d]
        key = self.k_proj(hidden_states)    # [B,T,n_kv*d]
        value = self.v_proj(hidden_states)  # [B,T,n_kv*d]

        # 把 head 维拆出来并移到时间维前面，方便批量做每头注意力。
        query = query.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, sequence_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, sequence_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        # 现在 Q=[B,n_q,T,d]，K/V=[B,n_kv,T,d]。

        # QK-Norm 在每一个位置、每一个头的 d 维上独立执行。
        query = self.q_norm(query)
        key = self.k_norm(key)

        # 同一批位置使用同一组 RoPE 角度；旋转发生在 Q/K 点积之前。
        past_length = 0 if past_key_value is None else past_key_value[0].shape[2]
        cos, sin = self.rotary_emb(sequence_length, hidden_states.device, past_length)
        query, key = apply_rotary_pos_emb(query, key, cos, sin)

        if past_key_value is not None:
            key = torch.cat((past_key_value[0], key), dim=2)
            value = torch.cat((past_key_value[1], value), dim=2)
        present = (key, value) if use_cache else None

        # GQA 中一组 Query 头共享一个 KV 头；重复后可用统一的矩阵乘法计算。
        # repeat_interleave 按头维复制，例：2 个 KV 头各重复 4 次得到 8 个头。
        if self.num_key_value_groups > 1:
            key = key.repeat_interleave(self.num_key_value_groups, dim=1)
            value = value.repeat_interleave(self.num_key_value_groups, dim=1)
        # 此时 Q/K/V 的头数都为 n_q，形状均可视作 [B,n_q,T,d]。

        if self.attention_backend == 'sdpa':
            allowed = None
            causal = past_length == 0 and attention_mask is None
            if not causal:
                # KV cache 的矩形掩码必须按绝对位置对齐，不能用左上角三角。
                qpos = torch.arange(past_length, past_length + sequence_length, device=query.device)
                kpos = torch.arange(key.shape[2], device=query.device)
                allowed = (kpos[None, :] <= qpos[:, None])[None, None, :, :]
                if attention_mask is not None:
                    if attention_mask.shape != (batch_size, key.shape[2]):
                        raise ValueError('attention_mask must cover past and current key positions')
                    allowed = allowed & attention_mask[:, None, None, :].bool()
            attention_output = F.scaled_dot_product_attention(
                query, key, value, attn_mask=allowed, is_causal=causal, dropout_p=0.0)
            attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, sequence_length, self.hidden_size)
            output = self.o_proj(attention_output)
            return (output, present) if use_cache else output

        # 用 float32 计算分数和 Softmax；除以 sqrt(d) 控制点积尺度。
        with torch.autocast(hidden_states.device.type, enabled=False):
            scores = torch.matmul(query.float(), key.float().transpose(-2, -1))
            scores = scores / math.sqrt(self.head_dim)  # [B,n_q,T,T]

        # 下三角为 True：位置 i 只看 j<=i。上三角未来位置填入极小分数。
        query_positions = torch.arange(past_length, past_length + sequence_length, device=scores.device)
        key_positions = torch.arange(key.shape[2], device=scores.device)
        causal_mask = key_positions[None, :] <= query_positions[:, None]
        allowed = causal_mask[None, None, :, :]
        if attention_mask is not None:
            if attention_mask.shape != (batch_size, key.shape[2]):
                raise ValueError("attention_mask must cover past and current key positions")
            allowed = allowed & attention_mask[:, None, None, :].bool()
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)

        # 最后一维是 Key 位置；每行概率和为 1，未来位置的概率为 0。
        attention_weights = F.softmax(scores, dim=-1)

        # 每个 Query 用权重对 V 加权求和，恢复至 V 的 dtype。
        attention_output = torch.matmul(attention_weights.to(dtype=value.dtype), value)
        # [B,n_q,T,d] -> [B,T,n_q,d] -> [B,T,H]，再回到残差宽度 H。
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, sequence_length, self.hidden_size)
        output = self.o_proj(attention_output)
        return (output, present) if use_cache else output


class LubanMLP(nn.Module):
    """SwiGLU 前馈层：gate 与 up 两支相乘，再由 down 投影回 H。"""

    def __init__(
        self, hidden_size: int, intermediate_size: int, linear_bias: bool = False
    ) -> None:
        super().__init__()
        # gate/up 将 H 扩展到 I；down 再从 I 投回 H。
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=linear_bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=linear_bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=linear_bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 两支输入相同但权重不同；逐元素门控后输出形状仍为 [B,T,I]。
        gate = F.silu(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        gated = gate * up
        return self.down_proj(gated)  # [B,T,I] -> [B,T,H]


class LubanBlock(nn.Module):
    """Pre-Norm Transformer Block，含 Attention 和 FFN 两条残差路径。"""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        intermediate_size: int,
        linear_bias: bool = False,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1_000_000.0,
    ) -> None:
        super().__init__()
        # 这两个 Norm 各有自己的 H 维 gamma，不共享权重。
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = LubanAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            linear_bias=linear_bias,
            rms_norm_eps=rms_norm_eps,
            rope_theta=rope_theta,
        )
        self.mlp = LubanMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            linear_bias=linear_bias,
        )

    def forward(self, hidden_states: torch.Tensor, attention_mask=None, past_key_value=None, use_cache=False):
        # 第一条 Pre-Norm 残差：输入 X 仍绕过归一化与 Attention。
        attention_input = self.input_layernorm(hidden_states)
        attention_output = self.self_attn(attention_input, attention_mask, past_key_value, use_cache)
        if use_cache:
            attention_output, present = attention_output
        hidden_states = hidden_states + attention_output

        # 第二条残差以 Attention 更新后的 U 为旁路，而不是使用最初的 X。
        mlp_input = self.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states + self.mlp(mlp_input)
        return (hidden_states, present) if use_cache else hidden_states
