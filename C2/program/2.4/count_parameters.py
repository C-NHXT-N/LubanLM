"""计算本章两种设计候选的参数量；不构造网络，不证明部署兼容性。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_json(path):
    """读取 UTF-8 JSON；兼容 Windows 文本可能带有的 BOM。"""
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def tokenizer_spec(directory):
    """从第 1 章交付目录读取实际词表，并验证跨文件的一致性。"""
    directory = Path(directory)
    tokenizer = read_json(directory / "tokenizer.json")
    # BPE 主词表和 added tokens 分开存储，二者都要计入模型 vocab_size。
    vocab = dict(tokenizer["model"]["vocab"])
    for added in tokenizer.get("added_tokens", []):
        token, token_id = added["content"], added["id"]
        if token in vocab and vocab[token] != token_id:
            raise ValueError(f"同一 Token 的 ID 不一致：{token}")
        vocab[token] = token_id
    # Embedding 按 ID 下标查表，要求 ID 非负、唯一且没有断档。
    ids = list(vocab.values())
    if any(type(token_id) is not int or token_id < 0 for token_id in ids):
        raise ValueError("Token ID 必须是非负整数")
    if len(set(ids)) != len(ids) or set(ids) != set(range(len(ids))):
        raise ValueError("本章要求实际 Token ID 唯一，且从 0 连续编号")
    # 三个边界 Token 是后续模型配置必须沿用的接口。
    special = {token: vocab.get(token) for token in ("<pad>", "<bos>", "<eos>")}
    if any(token_id is None for token_id in special.values()):
        raise ValueError("缺少本课程要求的 PAD/BOS/EOS")
    added_special = {entry["content"] for entry in tokenizer.get("added_tokens", []) if entry.get("special")}
    if not set(special).issubset(added_special):
        raise ValueError("PAD/BOS/EOS 必须被标记为特殊 Token")
    # 不能只相信 tokenizer.json；交付记录也必须报告同样的结果。
    training = read_json(directory / "training_config.json")
    config = read_json(directory / "tokenizer_config.json")
    if training["actual_vocab_size"] != len(ids) or config["vocab_size"] != len(ids):
        raise ValueError("实际词表与导出配置不一致")
    if training["special_token_ids"] != special:
        raise ValueError("特殊 Token ID 与训练记录不一致")
    for name in ("pad", "bos", "eos"):
        if config[f"{name}_token"] != f"<{name}>":
            raise ValueError(f"{name} Token 名称与本课程约定不一致")
    return len(ids), special


def count(config):
    """按 2.4 的无偏置、共享或独立 LM Head 假设累计参数。"""
    # 先验证公式适用的基本尺寸，错误配置不要静默给出一个数字。
    dimensions = ("vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads",
                  "num_key_value_heads", "head_dim", "intermediate_size", "context_length")
    for key in dimensions:
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} 必须为正整数")
    if type(config["tie_word_embeddings"]) is not bool:
        raise ValueError("tie_word_embeddings 必须是布尔值")
    if config.get("linear_bias") is not False:
        raise ValueError("本节公式仅适用于无偏置线性层")

    vocab = config["vocab_size"]
    hidden = config["hidden_size"]
    layers = config["num_hidden_layers"]
    query_heads = config["num_attention_heads"]
    kv_heads = config["num_key_value_heads"]
    head_dim = config["head_dim"]
    intermediate = config["intermediate_size"]
    # 本章候选采用每个 Query 头平均分配 hidden_size，并要求 GQA 可整除分组。
    if hidden != query_heads * head_dim:
        raise ValueError("本章候选要求 hidden_size = num_attention_heads × head_dim")
    if query_heads % kv_heads:
        raise ValueError("Query 头数必须能被 KV 头数整除")
    if head_dim % 2:
        raise ValueError("RoPE 需要偶数 head_dim")

    # Q/O 各有 H×(n_q·d)，K/V 各有 H×(n_kv·d)。
    attention = 2 * hidden * head_dim * (query_heads + kv_heads)
    ffn = 3 * hidden * intermediate
    block_norm = 2 * hidden + 2 * head_dim
    # 每层有 Attention、SwiGLU 三投影和两类 Norm，随后乘以 Block 层数。
    blocks = layers * (attention + ffn + block_norm)
    embedding = vocab * hidden
    # 共享输出权重时 LM Head 使用 Embedding 转置，不新增可训练参数。
    lm_head_extra = 0 if config["tie_word_embeddings"] else embedding
    total = embedding + blocks + hidden + lm_head_extra
    return {
        "q_per_layer": hidden * query_heads * head_dim,
        "k_per_layer": hidden * kv_heads * head_dim,
        "v_per_layer": hidden * kv_heads * head_dim,
        "o_per_layer": hidden * query_heads * head_dim,
        "attention_per_layer": attention,
        "ffn_per_layer": ffn,
        "norm_per_layer": block_norm,
        "embedding": embedding,
        "all_blocks": blocks,
        "final_norm": hidden,
        "lm_head_extra": lm_head_extra,
        "total": total,
    }


def main():
    """读取候选配置，并选择暂定、假设或实际 Tokenizer 词表进行计算。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="只计算指定候选；默认计算本目录两份配置")
    # 两种覆盖来源互斥：一个是手工假设，一个是从实际交付目录读取。
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--vocab-size", type=int, help="仅用于词表大小假设对照")
    source.add_argument("--tokenizer", type=Path, help="读取第 1 章实际导出的 Tokenizer 目录")
    args = parser.parse_args()
    try:
        vocab_override = args.vocab_size
        if args.tokenizer:
            vocab_override, special = tokenizer_spec(args.tokenizer)
            print(f"实际 Tokenizer: vocab_size={vocab_override}; 特殊 ID={special}")
        elif vocab_override is not None:
            print(f"假设词表大小: {vocab_override}")
        else:
            print("采用配置中的暂定词表；正式词表尚待实验确定。")
        base = Path(__file__).resolve().parent
        # 未指定 --config 时，成对输出 50M 和 0.5B 的同口径计算结果。
        paths = [args.config] if args.config else [base / "config_50m.json", base / "config_0.5b.json"]
        for path in paths:
            config = read_json(path)
            # 覆盖仅修改内存字典，不会改写原始 JSON 配置。
            if vocab_override is not None:
                config["vocab_size"] = vocab_override
            result = count(config)
            print(f"\n{config['name']} — 设计候选，尚未验证训练与部署")
            print(f"vocab_size: {config['vocab_size']:,}")
            for key, value in result.items():
                print(f"{key}: {value:,}")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
