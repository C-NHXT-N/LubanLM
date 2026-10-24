"""从随机 LubanLM 权重准备一份标准 Hugging Face Qwen3 模型目录。

这一步只验证 PyTorch -> 标准目录的数据整理和严格权重映射。它不运行
RKLLM Toolkit，也不说明该 LubanLM 配置最终一定能够转换。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
PROGRAM_ROOT = ROOT / "C3" / "program"
sys.path.insert(0, str(PROGRAM_ROOT / "3.2"))
from luban_lm import LubanConfig, LubanLM  # noqa: E402


def tokenizer_ids(tokenizer_directory: Path, vocab_size: int) -> dict[str, int]:
    """从第 1 章实际 Tokenizer 读取词表大小与特殊 Token ID。"""
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("需要安装与所选 RKLLM Toolkit 配套的 transformers") from exc

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_directory, local_files_only=True, use_fast=True
    )
    actual_vocab_size = len(tokenizer)
    if actual_vocab_size != vocab_size:
        raise ValueError(
            f"Tokenizer 实际词表为 {actual_vocab_size}，模型配置为 {vocab_size}。"
            "先依据 C1 最终词表更新 C2 配置，并重新计算参数量，再执行转换实验。"
        )

    vocabulary = tokenizer.get_vocab()
    result: dict[str, int] = {}
    for token_name, spelling in (
        ("pad_token_id", "<pad>"),
        ("bos_token_id", "<bos>"),
        ("eos_token_id", "<eos>"),
    ):
        if spelling not in vocabulary:
            raise ValueError(f"实际 Tokenizer 缺少特殊 Token {spelling!r}")
        token_id = int(vocabulary[spelling])
        if token_id < 0 or token_id >= vocab_size:
            raise ValueError(f"特殊 Token {spelling} 的 ID={token_id} 超出词表范围")
        result[token_name] = token_id
    return result


def build_hf_qwen3_directory(
    config: LubanConfig,
    tokenizer_directory: Path,
    output_directory: Path,
    seed: int,
    checkpoint_path: Path | None = None,
) -> None:
    """创建 Qwen3ForCausalLM，再严格载入 LubanLM 的同名权重。"""
    try:
        import transformers
        from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
    except ImportError as exc:
        raise RuntimeError(
            "需要提供包含 Qwen3Config/Qwen3ForCausalLM 的 Transformers 环境；"
            "此脚本不会联网下载模型或依赖。"
        ) from exc

    token_ids = tokenizer_ids(tokenizer_directory, config.vocab_size)
    # 特殊 ID 写回 LubanConfig，以便保存的记录与模型、Tokenizer 保持一致。
    config.pad_token_id = token_ids["pad_token_id"]
    config.bos_token_id = token_ids["bos_token_id"]
    config.eos_token_id = token_ids["eos_token_id"]
    config.validate()

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 50M 是默认实验规模；初始化 0.5B 会占用大量内存，不适合首轮转换验证。
    source_model = LubanLM(config).cpu().eval()
    if checkpoint_path is not None:
        state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if state['config'] != asdict(config):
            raise ValueError('Checkpoint config differs from export config')
        import hashlib
        digest = hashlib.sha256((tokenizer_directory / 'tokenizer.json').read_bytes()).hexdigest()
        if state['tokenizer_sha256'] != digest:
            raise ValueError('Checkpoint tokenizer hash differs from export tokenizer')
        source_model.load_state_dict(state['model'], strict=True)
    hf_config = Qwen3Config(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        max_position_embeddings=config.context_length,
        rms_norm_eps=config.rms_norm_eps,
        rope_theta=config.rope_theta,
        attention_bias=config.linear_bias,
        tie_word_embeddings=config.tie_word_embeddings,
        pad_token_id=config.pad_token_id,
        bos_token_id=config.bos_token_id,
        eos_token_id=config.eos_token_id,
        use_cache=True,
    )
    hf_config.architectures = ["Qwen3ForCausalLM"]
    target_model = Qwen3ForCausalLM(hf_config).cpu().eval()

    # strict=True 要求参数名称和形状都完整匹配；不允许漏权重或静默随机初始化。
    target_model.load_state_dict(source_model.state_dict(), strict=True)
    # 映射后还要验证计算；名称/形状匹配本身不能证明前向实现一致。
    target_model.config._attn_implementation = 'eager'
    probe = torch.tensor([[config.bos_token_id, 3, 4, 5]], dtype=torch.long)
    with torch.no_grad():
        source_logits = source_model(probe)
        target_logits = target_model(probe).logits
    mapping_error = float((source_logits - target_logits).abs().max())
    if mapping_error > 2e-4:
        raise RuntimeError(f'Qwen3 mapping logits differ: {mapping_error}')
    source_count = sum(parameter.numel() for parameter in source_model.parameters())
    target_count = sum(parameter.numel() for parameter in target_model.parameters())
    if source_count != target_count:
        raise RuntimeError(f"权重映射后参数量不符：LubanLM={source_count}，Qwen3={target_count}")

    # save_pretrained 产生 config.json 与 model.safetensors；Tokenizer 复制到同一目录。
    output_directory.mkdir(parents=True, exist_ok=False)
    target_model.save_pretrained(output_directory, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_directory, local_files_only=True, use_fast=True
    )
    tokenizer.save_pretrained(output_directory)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": "trained local checkpoint" if checkpoint_path else "randomly_initialized LubanLM; not a trained checkpoint",
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint_step": state['step'] if checkpoint_path else None,
        "training_input_tokens": state['input_tokens'] if checkpoint_path else 0,
        "qwen3_logits_max_abs_error": mapping_error,
        "source_config": asdict(config),
        "source_tokenizer_directory": str(tokenizer_directory.resolve()),
        "tokenizer_length": len(tokenizer),
        "token_ids": token_ids,
        "parameter_count": source_count,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "weight_mapping": "LubanLM state_dict -> Qwen3ForCausalLM strict=True",
        "rkllm_conversion_status": "not_run",
    }
    (output_directory / "lubanlm_conversion_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"标准模型目录已生成: {output_directory}")
    print(f"参数量: {source_count:,}；严格映射: 通过")
    print(f"Tokenizer: {len(tokenizer):,} tokens；特殊 ID: {token_ids}")
    print(f"Qwen3 logits 最大误差: {mapping_error}; RKLLM 转换状态: not_run")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "C2" / "program" / "2.4" / "config_50m.json",
        help="默认使用 50M 规格；首次实验不建议使用 0.5B",
    )
    parser.add_argument(
        "--tokenizer-dir", type=Path, required=True,
        help="C1 最终确定并导出的 Hugging Face Fast Tokenizer 目录",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="一个尚不存在的输出目录，将写入权重、config 和 tokenizer 文件",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument('--checkpoint', type=Path, help='可选：本地训练检查点；未传时仍只导出随机模型')
    args = parser.parse_args()

    config = LubanConfig.from_json(args.config)
    if not args.tokenizer_dir.is_dir():
        parser.error(f"Tokenizer 目录不存在：{args.tokenizer_dir}")
    if args.output_dir.exists():
        parser.error(f"输出目录已存在，请选一个新目录：{args.output_dir}")
    try:
        build_hf_qwen3_directory(config, args.tokenizer_dir, args.output_dir, args.seed, args.checkpoint)
    except (ImportError, RuntimeError, ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
