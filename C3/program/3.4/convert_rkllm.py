"""按 Rockchip RKLLM Toolkit 1.3.0 官方 Python API 尝试导出 .rkllm。

脚本只负责 Toolkit 构建与文件检查，不在 PC 上推理，也不验证板端 Runtime。
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_TOOLKIT_VERSION = "1.3.0"


def toolkit_version() -> str:
    """从 Python 包元数据或模块属性读取版本，便于实验记录可复现。"""
    for distribution_name in ("rkllm-toolkit", "rkllm_toolkit"):
        try:
            return importlib.metadata.version(distribution_name)
        except importlib.metadata.PackageNotFoundError:
            continue
    try:
        import rkllm

        return str(getattr(rkllm, "__version__", "unknown"))
    except ImportError:
        return "not-installed"


def run(args: argparse.Namespace) -> None:
    """加载标准 Hugging Face 目录、构建目标模型并导出文件。"""
    if platform.system() != "Linux":
        raise RuntimeError(
            "本节选择的 RKLLM Toolkit 在 WSL Ubuntu/Linux PC 侧运行；"
            "请在 WSL 终端执行，而不是 Windows Python。"
        )
    if not args.model_dir.is_dir():
        raise FileNotFoundError(f"标准模型目录不存在：{args.model_dir}")
    if args.model_dir.resolve() == args.output.resolve():
        raise ValueError("输出 .rkllm 不能与输入模型目录相同")
    if args.output.suffix.lower() != ".rkllm":
        raise ValueError("--output 文件扩展名必须是 .rkllm")
    if args.output.exists():
        raise FileExistsError(f"输出文件已存在，为避免覆盖请改用新文件名：{args.output}")
    if args.max_context <= 0:
        raise ValueError("--max-context 必须为正整数")
    if args.num_npu_core <= 0:
        raise ValueError("--num-npu-core 必须为正整数")
    if args.quantize and not args.calibration_data:
        raise ValueError("量化需要 --quantize 与 --calibration-data 一起指定")
    if args.calibration_data and not args.calibration_data.is_file():
        raise FileNotFoundError(f"校准数据文件不存在：{args.calibration_data}")

    version = toolkit_version()
    if version != EXPECTED_TOOLKIT_VERSION and not args.allow_other_version:
        raise RuntimeError(
            f"本脚本按 RKLLM Toolkit {EXPECTED_TOOLKIT_VERSION} 编写，"
            f"当前检测版本为 {version!r}。请安装选择版本，或核对该版本 API 后显式传入 --allow-other-version。"
        )
    if version != EXPECTED_TOOLKIT_VERSION:
        print(f"警告：脚本目标 Toolkit={EXPECTED_TOOLKIT_VERSION}，当前={version}；请单独核对 API。")

    # 延迟导入，让用户即使尚未安装 Toolkit，也能通过 --help 查看参数。
    try:
        from rkllm.api import RKLLM
    except ImportError as exc:
        raise RuntimeError(
            "找不到 rkllm.api.RKLLM；请先在 WSL 安装与当前 Python ABI 匹配的官方 Toolkit。"
        ) from exc

    args.output.parent.mkdir(parents=True, exist_ok=True)
    llm = RKLLM()

    # 这些关键字参数沿用官方 1.3.0 导出示例的 load_huggingface 接口。
    load_result = llm.load_huggingface(
        model=str(args.model_dir.resolve()),
        model_lora=None,
        device=args.device,
        dtype="float32",
        custom_config=None,
        load_weight=True,
    )
    print(f"load_huggingface 返回值: {load_result}")
    if load_result != 0:
        raise RuntimeError(f"Toolkit 加载模型失败，返回码 {load_result}")

    # 默认关闭量化，先确认随机权重能否通过架构转换；量化需另备有意义的校准样本。
    build_result = llm.build(
        do_quantization=args.quantize,
        optimization_level=args.optimization_level,
        quantized_dtype=args.quantized_dtype,
        quantized_algorithm=args.quantized_algorithm,
        target_platform=args.target_platform,
        num_npu_core=args.num_npu_core,
        extra_qparams=None,
        dataset=str(args.calibration_data.resolve()) if args.calibration_data else None,
        hybrid_rate=0,
        max_context=args.max_context,
    )
    print(f"build 返回值: {build_result}")
    if build_result != 0:
        raise RuntimeError(f"Toolkit 构建失败，返回码 {build_result}")

    # export_rkllm 返回码为 0 只是必要条件；还要确认实际产物可见且非空。
    export_result = llm.export_rkllm(str(args.output.resolve()))
    print(f"export_rkllm 返回值: {export_result}")
    if export_result != 0:
        raise RuntimeError(f"Toolkit 导出失败，返回码 {export_result}")
    if not args.output.is_file() or args.output.stat().st_size <= 0:
        raise RuntimeError("导出接口返回成功，但没有生成非空 .rkllm 文件")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "toolkit": "RKLLM Toolkit",
        "toolkit_version": version,
        "host": platform.platform(),
        "input_model_dir": str(args.model_dir.resolve()),
        "output_rkllm": str(args.output.resolve()),
        "output_size_bytes": args.output.stat().st_size,
        "target_platform": args.target_platform,
        "num_npu_core": args.num_npu_core,
        "max_context": args.max_context,
        "quantized": args.quantize,
        "calibration_data": str(args.calibration_data.resolve()) if args.calibration_data else None,
        "load_return_code": int(load_result),
        "build_return_code": int(build_result),
        "export_return_code": int(export_result),
        "board_runtime_test": "not_run",
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"产物: {args.output} ({args.output.stat().st_size:,} bytes)")
    print(f"记录: {manifest_path}")
    print("本脚本未验证板端 RKLLM Runtime；请将实验结论单独记录。")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True, help="prepare_rkllm_model.py 生成的 HF 目录")
    parser.add_argument("--output", type=Path, required=True, help="新 .rkllm 文件路径")
    parser.add_argument("--target-platform", default="RK3576", help="RKLLM Toolkit 平台名；本章默认 RK3576")
    parser.add_argument("--num-npu-core", type=int, default=3)
    parser.add_argument("--max-context", type=int, default=2048)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--optimization-level", type=int, default=1)
    parser.add_argument("--quantize", action="store_true", help="开启量化；随机模型演示默认关闭")
    parser.add_argument("--calibration-data", type=Path, help="量化用校准 JSON；启用 --quantize 时必需")
    parser.add_argument("--quantized-dtype", default="W8A8")
    parser.add_argument("--quantized-algorithm", default="normal")
    parser.add_argument(
        "--allow-other-version", action="store_true",
        help="允许运行在非 1.3.0 Toolkit 上；先查阅该版本官方 API 并将版本写入实验记录",
    )
    args = parser.parse_args()
    try:
        run(args)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
