"""Synthetic, data-free LoRA throughput benchmark; never trains on project data."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import random
import shutil
import statistics
import subprocess
import threading
import time
import traceback
from pathlib import Path

from scripts.common import WORKSPACE, sha256, utc, write_json
from scripts.linux_assistant.baseline import exclusive_job
from scripts.linux_assistant.common import config, verify_model
from scripts.linux_assistant.modeling import Padding, encode_training
from scripts.linux_assistant.train import make_model, seed_everything


def fixture_rows():
    """Return a fixed, source-free mix of Chinese, code, log, and multi-turn examples."""
    rows = []
    for i in range(32):
        variant = i + 1
        if i % 4 == 0:
            messages = [
                {"role": "user", "content": (
                    f"合成基准问题 {variant}：请说明如何只读检查 Linux 主机的发行版、内核和磁盘空间。"
                    "不要声称已经执行命令，并分别解释命令用途。"
                )},
                {"role": "assistant", "content": (
                    "可以运行 `cat /etc/os-release` 查看发行版信息、`uname -r` 查看内核版本，"
                    "以及 `df -h` 查看文件系统容量。它们只读取系统信息；实际结果以当前主机输出为准。"
                )},
            ]
        elif i % 4 == 1:
            messages = [
                {"role": "user", "content": (
                    f"合成基准问题 {variant}：我想在当前目录的 `service.log` 中查找包含 ERROR 的行号。"
                    "给出一条只读命令，并简要解释参数。"
                )},
                {"role": "assistant", "content": (
                    "可以使用 `grep -n 'ERROR' ./service.log`。`-n` 显示匹配行号，"
                    "`./service.log` 指当前目录中的文件；此命令读取文件，不会修改它。"
                )},
            ]
        elif i % 4 == 2:
            log_count = (i % 7) + 1
            log_lines = "\n".join(
                f"[合成日志 {j + 1:02d}] component=worker id={variant}-{j} status=ready elapsed_ms={20 + j}"
                for j in range(log_count)
            )
            messages = [
                {"role": "user", "content": (
                    f"合成基准问题 {variant}：请区分以下日志中可以确认的事实和仍需补充的信息。"
                    "不要推断未出现的硬件或运行状态。\n" + log_lines
                )},
                {"role": "assistant", "content": (
                    f"可确认的事实是：提供的合成日志包含 {log_count} 条记录，组件标记为 worker，"
                    "状态字段为 ready。日志没有给出实际板型、内存容量或故障原因，因此不能据此推断这些信息。"
                )},
            ]
        else:
            prior_lines = "\n".join(
                f"[合成上下文 {j + 1:02d}] module=app request={variant}-{j} result=unknown"
                for j in range((i % 6) + 1)
            )
            messages = [
                {"role": "user", "content": (
                    f"合成多轮基准问题 {variant}：我在 Linux 上排查一个应用启动问题。\n{prior_lines}"
                )},
                {"role": "assistant", "content": "请提供启动命令、完整错误信息和相关版本；目前这些日志不足以定位原因。"},
                {"role": "user", "content": (
                    "补充：命令是 `python3 ./app.py`，日志末行是 `ModuleNotFoundError: No module named 'yaml'`。"
                    "请给出不自动执行命令的排查步骤，并说明还要确认哪个 Python 环境。"
                )},
                {"role": "assistant", "content": (
                    "错误表明执行该脚本的 Python 环境没有找到 `yaml` 模块；它本身不能证明模块完全未安装。"
                    "先确认 `python3` 的解释器路径和项目虚拟环境，再在同一解释器下检查 PyYAML 是否可导入。"
                    "如果确认缺少依赖，再按项目锁定文件安装；不要混用系统 Python 与虚拟环境。"
                )},
            ]
        rows.append({"id": f"synthetic-benchmark-{variant:02d}", "messages": messages})
    return rows


def validate_role_pattern(rows):
    for row in rows:
        roles = [message.get("role") for message in row["messages"]]
        if not roles or roles[0] != "user" or roles[-1] != "assistant":
            raise ValueError(f"Malformed synthetic conversation: {row['id']}")
        if any(a == b for a, b in zip(roles, roles[1:])):
            raise ValueError(f"Synthetic messages must alternate user/assistant: {row['id']}")


def encode_fixtures(c, tokenizer):
    rows = fixture_rows()
    validate_role_pattern(rows)
    encoded = [encode_training(c, tokenizer, row) for row in rows]
    if len(encoded) != 32 or any(not any(token != -100 for token in row["labels"]) for row in encoded):
        raise ValueError("Synthetic benchmark must contain 32 nonempty supervised examples")
    return rows, encoded


def query_nvidia_smi(gpu_index, fields):
    command = [
        "nvidia-smi", f"--id={gpu_index}", f"--query-gpu={','.join(fields)}",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=5, check=True)
    record = next(csv.reader([result.stdout.strip()]))
    if len(record) != len(fields):
        raise RuntimeError(f"Unexpected nvidia-smi result: {result.stdout.strip()}")
    return record


def allocator_fraction_for_budget(free_bytes, total_bytes, reserve_mib, max_fraction):
    """Cap this process below currently free VRAM while preserving the requested reserve."""
    if total_bytes <= 0 or free_bytes < 0:
        raise ValueError("Invalid CUDA memory report")
    if not 0 < max_fraction <= 1:
        raise ValueError("allocator_max_memory_fraction must be in (0, 1]")
    reserve_bytes = int(reserve_mib * 2**20)
    available_bytes = min(free_bytes, total_bytes) - reserve_bytes
    if available_bytes <= 0:
        raise RuntimeError("Available VRAM is below the configured reserve before model load")
    fraction = min(max_fraction, available_bytes / total_bytes)
    if fraction <= 0:
        raise RuntimeError("No CUDA allocator budget remains after reserving headroom")
    return fraction


def memory_headroom(free_mib, used_mib, total_mib, reserve_mib):
    """Require both CUDA and nvidia-smi views to retain the full system reserve."""
    cuda_headroom = float(free_mib)
    system_headroom = float(total_mib - used_mib)
    headroom = min(cuda_headroom, system_headroom)
    return {
        "cuda_free_mib": round(cuda_headroom, 2),
        "nvidia_smi_used_mib": int(used_mib),
        "system_free_mib": round(system_headroom, 2),
        "conservative_headroom_mib": round(headroom, 2),
        "reserve_mib": int(reserve_mib),
        "passed": headroom >= reserve_mib,
    }


def read_memory_headroom(torch, gpu_index, total_memory_mib, reserve_mib):
    free_bytes, _ = torch.cuda.mem_get_info(gpu_index)
    used, _total = query_nvidia_smi(gpu_index, ["memory.used", "memory.total"])
    if int(_total.strip()) != int(total_memory_mib):
        raise RuntimeError("GPU memory total changed during benchmark")
    return memory_headroom(free_bytes / 2**20, int(used.strip()), total_memory_mib, reserve_mib)


class GpuSampler:
    def __init__(self, gpu_index, interval=0.5):
        self.gpu_index = gpu_index
        self.interval = interval
        self.samples = []
        self.errors = []
        self.data_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None

    def _sample(self):
        try:
            util, used, total = query_nvidia_smi(
                self.gpu_index, ["utilization.gpu", "memory.used", "memory.total"]
            )
            row = {
                "utilization_percent": int(util.strip().rstrip("%")),
                "memory_used_mib": int(used.strip()),
                "memory_total_mib": int(total.strip()),
                "monotonic_seconds": time.monotonic(),
            }
            with self.data_lock:
                self.samples.append(row)
        except Exception as exc:  # A transient sample error is retained in the report.
            with self.data_lock:
                self.errors.append(f"{type(exc).__name__}: {exc}")

    def _run(self):
        while not self.stop_event.is_set():
            self._sample()
            self.stop_event.wait(self.interval)

    def start(self):
        self._sample()
        self.thread = threading.Thread(target=self._run, name="nvidia-smi-sampler", daemon=True)
        self.thread.start()

    def reset_window(self):
        with self.data_lock:
            self.samples.clear()
            self.errors.clear()
        self._sample()

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=6)
        return self.summary()

    def summary(self):
        with self.data_lock:
            samples = list(self.samples)
            errors = list(self.errors)
        utils = [row["utilization_percent"] for row in samples]
        used = [row["memory_used_mib"] for row in samples]
        return {
            "sample_count": len(samples),
            "utilization_percent_mean": round(statistics.mean(utils), 2) if utils else None,
            "utilization_percent_max": max(utils) if utils else None,
            "memory_used_mib_max": max(used) if used else None,
            "samples": samples,
            "errors": errors,
        }


def ordered_indices(encoded, mode, seed, update):
    indices = list(range(len(encoded)))
    if mode == "random":
        random.Random(seed + update).shuffle(indices)
    elif mode == "length_bucketed":
        indices.sort(key=lambda index: (len(encoded[index]["input_ids"]), index))
    else:
        raise ValueError(f"Unknown batch ordering: {mode}")
    return indices


def update_once(torch, model, optimizer, encoded, ordering, micro_batch, pad_id,
                max_grad_norm, gradient_check_mode):
    collate = Padding(pad_id)
    total_supervised = sum(
        sum(token != -100 for token in encoded[index]["labels"][1:]) for index in ordering
    )
    if total_supervised <= 0:
        raise ValueError("Synthetic benchmark update has no supervised tokens")
    real_tokens = 0
    padded_tokens = 0
    supervised_tokens = 0
    optimizer.zero_grad(set_to_none=True)
    for offset in range(0, len(ordering), micro_batch):
        selected = ordering[offset:offset + micro_batch]
        batch = collate([encoded[index] for index in selected])
        real_tokens += sum(len(encoded[index]["input_ids"]) for index in selected)
        padded_tokens += int(batch["input_ids"].numel())
        supervised_tokens += sum(
            sum(token != -100 for token in encoded[index]["labels"][1:]) for index in selected
        )
        batch = {key: value.to("cuda:0", non_blocking=True) for key, value in batch.items()}
        count = int((batch["labels"][:, 1:] != -100).sum().item())
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(**batch, use_cache=False)
            loss = outputs.loss
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite synthetic benchmark loss")
        (loss * count / total_supervised).backward()
        del outputs, loss, batch
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad and parameter.grad is not None]
    if not trainable:
        raise FloatingPointError("Missing LoRA gradients")
    if gradient_check_mode == "per_parameter_sync":
        # Mirrors the current production check; each Python bool conversion may synchronize CUDA.
        gradients_finite = not any(not torch.isfinite(parameter.grad).all() for parameter in trainable)
    elif gradient_check_mode == "aggregated_sync":
        # Same predicates, reduced on-device with one host synchronization.
        finite_flags = torch.stack([torch.isfinite(parameter.grad).all() for parameter in trainable])
        gradients_finite = bool(finite_flags.all().item())
    else:
        raise ValueError(f"Unknown gradient-check mode: {gradient_check_mode}")
    if not gradients_finite:
        raise FloatingPointError("Missing or nonfinite LoRA gradients")
    if any(parameter.grad is not None for parameter in model.parameters() if not parameter.requires_grad):
        raise ValueError("Frozen base parameter received a gradient")
    torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
    optimizer.step()
    return {
        "real_input_tokens": real_tokens,
        "padded_input_tokens": padded_tokens,
        "supervised_tokens": supervised_tokens,
    }


def restore_trainable_parameters(torch, model, initial_trainable):
    current = dict(model.named_parameters())
    with torch.no_grad():
        for name, original in initial_trainable.items():
            current[name].copy_(original.to(device=current[name].device, dtype=current[name].dtype))


def run_candidate(c, encoded, model, tokenizer, targets, initial_trainable, mode, micro_batch,
                  optimizer_mode, gradient_check_mode, device_name, total_memory_mib, gpu_index):
    import torch

    seed_everything(c["seed"])
    restore_trainable_parameters(torch, model, initial_trainable)
    model.train()
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    effective_batch = c["benchmark"]["effective_batch_examples"]
    if len(encoded) != effective_batch or effective_batch % micro_batch != 0:
        raise ValueError("Fixture count must equal the effective batch and divide evenly by micro-batch")
    optimizer_args = {
        "lr": c["benchmark"]["learning_rate"],
        "weight_decay": c["benchmark"]["weight_decay"],
    }
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    sampler = GpuSampler(gpu_index)
    sampler.start()
    optimizer = None
    try:
        if optimizer_mode == "default":
            optimizer = torch.optim.AdamW(trainable_parameters, **optimizer_args)
            optimizer_settings = {"foreach": None, "fused": None}
        elif optimizer_mode == "fused":
            optimizer = torch.optim.AdamW(trainable_parameters, foreach=False, fused=True, **optimizer_args)
            optimizer_settings = {"foreach": False, "fused": True}
        else:
            raise ValueError(f"Unknown optimizer mode: {optimizer_mode}")
        torch.cuda.reset_peak_memory_stats(0)
        torch.cuda.synchronize(0)
        warmup_counts = []
        for warmup in range(c["benchmark"]["warmup_updates"]):
            warmup_order = ordered_indices(encoded, mode, c["seed"], -1 - warmup)
            warmup_counts.append(update_once(torch, model, optimizer, encoded, warmup_order, micro_batch, pad_id,
                        c["benchmark"]["max_grad_norm"], gradient_check_mode))
        torch.cuda.synchronize(0)
        warmup_memory = read_memory_headroom(
            torch, gpu_index, total_memory_mib, c["benchmark"]["reserve_mib"]
        )
        warmup_memory["cuda_peak_allocated_mib"] = round(torch.cuda.max_memory_allocated(0) / 2**20, 2)
        warmup_memory["cuda_peak_reserved_mib"] = round(torch.cuda.max_memory_reserved(0) / 2**20, 2)
        if not warmup_memory["passed"]:
            sampler_summary = sampler.stop()
            return {
                "state": "over_memory_budget",
                "batch_order": mode,
                "micro_batch": micro_batch,
                "optimizer_mode": optimizer_mode,
                "optimizer_settings": optimizer_settings,
                "gradient_check_mode": gradient_check_mode,
                "effective_batch_examples": effective_batch,
                "warmup_updates_completed": len(warmup_counts),
                "warmup_tokens": {
                    "real_input": sum(row["real_input_tokens"] for row in warmup_counts),
                    "padded_input": sum(row["padded_input_tokens"] for row in warmup_counts),
                    "supervised": sum(row["supervised_tokens"] for row in warmup_counts),
                },
                "warmup_memory_check": warmup_memory,
                "nvidia_smi": sampler_summary,
                "memory_budget_passed": False,
                "measured_updates": 0,
                "action": "Candidate stopped immediately after warmup; keep reserved GPU headroom.",
            }
        sampler.reset_window()
        measured = []
        for update in range(c["benchmark"]["measured_updates"]):
            ordering = ordered_indices(encoded, mode, c["seed"], update)
            torch.cuda.synchronize(0)
            started = time.perf_counter()
            counts = update_once(torch, model, optimizer, encoded, ordering, micro_batch, pad_id,
                                 c["benchmark"]["max_grad_norm"], gradient_check_mode)
            torch.cuda.synchronize(0)
            counts["seconds"] = time.perf_counter() - started
            measured.append(counts)
        sampler_summary = sampler.stop()
        peak_allocated_mib = torch.cuda.max_memory_allocated(0) / 2**20
        peak_reserved_mib = torch.cuda.max_memory_reserved(0) / 2**20
        torch.cuda.synchronize(0)
        mem_free_bytes, mem_total_bytes = torch.cuda.mem_get_info(0)
        reserve_mib = c["benchmark"]["reserve_mib"]
        smi_peak = sampler_summary["memory_used_mib_max"]
        device_peak = max(peak_reserved_mib, smi_peak or 0)
        within_budget = device_peak + reserve_mib <= total_memory_mib
        elapsed = sum(row["seconds"] for row in measured)
        real = sum(row["real_input_tokens"] for row in measured)
        padded = sum(row["padded_input_tokens"] for row in measured)
        supervised = sum(row["supervised_tokens"] for row in measured)
        if sampler_summary["sample_count"] == 0:
            raise RuntimeError("nvidia-smi produced no GPU utilization samples")
        if not within_budget:
            state = "over_memory_budget"
        else:
            state = "passed"
        return {
            "state": state,
            "batch_order": mode,
            "micro_batch": micro_batch,
            "optimizer_mode": optimizer_mode,
            "optimizer_settings": optimizer_settings,
            "gradient_check_mode": gradient_check_mode,
            "gradient_check_syncs_estimate_per_update": (
                len(trainable_parameters) if gradient_check_mode == "per_parameter_sync" else 1
            ),
            "effective_batch_examples": effective_batch,
            "measured_updates": len(measured),
            "model_name": device_name,
            "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "target_module_count": len(targets),
            "tokens": {"real_input": real, "padded_input": padded, "supervised": supervised},
            "padding_fraction": round(1 - real / padded, 6) if padded else 0,
            "seconds": round(elapsed, 4),
            "real_input_tokens_per_second": round(real / elapsed, 2) if elapsed else None,
            "padded_input_tokens_per_second": round(padded / elapsed, 2) if elapsed else None,
            "supervised_tokens_per_second": round(supervised / elapsed, 2) if elapsed else None,
            "update_seconds": [round(row["seconds"], 4) for row in measured],
            "cuda_peak_allocated_mib": round(peak_allocated_mib, 2),
            "cuda_peak_reserved_mib": round(peak_reserved_mib, 2),
            "nvidia_smi": sampler_summary,
            "gpu_utilization_window": "measured updates only; CUDA synchronized at timing boundaries",
            "gpu_memory_total_mib": total_memory_mib,
            "gpu_memory_free_after_candidate_mib": round(mem_free_bytes / 2**20, 2),
            "gpu_reserve_mib": reserve_mib,
            "memory_budget_passed": within_budget,
        }
    except torch.cuda.OutOfMemoryError as exc:
        sampler_summary = sampler.stop()
        return {
            "state": "oom_unavailable",
            "batch_order": mode,
            "micro_batch": micro_batch,
            "optimizer_mode": optimizer_mode,
            "optimizer_settings": optimizer_settings if "optimizer_settings" in locals() else None,
            "gradient_check_mode": gradient_check_mode,
            "effective_batch_examples": effective_batch,
            "nvidia_smi": sampler_summary,
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        sampler.stop()
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            del optimizer
        gc.collect()
        torch.cuda.empty_cache()


def run_benchmark(c, config_path, gpu_index=0):
    benchmark = c["benchmark"]
    threads = benchmark["cpu_threads"]
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(threads))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch

    output_dir = c["report"]
    output_dir.mkdir(parents=True, exist_ok=True)
    if any((output_dir / name).exists() for name in ("benchmark.json", "status.json", "failure.json")):
        raise FileExistsError("Benchmark output already exists; choose a fresh benchmark output directory")
    status_path = output_dir / "status.json"
    started = time.perf_counter()
    result = {
        "benchmark_only": True,
        "quality_evaluation": False,
        "training_approval_used": False,
        "dataset_read": False,
        "model_or_adapter_saved": False,
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256(config_path),
        "code_sha256": sha256(Path(__file__)),
        "started_utc": utc(),
        "python": __import__("sys").version,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cpu_threads_requested": benchmark["cpu_threads"],
        "cpu_threads_actual": torch.get_num_threads(),
        "cpu_interop_threads_actual": torch.get_num_interop_threads(),
        "optimizer_sweep": benchmark["optimizer_modes"],
        "gradient_check_sweep": benchmark["gradient_check_modes"],
        "effective_batch_examples": benchmark["effective_batch_examples"],
        "warmup_updates": benchmark["warmup_updates"],
        "measured_updates": benchmark["measured_updates"],
        "candidate_micro_batches": benchmark["micro_batch_candidates"],
        "batch_orders": benchmark["batch_orders"],
        "fixtures": {},
        "candidates": [],
    }

    def write_status(state, **fields):
        write_json(status_path, {
            "state": state,
            "pid": os.getpid(),
            "updated_utc": utc(),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            **fields,
        })

    write_status("running", stage="gpu_preflight")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to run a CPU performance substitute")
    if shutil.which("nvidia-smi") is None:
        raise RuntimeError("nvidia-smi is required for measured GPU utilization and memory telemetry")
    gpu_name, gpu_total = query_nvidia_smi(gpu_index, ["name", "memory.total"])
    gpu_name = gpu_name.strip()
    total_memory_mib = int(gpu_total.strip())
    torch_name = torch.cuda.get_device_name(0)
    if gpu_name != torch_name:
        raise RuntimeError(f"CUDA device and nvidia-smi GPU identity differ: {torch_name!r} vs {gpu_name!r}")
    if total_memory_mib <= benchmark["reserve_mib"]:
        raise RuntimeError("GPU has no usable VRAM after the configured reserve")
    verification_started = time.perf_counter()
    verify_model(c["model"], c["revision"])
    verification_seconds = time.perf_counter() - verification_started
    free_before_model, cuda_total_bytes = torch.cuda.mem_get_info(0)
    smi_used, _ = query_nvidia_smi(gpu_index, ["memory.used", "memory.total"])
    before_model_headroom = memory_headroom(
        free_before_model / 2**20, int(smi_used.strip()), total_memory_mib, benchmark["reserve_mib"]
    )
    result["memory_preflight"] = {"before_model_load": before_model_headroom}
    result["gpu"] = {
        "name": gpu_name,
        "index": gpu_index,
        "total_memory_mib": total_memory_mib,
        "reserve_mib": benchmark["reserve_mib"],
    }
    if not before_model_headroom["passed"]:
        write_json(output_dir / "benchmark.json", {**result, "state": "stopped_no_headroom_before_model_load"})
        raise RuntimeError("Available VRAM is below the configured reserve before model load.")
    try:
        allocator_fraction = allocator_fraction_for_budget(
            free_before_model, cuda_total_bytes, benchmark["reserve_mib"],
            benchmark["allocator_max_memory_fraction"],
        )
    except BaseException as exc:
        write_json(output_dir / "benchmark.json", {**result, "state": "stopped_allocator_budget_preflight"})
        raise
    torch.cuda.set_per_process_memory_fraction(allocator_fraction, 0)
    result["memory_preflight"].update({
        "allocator_fraction_cap": round(allocator_fraction, 6),
        "allocator_fraction_configured_max": benchmark["allocator_max_memory_fraction"],
        "cuda_total_mib": round(cuda_total_bytes / 2**20, 2),
        "reserve_mib": benchmark["reserve_mib"],
    })
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(benchmark["cpu_interop_threads"])
    result["cpu_threads_actual"] = torch.get_num_threads()
    result["cpu_interop_threads_actual"] = torch.get_num_interop_threads()
    model_load_started = time.perf_counter()
    seed_everything(c["seed"])
    try:
        model, tokenizer, targets = make_model(c)
    except torch.cuda.OutOfMemoryError as exc:
        write_json(output_dir / "benchmark.json", {**result, "state": "stopped_model_load_oom"})
        raise RuntimeError(
            "Base/LoRA model allocation exceeded the capped CUDA allocator budget; "
            "stopping before any benchmark candidate."
        ) from exc
    model_load_seconds = time.perf_counter() - model_load_started
    post_model_memory = read_memory_headroom(
        torch, gpu_index, total_memory_mib, benchmark["reserve_mib"]
    )
    result["memory_preflight"]["after_model_load"] = post_model_memory
    if not post_model_memory["passed"]:
        write_json(output_dir / "benchmark.json", {**result, "state": "stopped_no_headroom_after_model_load"})
        raise RuntimeError(
            "Model load left less than the configured VRAM reserve; no benchmark candidate was started."
        )
    initial_trainable = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    fixture_started = time.perf_counter()
    rows, encoded = encode_fixtures(c, tokenizer)
    fixture_seconds = time.perf_counter() - fixture_started
    lengths = [len(row["input_ids"]) for row in encoded]
    result["fixtures"] = {
        "source": "fixed synthetic messages authored in benchmark_training_pipeline.py; no dataset rows loaded",
        "count": len(rows),
        "role_pattern_validated": True,
        "native_template_encoded": True,
        "max_length": c["max_length"],
        "length_min": min(lengths),
        "length_median": statistics.median(lengths),
        "length_max": max(lengths),
        "input_tokens_total": sum(lengths),
        "supervised_tokens_total": sum(sum(token != -100 for token in row["labels"]) for row in encoded),
        "encoding_seconds": round(fixture_seconds, 4),
        "model_loading_seconds": round(model_load_seconds, 3),
        "model_checksum_verification_seconds": round(verification_seconds, 3),
        "target_module_count": len(targets),
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "sha256": __import__("hashlib").sha256(
            json.dumps(rows, ensure_ascii=False, sort_keys=True).encode("utf8")
        ).hexdigest(),
    }
    write_status("running", stage="synthetic_fixture_encoded", gpu=gpu_name, fixture_count=len(rows))

    jobs = []
    for mode in benchmark["batch_orders"]:
        for micro_batch in benchmark["micro_batch_candidates"]:
            jobs.append((mode, micro_batch, "default", "per_parameter_sync"))
    for optimizer_mode in benchmark["optimizer_modes"]:
        for gradient_check_mode in benchmark["gradient_check_modes"]:
            jobs.append((benchmark["ablation_batch_order"], benchmark["ablation_micro_batch"],
                         optimizer_mode, gradient_check_mode))
    ordered_jobs = []
    seen_jobs = set()
    for job in jobs:
        if job not in seen_jobs:
            ordered_jobs.append(job)
            seen_jobs.add(job)
    try:
        for mode, micro_batch, optimizer_mode, gradient_check_mode in ordered_jobs:
            write_status("running", stage="candidate", batch_order=mode, micro_batch=micro_batch,
                         optimizer_mode=optimizer_mode, gradient_check_mode=gradient_check_mode)
            candidate_started = time.perf_counter()
            try:
                candidate = run_candidate(
                    c, encoded, model, tokenizer, targets, initial_trainable, mode, micro_batch,
                    optimizer_mode, gradient_check_mode, gpu_name, total_memory_mib, gpu_index,
                )
            except torch.cuda.OutOfMemoryError as exc:
                gc.collect()
                torch.cuda.empty_cache()
                candidate = {
                    "state": "oom_unavailable",
                    "batch_order": mode,
                    "micro_batch": micro_batch,
                    "optimizer_mode": optimizer_mode,
                    "gradient_check_mode": gradient_check_mode,
                    "effective_batch_examples": benchmark["effective_batch_examples"],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            candidate["total_candidate_seconds"] = round(time.perf_counter() - candidate_started, 3)
            result["candidates"].append(candidate)
            write_json(output_dir / "benchmark.json", {**result, "updated_utc": utc()})
            if candidate["state"] not in ("passed", "over_memory_budget", "oom_unavailable"):
                raise RuntimeError(f"Unexpected benchmark state: {candidate['state']}")
    finally:
        del model, tokenizer, initial_trainable
        gc.collect()
        torch.cuda.empty_cache()
    result["finished_utc"] = utc()
    result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    result["selection"] = {
        "policy": "highest measured supervised-token throughput among candidates that pass the VRAM reserve",
        "best_candidate": None,
    }
    eligible = [row for row in result["candidates"] if row.get("state") == "passed"]
    if eligible:
        result["selection"]["best_candidate"] = max(
            eligible, key=lambda row: row.get("supervised_tokens_per_second", 0)
        )
    write_json(output_dir / "benchmark.json", result)
    if not eligible:
        raise RuntimeError("All benchmark candidates failed or exceeded the VRAM reserve")
    lines=["# Qwen3.5-2B 流水线性能验证", "",
           "性能实验已完成。使用固定合成输入，不读取候选训练集，不生成交付模型；这些数据不代表任务正确率。", "",
           "| 优化器 | 梯度检查 | 微批次 | 状态 | 监督 Token/s | GPU平均占用 | 峰值显存 MiB |",
           "|---|---|---:|---|---:|---:|---:|"]
    for candidate in result['candidates']:
        telemetry=candidate.get('nvidia_smi',{})
        lines.append(f"| {candidate['optimizer_mode']} | {candidate['gradient_check_mode']} | "
                     f"{candidate['micro_batch']} | {candidate['state']} | "
                     f"{candidate.get('supervised_tokens_per_second','—')} | "
                     f"{telemetry.get('utilization_percent_mean','—')} | "
                     f"{telemetry.get('memory_used_mib_max','—')} |")
    lines.extend(['',f"总耗时：{result['elapsed_seconds']:.1f} 秒。每次更新使用 {benchmark['effective_batch_examples']} 个合成样本。",
                  '该吞吐测试不估计正式训练耗时；正式配置的有效批次及数据长度不同。',
                  '完整输入、配置摘要、显存余量和自动采样记录见 benchmark.json。'])
    (output_dir/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    write_status("completed", benchmark_path=str(output_dir / "benchmark.json"),
                 eligible_candidates=len(eligible))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/linux-assistant-qwen35-2b-benchmark-20261022.json"))
    parser.add_argument("--gpu-index", type=int, default=0)
    args = parser.parse_args()
    c = config(args.config)
    lock_path = WORKSPACE / "runs/linux-assistant-gpu.lock"
    try:
        with exclusive_job(lock_path):
            result = run_benchmark(c, args.config, args.gpu_index)
            print(json.dumps({
                "state": "completed",
                "report": str(c["report"] / "benchmark.json"),
                "gpu": result.get("gpu"),
                "best_candidate": result.get("selection", {}).get("best_candidate"),
            }, ensure_ascii=False), flush=True)
    except BaseException as error:
        report = c["report"]
        report.mkdir(parents=True, exist_ok=True)
        existing_status = None
        if (report / "status.json").is_file():
            try:
                existing_status = json.loads((report / "status.json").read_text(encoding="utf8"))
            except Exception:
                existing_status = None
        if existing_status is None or existing_status.get("state") == "running":
            write_json(report / "failure.json", {
                "state": "failed",
                "error": str(error),
                "type": type(error).__name__,
                "traceback": traceback.format_exc(),
                "utc": utc(),
                "action": "Stop; no training or automatic configuration change was performed.",
            })
            write_json(report / "status.json", {"state": "failed", "error": str(error), "updated_utc": utc()})
        raise


if __name__ == "__main__":
    main()
