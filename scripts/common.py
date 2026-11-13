"""路径、可复现记录与累计 GPU 时间账本。导入本模块应先于 torch。"""
from __future__ import annotations
import hashlib
import json
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
DATA = Path(os.environ.get('LUBAN_DATA', WORKSPACE / 'datasets/general-10b-teaching-v2'))
RUNS = Path(os.environ.get('LUBAN_RUNS', WORKSPACE / 'runs'))
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ.setdefault('HF_HOME', str(WORKSPACE / '.cache/huggingface'))
os.environ.setdefault('HF_DATASETS_CACHE', str(WORKSPACE / '.cache/datasets'))
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()

def utc():
    return datetime.now(timezone.utc).isoformat()

class GPUBudget:
    """账本锁阻止并行 GPU 实验；默认五秒持久化，退出时结算全部耗时。"""
    def __init__(self, stage, minutes, heartbeat_seconds=5):
        self.stage = stage
        self.limit = minutes * 60
        self.heartbeat_seconds = heartbeat_seconds
        self.last_write = float('-inf')
        self.path = RUNS / 'reports/gpu_budget.json'
        self.lock = self.path.with_suffix('.lock')

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.lock.open('x', encoding='utf-8')
        self.handle.write(str(os.getpid()))
        self.handle.flush()
        self.entries = read_json(self.path) if self.path.exists() else []
        # 未结束记录保留最后一次 heartbeat，必须先检查上次进程状态再清理 lock。
        used = sum(row['seconds'] for row in self.entries)
        cumulative_minutes = float(os.environ.get('LUBAN_GPU_BUDGET_MINUTES', '240'))
        if not 0 < cumulative_minutes <= 10080:
            self.handle.close()
            self.lock.unlink()
            raise ValueError('LUBAN_GPU_BUDGET_MINUTES must be in (0, 10080]')
        self.allowed = min(self.limit, max(0, cumulative_minutes * 60 - used))
        if self.allowed <= 0:
            self.handle.close()
            self.lock.unlink()
            raise RuntimeError('GPU cumulative budget exhausted')
        self.start = time.monotonic()
        self.entries.append({'stage': self.stage, 'started_utc': utc(), 'seconds': 0, 'finished': False})
        self.tick()
        return self

    def tick(self, force=False):
        elapsed = time.monotonic() - self.start
        self.entries[-1]['seconds'] = round(elapsed, 3)
        if force or elapsed - self.last_write >= self.heartbeat_seconds or elapsed >= self.allowed:
            write_json(self.path, self.entries)
            self.last_write = elapsed
        return elapsed >= self.allowed

    def remaining(self):
        return max(0, self.allowed - (time.monotonic() - self.start))

    def __exit__(self, kind, value, trace):
        self.tick(force=True)
        self.entries[-1].update(finished=True, ended_utc=utc(), error=None if kind is None else str(value))
        write_json(self.path, self.entries)
        self.handle.close()
        self.lock.unlink()

def require_gpu():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; do not silently fall back to CPU')
    if torch.cuda.device_count() < 1:
        raise RuntimeError('No visible CUDA device; set CUDA_VISIBLE_DEVICES explicitly')
    properties = torch.cuda.get_device_properties(0)
    if properties.total_memory <= 0:
        raise RuntimeError('Visible CUDA device has no usable memory')
    # A visible device name does not prove that the installed wheel has working kernels.
    probe = torch.ones(1, device='cuda:0')
    probe.add_(1)
    torch.cuda.synchronize(0)
    if probe.item() != 2:
        raise RuntimeError('CUDA computation check failed')
    return 'cuda:0'
