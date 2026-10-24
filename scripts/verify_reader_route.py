"""Run the offline reader route and compare continuous/resumed state exactly."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        p.error("audit output must be empty")
    output.mkdir(parents=True, exist_ok=True)
    tutorial = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(tutorial), PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1")
    wall = {}
    for name, extra in [("part", ["--stop-after", "8"]),
                        ("resumed", ["--resume", str(output/"part/last.pt"), "--stop-after", "16"]),
                        ("continuous", ["--stop-after", "16"])]:
        command = [sys.executable, "-B", "-X", "utf8", "-m", "training.reader_lab", "--output", str(output/name), *extra]
        started = time.perf_counter()
        result = subprocess.run(command, cwd=tutorial, env=env, capture_output=True, text=True, encoding="utf8")
        wall[name] = time.perf_counter()-started
        (output/f"{name}.stdout.txt").write_text(result.stdout, encoding="utf8")
        (output/f"{name}.stderr.txt").write_text(result.stderr, encoding="utf8")
        result.check_returncode()
    a = torch.load(output/"resumed/last.pt", map_location="cpu", weights_only=False)
    b = torch.load(output/"continuous/last.pt", map_location="cpu", weights_only=False)
    checks = {key: equal(a[key], b[key]) for key in b}
    r = json.loads((output/"resumed/result.json").read_text(encoding="utf8"))
    c = json.loads((output/"continuous/result.json").read_text(encoding="utf8"))
    checks["generated_ids"] = r["generated_ids"] == c["generated_ids"]
    checks["resumed_only_eight_updates"] = r["updates_this_run"] == 8 and r["step"] == 16
    checks["tokenizer_bytes"] = (output/"resumed/tokenizer.json").read_bytes() == (output/"continuous/tokenizer.json").read_bytes()
    summary = dict(tutorial=str(tutorial), interpreter=sys.executable, torch_version=torch.__version__,
        subprocess_wall_seconds=wall, checks=checks, passed=all(checks.values()),
        resumed_result={key: value for key, value in r.items() if key != "history"})
    (output/"audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n", encoding="utf8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["passed"]:
        raise RuntimeError("reader route state comparison failed")


if __name__ == "__main__":
    main()
