"""仅在下一次原子保存后暂停已核对进程，保留状态和账本，再限时短测。"""
from scripts.common import RUNS, ROOT, read_json, write_json, utc, sha256
from training.train import load_checkpoint
from pathlib import Path
from datetime import datetime, timezone
import argparse
import os
import shutil
import subprocess
import sys
import time
import psutil


def checkpoint_is_newer(checkpoint_mtime_ns, baseline_mtime_ns):
    """Use filesystem nanoseconds and a captured baseline, never second buckets."""
    return int(checkpoint_mtime_ns) > int(baseline_mtime_ns)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--after-utc',required=True)
    args=p.parse_args()
    after=datetime.fromisoformat(args.after_utc).timestamp()
    folder=RUNS/'checkpoints/50m-base-v2';latest=folder/'latest.pt'
    if not latest.is_file():raise FileNotFoundError(latest)
    # Compare against the file identity observed when this watcher starts. The
    # previous whole-second comparison could mistake the old checkpoint for a
    # new save when both timestamps fell within the same second.
    baseline_mtime_ns=max(latest.stat().st_mtime_ns,int(after*1_000_000_000))
    original=read_json(RUNS/'reports/pipeline_status.json')
    if original['phase']!='pretrain-50m-v2':raise RuntimeError('not currently in base training')
    expected_pid,expected_child=original['pid'],original['child_pid']
    deadline=time.monotonic()+35*60
    while not checkpoint_is_newer(latest.stat().st_mtime_ns,baseline_mtime_ns):
        current=read_json(RUNS/'reports/pipeline_status.json')
        if (current.get('pid'),current.get('child_pid'),current.get('phase'))!=(expected_pid,expected_child,'pretrain-50m-v2'):
            raise RuntimeError('pipeline changed while waiting; reassess before stopping')
        if time.monotonic()>deadline:raise TimeoutError('next save did not arrive')
        print('WAITING_FOR_ATOMIC_SAVE',utc(),flush=True);time.sleep(20)
    saved=load_checkpoint(latest)
    if saved['settings']['stage']!='pretrain' or saved['input_tokens']<original.get('input_tokens',0):raise RuntimeError('unexpected checkpoint')
    backup=RUNS/'checkpoints/performance-switch';backup.mkdir(parents=True,exist_ok=True)
    source=backup/'source.pt'
    if source.exists():raise RuntimeError('switch source already exists; inspect prior attempt')
    shutil.copy2(latest,source)
    digest=sha256(source)
    if digest!=sha256(latest):raise RuntimeError('source changed while copying')
    shutil.copy2(folder/'validation_latest.json',backup/'validation_latest.json')
    processes=[psutil.Process(expected_pid),psutil.Process(expected_child)]
    if 'scripts.train_chat_v2' not in processes[0].cmdline() or 'training.train' not in processes[1].cmdline():raise RuntimeError('PID command mismatch')
    if str(folder) not in processes[1].cmdline():raise RuntimeError('training output mismatch')
    # 控制器先退出，避免训练终止后自动恢复或切入 probe。
    for process in processes:
        process.terminate();process.wait(timeout=20)
    if any(psutil.pid_exists(pid) for pid in (expected_pid,expected_child)):raise RuntimeError('process still exists')
    ledger_path=RUNS/'reports/gpu_budget.json';ledger=read_json(ledger_path)
    if not ledger or ledger[-1].get('finished') or ledger[-1]['stage']!='pretrain-LubanLM-50M':raise RuntimeError('unexpected GPU ledger owner')
    now=datetime.now(timezone.utc)
    started=datetime.fromisoformat(ledger[-1]['started_utc'])
    ledger[-1].update(seconds=max(ledger[-1]['seconds'],(now-started).total_seconds()),finished=True,ended_utc=now.isoformat(),error=None,stop_reason='authorized_performance_switch')
    write_json(ledger_path,ledger)
    pipeline_lock=RUNS/'pipeline.lock'
    if pipeline_lock.exists():
        if pipeline_lock.read_text().strip()!=str(expected_pid):raise RuntimeError('pipeline lock owner mismatch')
        pipeline_lock.unlink()
    ledger_path.with_suffix('.lock').unlink(missing_ok=True)
    metrics=folder/'metrics.jsonl'
    json_last=None
    import json
    for line in metrics.read_text(encoding='utf-8').splitlines():
        try:json_last=json.loads(line)
        except json.JSONDecodeError:pass
    receipt=dict(created_utc=utc(),old_controller_pid=expected_pid,old_child_pid=expected_child,source=str(source),sha256=digest,restored_step=saved['step'],restored_tokens=saved['supervised_tokens'],last_logged_tokens=json_last['supervised_tokens'],discarded_logged_tokens=max(0,json_last['supervised_tokens']-saved['supervised_tokens']),gpu_budget_preserved=True)
    write_json(RUNS/'reports/performance_switch.json',receipt)
    command=[sys.executable,'-u','-m','scripts.benchmark_training','--checkpoint',str(source),'--minutes','20']
    log=RUNS/'logs/performance-benchmark.log'
    with log.open('a',encoding='utf-8') as stream:
        process=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        write_json(RUNS/'reports/pipeline_status.json',dict(phase='performance_benchmark',pid=os.getpid(),child_pid=process.pid,command=command,log=str(log),updated_utc=utc(),capability_status='not_accepted'))
        print('BENCHMARK_STARTED',process.pid,'CHECKPOINT_STEP',saved['step'],flush=True)
        code=process.wait()
    receipt.update(benchmark_exit_code=code,benchmark_completed_utc=utc());write_json(RUNS/'reports/performance_switch.json',receipt)
    print('BENCHMARK_COMPLETED',code,flush=True)
    if code:raise RuntimeError('benchmark failed; inspect log and resume baseline from source')

if __name__=='__main__':main()
