"""Honor a one-shot user pause after the next atomically committed checkpoint."""
import argparse
from datetime import datetime, timezone
import json
import time
from pathlib import Path

import psutil

from scripts.common import read_json, write_json, sha256, utc


def matching(pid, created, module):
    process = psutil.Process(pid)
    if abs(process.create_time() - created) > .1 or module not in process.cmdline():
        raise RuntimeError('Process identity changed; refusing to stop it')
    return process


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    args = parser.parse_args()
    request = read_json(args.request)
    reports = args.request.resolve().parent
    marker = reports / 'maintenance.json'
    checkpoint = Path(request['checkpoint'])
    owner = request['request_id']

    def publish(status, **extra):
        if read_json(marker).get('request_id') != owner:
            raise RuntimeError('Pause request ownership changed')
        request.update(status=status, updated_utc=utc(), **extra)
        write_json(args.request, request)
        write_json(marker, request)

    try:
        publish('waiting_for_next_checkpoint', watcher_pid=psutil.Process().pid,
                watcher_created=psutil.Process().create_time())
        deadline = time.monotonic() + 75 * 60
        while time.monotonic() < deadline:
            if read_json(marker).get('request_id') != owner:
                raise RuntimeError('Pause request ownership changed')
            controller = matching(request['controller_pid'], request['controller_created'], 'scripts.run_general_10b')
            trainer = matching(request['trainer_pid'], request['trainer_created'], 'training.train')
            stat = checkpoint.stat()
            if stat.st_mtime_ns != request['baseline_checkpoint_mtime_ns'] and stat.st_size > 0:
                # The trainer fsyncs a .tmp file, then atomically replaces this path.
                # No checkpoint is read or stopped while it is being written.
                if not checkpoint.with_name(checkpoint.name + '.tmp').exists():
                    break
            time.sleep(1)
        else:
            raise TimeoutError('No new checkpoint within 75 minutes; no process stopped')
        state = read_json(reports / 'controller.json')
        if (state.get('pid'), state.get('child_pid')) != (controller.pid, trainer.pid):
            raise RuntimeError('Controller worker changed; refusing to stop it')
        publish('pausing', checkpoint_bytes=stat.st_size, checkpoint_mtime_ns=stat.st_mtime_ns)
        trainer.suspend()
        try:
            controller.terminate()
            controller.wait(timeout=15)
            trainer.terminate()
            trainer.wait(timeout=15)
        finally:
            if trainer.is_running():
                trainer.resume()
        paused_utc = utc()
        ledger_path = reports / 'gpu_budget.json'
        ledger = read_json(ledger_path)
        if ledger and not ledger[-1]['finished']:
            entry = ledger[-1]
            if entry['stage'] != 'pretrain-LubanLM-50M':
                raise RuntimeError('Unexpected GPU budget stage')
            seconds = (datetime.fromisoformat(paused_utc) - datetime.fromisoformat(entry['started_utc'])).total_seconds()
            entry.update(seconds=max(entry['seconds'], round(seconds, 3)), finished=True,
                         ended_utc=paused_utc, error=None, stop_reason='user_pause_after_checkpoint')
            write_json(ledger_path, ledger)
        state.update(status='paused_by_user', updated_utc=utc(), pause_checkpoint=str(checkpoint),
                     pause_report=str(reports / 'paused.json'))
        state.pop('child_pid', None)
        state.pop('child_created', None)
        write_json(reports / 'controller.json', state)
        supervisor = read_json(reports / 'supervisor.json')
        supervisor.update(status='paused_by_user', failed_without_progress=0, updated_utc=utc())
        write_json(reports / 'supervisor.json', supervisor)
        publish('paused_by_user', paused_utc=paused_utc, processes_exited=True, checkpoint_readable=False)
        write_json(reports / 'paused.json', request)
        # Verify only after the GPU worker exits, to avoid competing for RAM.
        import torch
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
        required = ['model', 'optimizer', 'step', 'input_tokens', 'supervised_tokens', 'sampler',
                    'settings', 'dataset', 'data_rng', 'torch_rng', 'cuda_rng', 'python_rng',
                    'numpy_rng', 'tokenizer_sha256']
        if not all(key in saved and saved[key] is not None for key in required):
            raise RuntimeError('Checkpoint resume states missing')
        publish('paused_by_user', checkpoint_readable=True, checkpoint_sha256=sha256(checkpoint),
                saved_step=saved['step'], saved_input_tokens=saved['input_tokens'],
                saved_supervised_tokens=saved['supervised_tokens'],
                saved_cumulative_pretrain_tokens=saved['cumulative_pretrain_tokens'],
                resume_states_present=required)
        write_json(reports / 'paused.json', request)
        print(json.dumps(request), flush=True)
    except BaseException as exc:
        request.update(status='pause_failed', error=f'{type(exc).__name__}: {exc}', updated_utc=utc())
        write_json(args.request, request)
        if marker.exists() and read_json(marker).get('request_id') == owner:
            write_json(marker, request)
        raise


if __name__ == '__main__':
    main()
