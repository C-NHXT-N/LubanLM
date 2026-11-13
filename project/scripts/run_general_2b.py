"""准备新语料并将可信Base续训至累计2B；独立有界账本，保留历史记录。"""
from __future__ import annotations
from scripts.common import ROOT, read_json, write_json, sha256, utc
import argparse
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np


def verify_manifest(path, tokenizer_hash, min_shard_tokens=257):
    manifest = read_json(path)
    if manifest['dtype'] != 'uint16' or manifest['tokenizer_sha256'] != tokenizer_hash:
        raise ValueError('manifest codec/tokenizer mismatch')
    counted = 0
    for row in manifest['shards']:
        shard = Path(row['path'])
        if shard.stat().st_size != 2 * row['tokens'] or sha256(shard) != row['sha256']:
            raise ValueError(f'shard hash/length mismatch: {shard}')
        ids = np.memmap(shard, mode='r', dtype=np.uint16)
        if len(ids) < min_shard_tokens or int(ids.max()) >= 32000 or not np.any(ids == 2):
            raise ValueError(f'shard token range/EOS invalid: {shard}')
        counted += len(ids)
        del ids
    if counted != manifest['tokens']:
        raise ValueError('manifest token total mismatch')
    return dict(path=str(path), sha256=sha256(path), tokens=counted,
                source_tokens=manifest.get('source_tokens'), checked_all_shards=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=ROOT/'configs/50m-general-pretrain-2b.json')
    args = parser.parse_args()
    args.plan = args.plan.resolve()
    plan = read_json(args.plan)
    runs, data = Path(plan['runs_root']), Path(plan['data_root'])
    runs.mkdir(parents=True, exist_ok=True)
    report = runs / 'reports/controller.json'
    lock = runs / 'controller.lock'
    # x 模式只创建自己的锁，不删除未知进程的锁。
    with lock.open('x', encoding='utf-8') as handle:
        handle.write(str(os.getpid())); handle.flush()
    helper = None
    started = time.monotonic()
    state = dict(pid=os.getpid(), started_utc=utc(), plan=str(args.plan),
                 plan_sha256=sha256(args.plan), status='starting',
                 cumulative_target=plan['target_cumulative_tokens'],
                 historical_gpu_minutes=881.3897566833335,
                 new_gpu_minutes_cap=plan['fresh_gpu_budget_minutes'],
                 capability_status='not_accepted')
    def status(value, **details):
        state.update(status=value, updated_utc=utc(), **details)
        write_json(report, state)
        print(json.dumps(state, ensure_ascii=False), flush=True)
    def run(module, arguments, label):
        status(label)
        subprocess.run([sys.executable, '-u', '-m', module, *arguments], cwd=ROOT, env=env, check=True)
    import json
    env = os.environ.copy()
    env.update(LUBAN_DATA=str(data), LUBAN_RUNS=str(runs),
               LUBAN_GPU_BUDGET_MINUTES=str(plan['fresh_gpu_budget_minutes']),
               CUDA_VISIBLE_DEVICES='GPU-c50d3c54-c5d3-f0da-7fa4-a48d195a849f',
               PYTHONIOENCODING='utf-8', TOKENIZERS_PARALLELISM='false')
    try:
        if os.name == 'nt':
            helper = subprocess.Popen([sys.executable, '-m', 'scripts.keep_awake', '--pid', str(os.getpid())],
                                      cwd=ROOT, env=env, creationflags=subprocess.CREATE_NO_WINDOW)
        old_ledger = Path('D:/LubanLM-Runs/chat-v2/reports/gpu_budget.json')
        state['historical_ledger_sha256'] = sha256(old_ledger)
        state['historical_ledger'] = str(old_ledger)
        # 仅检查本地项目可信checkpoint，不下载或加载任何远端模型权重。
        import torch
        parent = torch.load(plan['parent_checkpoint'], map_location='cpu', weights_only=False)
        parent_tokens = parent.get('cumulative_pretrain_tokens', parent['supervised_tokens'])
        if parent['settings']['stage'] != 'pretrain' or parent_tokens != plan['parent_tokens'] or parent['tokenizer_sha256'] != plan['tokenizer_sha256']:
            raise ValueError('parent Base lineage/tokenizer mismatch')
        state['parent_checkpoint_sha256'] = sha256(plan['parent_checkpoint'])
        state['parent_tokens'] = parent_tokens
        del parent
        run('scripts.prepare_general_2b', ['--plan', str(args.plan)], 'preparing_cpu_data')
        prepared = read_json(runs/'reports/preparation_complete.json')
        if prepared['plan_sha256'] != state['plan_sha256']:
            raise ValueError('prepared plan mismatch')
        status('verifying_all_shards')
        manifests = {s: verify_manifest(data/f'processed/pretrain-{s}/manifest.json', plan['tokenizer_sha256']) for s in ('train', 'validation')}
        for split, result in manifests.items():
            if result['sha256'] != prepared['manifests'][split]['sha256']:
                raise ValueError('prepared manifest changed')
        if prepared['new_train_tokens'] < plan['minimum_new_train_tokens']:
            raise ValueError('new corpus minimum not met')
        write_json(runs/'reports/pretraining_data_audit.json', dict(created_utc=utc(), manifests=manifests,
            fixed_source_revisions=True, download_hashes_verified=True, tokenizer_unchanged=True,
            validation_is_not_chat_acceptance=True, factual_accuracy_not_guaranteed=True))
        output = runs/'checkpoints/50m-general-base-2b'
        remaining_tokens = plan['target_cumulative_tokens'] - parent_tokens
        if remaining_tokens <= 0:
            raise ValueError('cumulative target already reached')
        t = plan['training']
        training_args = ['--config', str(ROOT/plan['model_config']), '--stage', 'pretrain',
                         '--output', str(output), '--minutes', str(plan['pretrain_minutes_cap']),
                         '--tokens', str(remaining_tokens), '--execution', 'optimized',
                         '--attention-backend', 'sdpa', '--optimizer-backend', 'fused',
                         '--pin-memory', '--validation-micro-batch', '2']
        for name, value in t.items():
            training_args += ['--'+name.replace('_', '-'), str(value)]
        if (output/'last.pt').exists():
            training_args += ['--resume', str(output/'last.pt')]
        elif output.exists() and any(output.iterdir()):
            raise RuntimeError('output incomplete without last.pt: inspect checkpoint before resuming')
        else:
            training_args += ['--continue-from', plan['parent_checkpoint']]
        write_json(runs/'reports/training_invocation.json', dict(module='training.train', arguments=training_args,
            requested_additional_tokens=remaining_tokens,
            rounded_cumulative_target=parent_tokens+math.ceil(remaining_tokens/(t['length']*t['micro_batch']*t['accumulation']))*(t['length']*t['micro_batch']*t['accumulation']),
            plan_sha256=state['plan_sha256']))
        run('training.train', training_args, 'training_pretrain')
        summary = read_json(output/'summary.json')
        ledger = read_json(runs/'reports/gpu_budget.json')
        new_minutes = sum(r['seconds'] for r in ledger)/60
        reached = summary['cumulative_pretrain_tokens'] >= plan['target_cumulative_tokens']
        status('base_target_reached_sft_pending' if reached else 'base_stopped_before_target',
               actual_cumulative_pretrain_tokens=summary['cumulative_pretrain_tokens'],
               stop_reason=summary['stop_reason'], checkpoint=str(output/'last.pt'),
               new_used_gpu_minutes=new_minutes,
               combined_used_gpu_minutes=state['historical_gpu_minutes']+new_minutes)
        if sha256(old_ledger) != state['historical_ledger_sha256']:
            raise RuntimeError('historical ledger changed during independent run')
    except BaseException as exc:
        status('failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        state['controller_elapsed_seconds'] = round(time.monotonic()-started)
        write_json(report, state)
        lock.unlink(missing_ok=True)
        if helper is not None:
            helper.terminate()
            helper.wait(timeout=10)


if __name__ == '__main__':
    main()
