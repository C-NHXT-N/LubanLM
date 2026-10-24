"""在独立模型副本上测物理batch吞吐；记GPU账本，不改父checkpoint。"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=Path('configs/50m-general-pretrain-2b.json'))
    parser.add_argument('--batches', type=int, nargs='+', default=[8, 16, 32, 48, 64])
    parser.add_argument('--steps', type=int, default=25)
    parser.add_argument('--report-name', default='throughput_benchmark.json')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding='utf-8'))
    os.environ.update(LUBAN_DATA=plan['data_root'], LUBAN_RUNS=plan['runs_root'],
                      LUBAN_GPU_BUDGET_MINUTES=str(plan['fresh_gpu_budget_minutes']),
                      CUDA_VISIBLE_DEVICES='GPU-c50d3c54-c5d3-f0da-7fa4-a48d195a849f',
                      TOKENIZERS_PARALLELISM='false')
    from scripts.common import ROOT, RUNS, GPUBudget, require_gpu, sha256, write_json, utc
    from model import LubanLM, LubanConfig
    from training.batches import PackedTokens,FullPassTokens
    from training.execution import update, BatchTransfer, configure_model, configure_optimizer
    from training.train import validation
    import torch
    torch.set_num_threads(4)
    parent_hash = sha256(plan['parent_checkpoint'])
    result = dict(created_utc=utc(), plan_sha256=sha256(args.plan),
                  parent_checkpoint_sha256=parent_hash, benchmark_only=True,
                  parameters_are_not_saved_or_used_for_training=True, rows=[])
    with GPUBudget('pretrain-batch-throughput-benchmark', 4) as budget:
        device = require_gpu()
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError('benchmark requires same BF16 path as prior training')
        state = torch.load(plan['parent_checkpoint'], map_location='cpu', weights_only=False)
        model = LubanLM(LubanConfig.from_json(ROOT/plan['model_config'])).to(device)
        configure_model(model, 'sdpa')
        data_path = Path(plan['data_root'])/'processed'
        data = FullPassTokens(data_path/'pretrain-train', plan['training']['length'],plan['seed'],plan['additional_tokens'])
        valid = PackedTokens(data_path/'pretrain-validation', plan['training']['length'])
        result['data_manifest_sha256'] = sha256(data_path/'pretrain-train/manifest.json')
        result['execution_sha256'] = sha256(ROOT/'training/execution.py')
        result['reader']='FullPassTokens; actual prepared corpus'
        transfer = BatchTransfer(True)
        for batch_size in args.batches:
            if budget.remaining() < 30: break
            torch.cuda.empty_cache()
            model.load_state_dict(state['model'], strict=True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=plan['training']['lr'],
                betas=(.9,.95), weight_decay=.1, foreach=False, fused=True)
            optimizer.load_state_dict(state['optimizer'])
            configure_optimizer(optimizer, 'fused')
            for group in optimizer.param_groups: group['lr'] = plan['training']['lr']
            scaler = torch.amp.GradScaler('cuda', enabled=False)
            generator = torch.Generator().manual_seed(2026)
            row = dict(micro_batch=batch_size, accumulation=1, length=data.length)
            try:
                model.train()
                for _ in range(5):
                    update(model, optimizer, scaler, transfer.prepare([data.batch(batch_size,generator)]),
                           device, torch.bfloat16, 'optimized', True)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start = time.monotonic()
                losses, norms = [], []
                for _ in range(args.steps):
                    loss, norm, inputs, targets = update(model, optimizer, scaler,
                        transfer.prepare([data.batch(batch_size,generator)]), device,
                        torch.bfloat16, 'optimized', True)
                    losses.append(loss.detach()); norms.append(norm.detach())
                torch.cuda.synchronize()
                elapsed = time.monotonic()-start
                row.update(status='completed', steps=args.steps,
                    seconds=elapsed, tokens_per_second=args.steps*batch_size*data.length/elapsed,
                    tokens_per_hour=args.steps*batch_size*data.length/elapsed*3600,
                    peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                    peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                    loss_first=float(losses[0]), loss_last=float(losses[-1]),
                    max_gradient_norm=max(float(n) for n in norms), all_updates_finite=True)
                # 此loss只检查副本短更新后的数值；不同batch训练Token数不同，不用于比较泛化。
                row['heldout_numeric_loss'] = validation(model,valid,2,device,torch.bfloat16,batches=4)
            except torch.cuda.OutOfMemoryError:
                row.update(status='out_of_memory')
                model.zero_grad(set_to_none=True)
            finally:
                del optimizer, scaler
                if 'losses' in locals(): del losses, norms
                model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                budget.tick()
            result['rows'].append(row)
            write_json(RUNS/'reports'/args.report_name,result)
            print(json.dumps(row),flush=True)
        result.update(finished_utc=utc(), parent_checkpoint_unchanged=sha256(plan['parent_checkpoint'])==parent_hash,
                      larger_effective_batch_requires_token_based_schedule=True,
                      benchmark_is_not_a_convergence_or_quality_comparison=True)
        if not result['parent_checkpoint_unchanged']:
            raise RuntimeError('parent checkpoint changed')
        write_json(RUNS/'reports'/args.report_name,result)


if __name__ == '__main__':
    main()
