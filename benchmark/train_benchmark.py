"""0.5B 真实 optimizer-step 基准；包括 FP32 参数、梯度及 AdamW 状态。"""
from scripts.common import ROOT, RUNS, GPUBudget, require_gpu, write_json
import argparse
import gc
import time
import torch
from torch.nn import functional as F
from model import LubanConfig, LubanLM

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default=ROOT / 'configs/config_0.5b.json')
    p.add_argument('--steps', type=int, default=10)
    p.add_argument('--warmup', type=int, default=3)
    args = p.parse_args()
    torch.set_num_threads(4)
    device = require_gpu()
    config = LubanConfig.from_json(args.config)
    report = {'gpu': torch.cuda.get_device_name(0), 'config': config.__dict__, 'attempts': [], 'data': 'random IDs, performance only; not language quality'}
    with GPUBudget('0.5b-benchmark', 20) as budget:
        for length in (256, 128, 64):
            model = optimizer = x = logits = loss = None
            row = {'sequence_length': length, 'micro_batch': 1, 'accumulation': 4, 'gradient_checkpointing': True}
            try:
                torch.manual_seed(2026)
                model = LubanLM(config).to(device).train()
                model.gradient_checkpointing = True
                optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, foreach=False)
                torch.cuda.reset_peak_memory_stats()
                start, measured_tokens, measured = None, 0, 0
                for step in range(args.warmup + args.steps):
                    if budget.remaining() < 20:
                        break
                    if step == args.warmup:
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                    optimizer.zero_grad(set_to_none=True)
                    for _ in range(4):
                        x = torch.randint(3, config.vocab_size, (1, length+1), device=device)
                        with torch.autocast('cuda', dtype=torch.bfloat16):
                            logits = model(x[:, :-1])
                            loss = F.cross_entropy(logits.float().flatten(0, 1), x[:, 1:].flatten()) / 4
                        loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                    optimizer.step()
                    if step >= args.warmup:
                        measured_tokens += 4*length
                        measured += 1
                    budget.tick()
                torch.cuda.synchronize()
                seconds = time.perf_counter()-start if start else 0
                if not measured:
                    raise RuntimeError('budget ended before measured steps')
                speed = measured_tokens / seconds
                row.update(status='passed', measured_steps=measured, warmup_steps=args.warmup, seconds=seconds, tokens_per_second=speed, peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20, peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20, parameters=sum(p.numel() for p in model.parameters()), training_tokens_estimate={f'{h}h': speed*h*3600 for h in (24, 36, 48)})
            except torch.cuda.OutOfMemoryError as exc:
                row.update(status='out_of_memory', error=str(exc))
            finally:
                report['attempts'].append(row)
                write_json(RUNS / 'reports/benchmark_0.5b.json', report)
                print(row, flush=True)
                del model, optimizer, x, logits, loss
                gc.collect()
                torch.cuda.empty_cache()
            if row['status'] == 'passed':
                break

if __name__ == '__main__':
    main()
