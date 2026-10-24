"""实测 GPU、cache、padding、checkpointing、梯度及可恢复训练状态。"""
from scripts.common import RUNS, GPUBudget, require_gpu, write_json
from dataclasses import replace
import torch
from torch.nn import functional as F
from model import LubanLM, LubanConfig

def main():
    torch.set_num_threads(4)
    device = require_gpu()
    with GPUBudget('gpu-functional-checks', 10):
        torch.manual_seed(2026)
        config = LubanConfig('tiny-verification', 'verification', 64, 32, 2, 4, 2, 8, 64, 128, True, False, 0, 1, 2)
        model = LubanLM(config).to(device).eval()
        ids = torch.tensor([[3, 4, 5, 6, 7]], device=device)
        with torch.no_grad():
            full = model(ids)
            first = model(ids[:, :3], use_cache=True)
            rest = model(ids[:, 3:], past_key_values=first.past_key_values, use_cache=True)
            cache_error = float((full[:, 3:] - rest.logits).abs().max())
            assert cache_error < 2e-6, cache_error
            assert rest.past_key_values[0][0].shape == (1, 2, 5, 8)
            changed = ids.clone()
            changed[:, 3:] = torch.tensor([9, 10], device=device)
            causal_error = float((full[:, :3] - model(changed)[:, :3]).abs().max())
            assert causal_error == 0
            padded = torch.tensor([[3, 4, 5, 0, 0]], device=device)
            mask = torch.tensor([[1, 1, 1, 0, 0]], device=device)
            padding_error = float((model(ids[:, :3]) - model(padded, attention_mask=mask)[:, :3]).abs().max())
            assert padding_error < 2e-6
        model.train()
        plain = F.cross_entropy(model(ids[:, :-1]).flatten(0, 1), ids[:, 1:].flatten())
        plain.backward()
        gradients = {n: p.grad.clone() for n, p in model.named_parameters()}
        model.zero_grad(set_to_none=True)
        model.gradient_checkpointing = True
        recomputed = F.cross_entropy(model(ids[:, :-1]).flatten(0, 1), ids[:, 1:].flatten())
        recomputed.backward()
        gradient_error = max(float((p.grad-gradients[n]).abs().max()) for n, p in model.named_parameters())
        assert gradient_error < 2e-6
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, foreach=False)
        before = model.model.embed_tokens.weight.detach().clone()
        optimizer.step()
        assert not torch.equal(before, model.model.embed_tokens.weight)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = F.cross_entropy(model(ids[:, :-1]).float().flatten(0, 1), ids[:, 1:].flatten())
        assert torch.isfinite(loss)
        write_json(RUNS / 'reports/gpu_checks.json', {'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__, 'cuda_runtime': torch.version.cuda, 'bf16_supported': torch.cuda.is_bf16_supported(), 'cache_max_abs_error': cache_error, 'causal_error': causal_error, 'padding_max_abs_error': padding_error, 'checkpoint_gradient_max_abs_error': gradient_error, 'bf16_loss': float(loss), 'parameter_update': True, 'passed': True})
        print('GPU cache/padding/checkpoint/backprop checks passed')

if __name__ == '__main__':
    main()
