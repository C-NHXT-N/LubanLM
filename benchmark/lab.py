"""七组短实验：记录配置、参数量、训练变化和速度；不作语言质量排名。"""
from scripts.common import ROOT, DATA, RUNS, GPUBudget, require_gpu, read_json, write_json
from data.prepare import load_jsonl
import gc
import time
from dataclasses import replace
import torch
from torch import nn
from torch.nn import functional as F
from tokenizers import Tokenizer
from model import LubanLM, LubanConfig
from model.blocks import RMSNorm

class GELUFeedForward(nn.Module):
    def __init__(self, hidden, intermediate):
        super().__init__()
        self.up = nn.Linear(hidden, intermediate, bias=False)
        self.down = nn.Linear(intermediate, hidden, bias=False)
    def forward(self, x):
        return self.down(F.gelu(self.up(x)))

def main():
    torch.set_num_threads(4)
    device = require_gpu()
    base = LubanConfig.from_json(ROOT / 'configs/config_50m.json')
    cases = [('baseline', base, 64, 'none', 32000), ('mha', replace(base, num_key_value_heads=base.num_attention_heads), 64, 'none', 32000), ('no_qk_norm', base, 64, 'no_qk_norm', 32000), ('layer_norm', base, 64, 'layer_norm', 32000), ('gelu', base, 64, 'gelu', 32000), ('vocab_16k', replace(base, vocab_size=16000), 64, 'none', 16000), ('vocab_40k', replace(base, vocab_size=40000), 64, 'none', 40000), ('deep_narrow', replace(base, hidden_size=256, num_hidden_layers=22, num_attention_heads=4, num_key_value_heads=1, intermediate_size=768), 64, 'none', 32000), ('shallow_wide', replace(base, hidden_size=768, num_hidden_layers=6, num_attention_heads=12, num_key_value_heads=3, intermediate_size=2304), 64, 'none', 32000), ('context_128', base, 128, 'none', 32000), ('context_256', base, 256, 'none', 32000)]
    # 同一小段真实训练侧文本；不同词表重新编码，避免把另一词表 ID 喂给模型。
    text = '\n'.join(row['text'] for row in load_jsonl(DATA / 'processed/wiki-train.jsonl')[:20])
    rows = []
    with GPUBudget('appendix-controlled-lab', 20) as budget:
        for name, config, length, variant, vocabulary in cases:
            if budget.remaining() < 30:
                break
            tokenizer = Tokenizer.from_file(str(DATA / f'tokenizers/vocab-{vocabulary}/tokenizer.json'))
            config = replace(config, vocab_size=tokenizer.get_vocab_size())
            ids = tokenizer.encode(text).ids[:length+1]
            x = torch.tensor([ids], device=device)
            torch.manual_seed(2026)
            model = LubanLM(config).to(device).train()
            if variant == 'no_qk_norm':
                for block in model.model.layers:
                    block.self_attn.q_norm = nn.Identity()
                    block.self_attn.k_norm = nn.Identity()
            elif variant == 'layer_norm':
                for block in model.model.layers:
                    block.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps, bias=False).to(device)
                    block.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps, bias=False).to(device)
                    block.self_attn.q_norm = nn.LayerNorm(config.head_dim, eps=config.rms_norm_eps, bias=False).to(device)
                    block.self_attn.k_norm = nn.LayerNorm(config.head_dim, eps=config.rms_norm_eps, bias=False).to(device)
                model.model.norm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps, bias=False).to(device)
            elif variant == 'gelu':
                for block in model.model.layers:
                    block.mlp = GELUFeedForward(config.hidden_size, config.intermediate_size).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, foreach=False)
            losses, durations = [], []
            torch.cuda.reset_peak_memory_stats()
            for step in range(10):
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                start = time.perf_counter()
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(x[:, :-1])
                    loss = F.cross_entropy(logits.float().flatten(0, 1), x[:, 1:].flatten())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
                optimizer.step()
                torch.cuda.synchronize()
                losses.append(float(loss))
                durations.append(time.perf_counter()-start)
                budget.tick()
            row = {'case': name, 'config': config.__dict__, 'variant': variant, 'context': length, 'parameters': sum(p.numel() for p in model.parameters()), 'steps': 10, 'initial_loss': losses[0], 'final_loss': losses[-1], 'loss_history': losses, 'tokens_per_second_after_first_2_steps': length*8/sum(durations[2:]), 'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20, 'vocab': tokenizer.get_vocab_size(), 'comparison_limits': 'same repeated text, not held-out quality; parameter counts not matched; variant initialization and tokenization may differ'}
            rows.append(row)
            write_json(RUNS / 'reports/lab_results.json', rows)
            print(row['case'], row['initial_loss'], row['final_loss'], flush=True)
            del logits, loss, model, optimizer
            gc.collect()
            torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
