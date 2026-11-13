"""保存真实续写、Base/Chat 对比、采样差异及 cache 计时。"""
from scripts.common import RUNS, GPUBudget, require_gpu, write_json
import argparse
import gc
import time
from pathlib import Path
import torch
from inference.generate import load_model, complete
from sft.template import prompt

COMPLETIONS = ['Linux 是', '中国的首都是', 'C语言是一种', '人工智能是']
QUESTIONS = ['你好', '你是谁', 'Linux是什么', '操作系统是什么', '1+2是多少']

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', default=RUNS / 'checkpoints/50m-base')
    p.add_argument('--chat', default=RUNS / 'checkpoints/50m-chat/last.pt')
    args = p.parse_args()
    device = require_gpu()
    report = {'completions': [], 'base_chat': [], 'sampling': [], 'timing': []}
    with GPUBudget('generation-observation', 10) as budget:
        checkpoints = [Path(args.base)/'initial.pt'] + sorted(Path(args.base).glob('step-*.pt')) + [Path(args.base)/'last.pt']
        for path in checkpoints:
            if not path.exists() or budget.remaining() < 30:
                continue
            model, tokenizer = load_model(path, device)
            for text in COMPLETIONS:
                output, ids = complete(model, tokenizer, text, max_new_tokens=32)
                report['completions'].append({'checkpoint': path.name, 'prompt': text, 'completion': output, 'generated_tokens': len(ids)})
            del model
            gc.collect()
            torch.cuda.empty_cache()
            budget.tick()
        for kind, path in [('Base', Path(args.base)/'last.pt'), ('Chat', Path(args.chat))]:
            model, tokenizer = load_model(path, device)
            for question in QUESTIONS:
                output, ids = complete(model, tokenizer, prompt(question), max_new_tokens=48)
                report['base_chat'].append({'model': kind, 'question': question, 'output': output, 'generated_tokens': len(ids)})
            if kind == 'Chat':
                for temperature, k, top_p in [(0., 0, 1.), (.7, 40, .9), (1., 0, .95)]:
                    torch.manual_seed(2026)
                    output, ids = complete(model, tokenizer, prompt('Linux是什么'), max_new_tokens=48, temperature=temperature, top_k=k, top_p=top_p, repetition_penalty=1.1)
                    report['sampling'].append({'temperature': temperature, 'top_k': k, 'top_p': top_p, 'output': output})
                text = prompt('请介绍操作系统。')
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                    x = torch.tensor([tokenizer.encode(text).ids], device=device)
                    model(x, use_cache=True)
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    result = model(x, use_cache=True)
                    torch.cuda.synchronize()
                    prefill = time.perf_counter()-start
                    report['prefill'] = {'tokens': x.numel(), 'seconds': prefill, 'tokens_per_second': x.numel()/prefill}
                    report['cache'] = {'bytes': sum(k.numel()*k.element_size()+v.numel()*v.element_size() for k,v in result.past_key_values), 'kv_heads': model.config.num_key_value_heads, 'query_heads': model.config.num_attention_heads}
                for cache in (False, True):
                    start = time.perf_counter()
                    output, ids = complete(model, tokenizer, text, max_new_tokens=64, use_cache=cache, amp=False)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter()-start
                    report['timing'].append({'use_cache': cache, 'precision': 'float32; controlled cache comparison', 'total_seconds': elapsed, 'generated_tokens': len(ids), 'tokens_per_second_including_prefill': len(ids)/elapsed, 'output': output})
                assert report['timing'][0]['output'] == report['timing'][1]['output']
            del model
            gc.collect()
            torch.cuda.empty_cache()
            budget.tick()
        write_json(RUNS / 'reports/generation_observations.json', report)
    print('Saved fixed prompts and timing')

if __name__ == '__main__':
    main()
