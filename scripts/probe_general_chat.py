"""冻结候选的通用聊天诊断：仅推理，记录新问题与数值一致性，不训练。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = [
    '今天午饭吃了面条，你想和我聊聊吃饭吗？',
    '周末想待在家里，给我推荐一种轻松的活动。',
    '我刚学会骑自行车，有点开心。',
    '朋友没回我的消息，我该怎样礼貌地再问一次？',
    '我想学做饭，第一道菜选什么比较简单？',
    '东西太多找不到钥匙，你会怎么建议我？',
    '帮我写一句邀请朋友一起喝茶的话。',
    '我明天要去看望爷爷，可以带点什么？',
    '今天没什么特别的事，想随便聊两句。',
    '我喜欢听音乐，你能接着这个话题聊一句吗？',
    '请把这句话说得更礼貌：快把书还给我。',
    '请从苹果、铅笔、香蕉里挑出水果。',
    '请用一句话解释为什么洗完的衣服要晾干。',
    '我想了解咖啡和茶在味道上有什么不同。',
    '帮我写一句祝朋友搬家顺利的话。',
    '我没有告诉你我的名字，你知道我叫什么吗？',
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, default=Path('D:/LubanLM-Runs/chat-v2/checkpoints/50m-chat-v7-clean/last.pt'))
    p.add_argument('--output-root', type=Path, default=ROOT/'docs/run-results/chat-diagnosis-20261005')
    p.add_argument('--minutes', type=float, default=3)
    args = p.parse_args()
    if not 0 < args.minutes <= 5:
        p.error('diagnostic GPU budget must be in (0,5] minutes')
    output = args.output_root/'general-probe.json'
    if output.exists():
        p.error('output already exists; preserve recorded diagnostic answers')
    os.environ['LUBAN_DATA'] = 'D:/LubanLM-Data/chat-v2'
    # 独立的工作区诊断账本，最终报告显式加回旧训练总账，不重置旧账。
    os.environ['LUBAN_RUNS'] = str(args.output_root/'gpu-accounting')
    os.environ['LUBAN_GPU_BUDGET_MINUTES'] = str(args.minutes)
    import torch
    from inference.generate import load_model, complete
    from sft.template import prompt
    from training.execution import configure_model
    from scripts.common import GPUBudget, require_gpu, sha256, write_json, utc, read_json

    torch.set_num_threads(4)
    write_json(args.output_root/'probe-definition.json', dict(questions=QUESTIONS, seeds=[2026,2027,2028],
               status='diagnostic development prompts; not a final test; do not insert answers into training', created_utc=utc()))
    started = time.monotonic()
    results, numerical = [], {}
    with GPUBudget('diagnostic-inference-only-50m', args.minutes) as budget:
        model, tok = load_model(args.checkpoint, require_gpu())
        checkpoint_hash = sha256(args.checkpoint)
        with torch.inference_mode():
            x = torch.tensor([tok.encode(prompt('和我聊聊你能提供什么帮助。'), add_special_tokens=False).ids], device=next(model.parameters()).device)
            manual = model(x)
            cached = model(x[:,:-1], use_cache=True)
            last = model(x[:,-1:], past_key_values=cached.past_key_values, use_cache=True).logits
            numerical['fp32_cache_last_logit_max_abs_error'] = float((manual[:,-1]-last[:,-1]).abs().max())
            numerical['fp32_cache_last_argmax_equal'] = bool(manual[:,-1].argmax(-1).eq(last[:,-1].argmax(-1)).all())
            configure_model(model, 'sdpa')
            sdpa = model(x)
            numerical['fp32_manual_sdpa_logit_max_abs_error'] = float((manual-sdpa).abs().max())
            numerical['fp32_manual_sdpa_last_argmax_equal'] = bool(manual[:,-1].argmax(-1).eq(sdpa[:,-1].argmax(-1)).all())
            configure_model(model, 'manual')
        for mode in ('greedy', 'sampling'):
            prompts = QUESTIONS if mode == 'greedy' else QUESTIONS[:6]
            seeds = [2026] if mode == 'greedy' else [2026,2027,2028]
            for seed in seeds:
                torch.manual_seed(seed)
                for index, question in enumerate(prompts):
                    if budget.remaining() < 10:
                        break
                    answer, ids = complete(model, tok, prompt(question), max_new_tokens=96,
                        temperature=0 if mode=='greedy' else .7,
                        top_k=0 if mode=='greedy' else 40, top_p=1 if mode=='greedy' else .9,
                        repetition_penalty=1 if mode=='greedy' else 1.1)
                    r = dict(id=f'{mode}-{seed}-{index:02d}', mode=mode, seed=seed, question=question, answer=answer,
                             generated_ids=ids, eos=bool(ids and ids[-1]==model.config.eos_token_id),
                             replacement_character='\ufffd' in answer,
                             unexpected_controls=any(i in (model.config.pad_token_id, model.config.bos_token_id) for i in ids))
                    results.append(r)
                    print(json.dumps({k:r[k] for k in ('id','question','answer','eos','replacement_character')}, ensure_ascii=False), flush=True)
                    budget.tick()
    expected = len(QUESTIONS)+3*6
    diagnostics = read_json(args.output_root/'gpu-accounting/reports/gpu_budget.json')
    original = read_json('D:/LubanLM-Runs/chat-v2/reports/gpu_budget.json')
    write_json(output, dict(created_utc=utc(), checkpoint=str(args.checkpoint), checkpoint_sha256=checkpoint_hash,
               training_started=False, status='diagnostic_completed' if len(results)==expected else 'diagnostic_incomplete',
               numerical_checks=numerical, results=results, expected_outputs=expected,
               wall_seconds=time.monotonic()-started,
               diagnostic_gpu_minutes=sum(r['seconds'] for r in diagnostics)/60,
               original_plus_diagnostic_gpu_minutes=sum(r['seconds'] for r in diagnostics+original)/60))
    print('Saved:', output)
    print(json.dumps(numerical))


if __name__ == '__main__':
    main()
