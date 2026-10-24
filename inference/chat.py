"""PowerShell: python -m inference.chat --checkpoint ... [--question ...]"""
from scripts.common import GPUBudget, require_gpu
import argparse
import torch
from inference.generate import load_model, generate_ids
from sft.template import prompt

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--question')
    p.add_argument('--temperature', type=float, default=.7)
    p.add_argument('--top-k', type=int, default=40)
    p.add_argument('--top-p', type=float, default=.9)
    p.add_argument('--repetition-penalty', type=float, default=1.1)
    p.add_argument('--max-new-tokens', type=int, default=96)
    p.add_argument('--minutes', type=float, default=10)
    args = p.parse_args()
    torch.manual_seed(2026)
    with GPUBudget('chat', args.minutes) as budget:
        model, tokenizer = load_model(args.checkpoint, require_gpu())
        while budget.remaining() > 10:
            question = args.question if args.question is not None else input('用户 > ')
            if question.strip().lower() in ('exit', 'quit'):
                break
            print('助手 > ', end='', flush=True)
            printed = ''
            for ids in generate_ids(model, tokenizer, prompt(question), max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p, repetition_penalty=args.repetition_penalty):
                text = tokenizer.decode(ids, skip_special_tokens=True)
                # 单个 BPE token 可能只是 UTF-8 字节的一部分；等字符完整后再输出。
                if '\ufffd' not in text and text.startswith(printed):
                    print(text[len(printed):], end='', flush=True)
                    printed = text
                if budget.tick():
                    break
            if text.startswith(printed):
                print(text[len(printed):], end='')
            print()
            if args.question is not None:
                break

if __name__ == '__main__':
    main()
