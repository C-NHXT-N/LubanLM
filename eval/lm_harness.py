"""把手写 LubanLM 接入 lm-evaluation-harness，运行 20 道 C-Eval 验证题。"""
from scripts.common import RUNS, GPUBudget, require_gpu, write_json, utc
import argparse
import math
import torch
from lm_eval.api.model import LM
from lm_eval import evaluator, utils
from lm_eval.tasks import TaskManager, get_task_dict
from huggingface_hub import HfApi
from inference.generate import load_model, complete

class LubanHarness(LM):
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model, self.tokenizer = model, tokenizer
        self.device = next(model.parameters()).device
        self.trace = []

    @torch.inference_mode()
    def score(self, context, targets):
        if not targets:
            return 0., True
        max_length = self.model.config.context_length
        if len(targets) > max_length:
            raise ValueError('continuation exceeds context limit; use rolling scoring')
        context = context or [self.model.config.bos_token_id]
        # 左侧截断只发生在上下文，保留全部待评分答案。
        combined = (context + targets)[-(max_length+1):]
        x = torch.tensor([combined[:-1]], device=self.device)
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == 'cuda'):
            logits = self.model(x)[0, -len(targets):].float()
        target = torch.tensor(targets, device=self.device)
        log_probs = logits.log_softmax(-1)
        total = float(log_probs.gather(1, target[:, None]).sum())
        greedy = bool((logits.argmax(-1) == target).all())
        return total, greedy

    def loglikelihood(self, requests):
        results = []
        for request in requests:
            context, continuation = request.args
            # 按完整文本编码；跨文字边界的 token 纳入答案评分，不丢掉该 token。
            full = self.tokenizer.encode(context+continuation, add_special_tokens=False)
            split = next((i for i, (_, end) in enumerate(full.offsets) if end > len(context)), len(full.ids))
            if split == len(full.ids):
                raise ValueError('empty encoded continuation')
            results.append(self.score(full.ids[:split], full.ids[split:]))
            self.trace.append({'context_tokens': split, 'answer_tokens': len(full.ids)-split, 'truncated_context': len(full.ids) > self.model.config.context_length+1, 'answer': continuation, 'loglikelihood': results[-1][0]})
        return results

    def loglikelihood_rolling(self, requests):
        results = []
        for request in requests:
            ids = self.tokenizer.encode(request.args[0], add_special_tokens=False).ids
            windows = utils.get_rolling_token_windows(token_list=ids, prefix_token=self.model.config.bos_token_id, max_seq_len=self.model.config.context_length, context_len=1)
            results.append(sum(self.score(*utils.make_disjoint_window(window))[0] for window in windows))
        return results

    def generate_until(self, requests):
        results = []
        for request in requests:
            text, settings = request.args
            output, _ = complete(self.model, self.tokenizer, text, max_new_tokens=settings.get('max_gen_toks', 64))
            until = settings.get('until', [])
            until = [until] if isinstance(until, str) else until
            for stop in until:
                if stop:
                    output = output.split(stop)[0]
            results.append(output)
        return results

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', default=RUNS / 'checkpoints/50m-base/last.pt')
    args = p.parse_args()
    tasks = ['ceval-valid_computer_network', 'ceval-valid_operating_system']
    revision = HfApi().dataset_info('ceval/ceval-exam').sha
    # 数据下载与任务构造在 GPU 计时外完成。
    task_dict = get_task_dict(tasks, task_manager=TaskManager())
    with GPUBudget('ceval-20-questions', 10) as budget:
        model, tokenizer = load_model(args.checkpoint, require_gpu())
        adapter = LubanHarness(model, tokenizer)
        results = evaluator.simple_evaluate(model=adapter, tasks=list(task_dict.values()), num_fewshot=0, limit=10, batch_size=1, bootstrap_iters=100, log_samples=True, random_seed=2026, numpy_random_seed=2026, torch_random_seed=2026)
        # NumPy 数值与框架配置使用标准序列化，供后续 Markdown 报告读取。
        import json
        plain = json.loads(json.dumps(results, default=lambda x: x.item() if hasattr(x, 'item') else str(x)))
        plain['luban_metadata'] = {'created_utc': utc(), 'checkpoint': str(args.checkpoint), 'adapter': 'native LubanLM, whole-text BPE offsets, no HF generate', 'dataset': 'ceval/ceval-exam', 'revision_observed_before_load': revision, 'license': 'CC-BY-NC-SA-4.0', 'limit_per_task': 10, 'purpose': 'tool demonstration; not a reliable capability benchmark', 'scoring_trace': adapter.trace}
        write_json(RUNS / 'reports/ceval_results.json', plain)
        budget.tick()
    print(plain['results'])

if __name__ == '__main__':
    main()
