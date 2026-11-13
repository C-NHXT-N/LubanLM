"""数据隔离、聊天模板/掩码，以及中间检查点恢复与连续训练一致性。"""
from scripts.common import ROOT, DATA, RUNS, read_json, write_json, sha256
from data.prepare import load_jsonl
import json
from sft.template import encode_pair, prompt, SYSTEM
from training.train import load_checkpoint
from pathlib import Path
from tokenizers import Tokenizer
import subprocess
import sys
import torch

def run():
    def hashes(path):
        with path.open(encoding='utf-8') as stream:
            return {json.loads(line)['sha256'] for line in stream if line.strip()}
    assert not hashes(DATA / 'processed/wiki-train.jsonl') & hashes(DATA / 'processed/wiki-validation.jsonl')
    sft_train = load_jsonl(DATA / 'processed/sft-train-tokens.jsonl')
    sft_valid = load_jsonl(DATA / 'processed/sft-validation-tokens.jsonl')
    assert not set(r['tree_id'] for r in sft_train) & set(r['tree_id'] for r in sft_valid)
    directory = DATA / 'tokenizers/vocab-32000'
    tokenizer = Tokenizer.from_file(str(directory / 'tokenizer.json'))
    pair = encode_pair(tokenizer, '1+2是多少？', '1+2=3。')
    prefix = prompt('1+2是多少？')
    encoded = tokenizer.encode(prefix+'1+2=3。<eos>')
    for label, (start,end) in zip(pair['labels'], encoded.offsets):
        if end <= len(prefix):
            assert label == -100
        if label != -100:
            assert start >= len(prefix)
    assert pair['labels'][-1] == tokenizer.token_to_id('<eos>')
    from transformers import AutoTokenizer
    hf = AutoTokenizer.from_pretrained(directory, local_files_only=True)
    rendered = hf.apply_chat_template([{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': '1+2是多少？'}], tokenize=False, add_generation_prompt=True)
    assert rendered == prefix, repr(rendered)
    config = read_json(ROOT / 'configs/config_50m.json')
    config.update(name='tiny-resume-check', hidden_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16, intermediate_size=128)
    config_path = RUNS / 'reports/resume-config.json'
    write_json(config_path, config)
    common = [sys.executable, '-m', 'training.train', '--config', str(config_path), '--steps', '4', '--minutes', '5', '--length', '32', '--micro-batch', '2', '--accumulation', '2']
    identity = sha256(DATA / 'processed/sft-train-tokens.jsonl')[:10]
    original = RUNS / f'checkpoints/verification-{identity}-continuous'
    resumed = RUNS / f'checkpoints/verification-{identity}-resumed'
    if not (original / 'last.pt').exists():
        subprocess.run(common + ['--output', str(original)], check=True, cwd=ROOT)
    if not (resumed / 'last.pt').exists():
        subprocess.run(common + ['--output', str(resumed), '--resume', str(original / 'step-000002.pt')], check=True, cwd=ROOT)
    first, second = load_checkpoint(original / 'last.pt'), load_checkpoint(resumed / 'last.pt')
    weight_error = max(float((first['model'][name]-second['model'][name]).abs().max()) for name in first['model'])
    optimizer_error = 0.
    for index, state in first['optimizer']['state'].items():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                optimizer_error = max(optimizer_error, float((value-second['optimizer']['state'][index][key]).abs().max()))
    assert weight_error == 0 and optimizer_error == 0
    assert torch.equal(first['data_rng'], second['data_rng'])
    assert first['input_tokens'] == second['input_tokens']
    result = {'data_root': str(DATA), 'data_split_disjoint': True, 'sft_tree_split_disjoint': True, 'assistant_mask_verified': True, 'hf_template_equals_native': True, 'resume_weight_max_abs_error': weight_error, 'resume_optimizer_max_abs_error': optimizer_error, 'resume_data_rng_equal': True, 'steps': 4, 'passed': True}
    # SFT 有排列、游标和 epoch 状态，额外核对这一条不同于随机窗口的恢复路径。
    sft_original = RUNS / f'checkpoints/verification-{identity}-sft-continuous'
    sft_resumed = RUNS / f'checkpoints/verification-{identity}-sft-resumed'
    sft_common = common + ['--stage', 'sft', '--lr', '5e-5']
    if not (sft_original / 'last.pt').exists():
        subprocess.run(sft_common + ['--output', str(sft_original), '--init', str(original / 'last.pt')], check=True, cwd=ROOT)
    if not (sft_resumed / 'last.pt').exists():
        subprocess.run(sft_common + ['--output', str(sft_resumed), '--resume', str(sft_original / 'step-000002.pt')], check=True, cwd=ROOT)
    a, b = load_checkpoint(sft_original / 'last.pt'), load_checkpoint(sft_resumed / 'last.pt')
    sft_error = max(float((a['model'][name]-b['model'][name]).abs().max()) for name in a['model'])
    sft_optimizer_error = max(float((value-b['optimizer']['state'][index][key]).abs().max()) for index,state in a['optimizer']['state'].items() for key,value in state.items() if isinstance(value,torch.Tensor))
    assert sft_error == 0 and sft_optimizer_error == 0
    assert a['sampler'] == b['sampler'] and torch.equal(a['data_rng'],b['data_rng'])
    result.update(sft_resume_weight_max_abs_error=sft_error, sft_resume_optimizer_max_abs_error=sft_optimizer_error, sft_sampler_and_rng_equal=True)
    write_json(RUNS / 'reports/training_checks.json', result)
    print(result)

if __name__ == '__main__':
    run()
