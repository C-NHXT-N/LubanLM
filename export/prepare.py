"""复用 C3 严格 Qwen3 映射，导出真正训练过的 Base/Chat。"""
from scripts.common import DATA, ROOT, RUNS, read_json, write_json
import argparse
import importlib.util
import torch
from model import LubanConfig, LubanLM
from training.train import load_checkpoint

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    state = load_checkpoint(args.checkpoint)
    config = LubanConfig.from_dict(state['config'])
    source = ROOT / 'C3/program/3.4/prepare_rkllm_model.py'
    spec = importlib.util.spec_from_file_location('hf_prepare', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from pathlib import Path
    module.build_hf_qwen3_directory(config, DATA / 'tokenizers/vocab-32000', Path(args.output), 2026, Path(args.checkpoint))
    from transformers import AutoModelForCausalLM, AutoTokenizer
    restored = AutoModelForCausalLM.from_pretrained(args.output, local_files_only=True, attn_implementation='eager').eval()
    original = LubanLM(config).eval()
    original.load_state_dict(state['model'], strict=True)
    tokenizer = AutoTokenizer.from_pretrained(args.output, local_files_only=True)
    x = tokenizer('Linux 是一个操作系统', return_tensors='pt', add_special_tokens=False)['input_ids']
    with torch.no_grad():
        error = float((original(x)-restored(x).logits).abs().max())
    if error > 2e-4:
        raise RuntimeError(f'export/reload logits mismatch: {error}')
    write_json(Path(args.output) / 'reload_verification.json', {'passed': True, 'logits_max_abs_error': error, 'parameter_count': sum(p.numel() for p in restored.parameters()), 'checkpoint_step': state['step'], 'native_reload': 'supported by inference.generate.load_model'})
    print('Trained export and HF reload passed:', error)

if __name__ == '__main__':
    main()
