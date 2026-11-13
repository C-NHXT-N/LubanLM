"""对导出目录执行原生重新加载、权重一致性及真正的 CLI 生成。"""
from scripts.common import RUNS, GPUBudget, require_gpu, write_json
from inference.generate import load_model, complete
from sft.template import prompt
import gc
import subprocess
import sys
import torch
import argparse
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-checkpoint',type=Path,default=RUNS/'checkpoints/50m-base/last.pt')
    parser.add_argument('--chat-checkpoint',type=Path,default=RUNS/'checkpoints/50m-chat/last.pt')
    args=parser.parse_args()
    device = require_gpu()
    rows = []
    with GPUBudget('native-export-reload', 5):
        for kind in ('Base', 'Chat'):
            path = RUNS / f'exports/LubanLM-50M-{kind}'
            checkpoint = args.base_checkpoint if kind=='Base' else args.chat_checkpoint
            model, tokenizer = load_model(checkpoint, device)
            exported, exported_tok = load_model(path, device)
            x = torch.tensor([[1, 3, 4, 5]], device=device)
            with torch.no_grad():
                error = float((model(x)-exported(x)).abs().max())
            assert error == 0
            output, ids = complete(exported, exported_tok, prompt('你好'), max_new_tokens=24)
            rows.append({'kind': kind, 'native_export_logits_max_abs_error': error, 'output': output, 'generated_tokens': len(ids), 'passed': True})
            del model, exported
            gc.collect()
            torch.cuda.empty_cache()
    # CLI 自己持有预算锁，不能嵌套在上面的预算区间中。
    result = subprocess.run([sys.executable, '-m', 'inference.chat', '--checkpoint', str(RUNS/'exports/LubanLM-50M-Chat'), '--question', '你好', '--temperature', '0', '--max-new-tokens', '24'], check=True, capture_output=True, text=True, encoding='utf-8')
    write_json(RUNS/'reports/export_checks.json', {'models': rows, 'chat_cli_stdout': result.stdout, 'chat_cli_stderr': result.stderr, 'chat_cli_exit_code': result.returncode, 'passed': True})
    print(result.stdout)

if __name__ == '__main__':
    main()
