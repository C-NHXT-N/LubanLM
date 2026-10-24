"""等待当前 Base 训练完成，顺序执行剩余 Windows 实验与交付检查。"""
from scripts.common import ROOT, RUNS, read_json, write_json
import subprocess
import sys
import time

def main():
    base = RUNS / 'checkpoints/50m-base'
    chat = RUNS / 'checkpoints/50m-chat'
    start = time.monotonic()
    while not (base / 'summary.json').exists() or (RUNS / 'reports/gpu_budget.lock').exists():
        if time.monotonic() - start > 140*60:
            raise RuntimeError('Base training did not finish within its expected budget; inspect process and logs')
        time.sleep(3)
    summary = read_json(base / 'summary.json')
    if summary['stop_reason'].startswith('error') or summary['step'] == 0:
        raise RuntimeError('Base training failed; do not run dependent SFT')
    commands = [
        ['-m', 'training.train', '--config', str(ROOT/'configs/config_50m.json'), '--stage', 'sft', '--init', str(base/'last.pt'), '--output', str(chat), '--epochs', '2', '--minutes', '40', '--micro-batch', '2', '--accumulation', '4', '--lr', '5e-5'],
        ['-m', 'export.prepare', '--checkpoint', str(base/'last.pt'), '--output', str(RUNS/'exports/LubanLM-50M-Base')],
        ['-m', 'export.prepare', '--checkpoint', str(chat/'last.pt'), '--output', str(RUNS/'exports/LubanLM-50M-Chat')],
        ['-m', 'eval.observe', '--base', str(base), '--chat', str(chat/'last.pt')],
        ['-m', 'eval.lm_harness', '--checkpoint', str(base/'last.pt')],
        ['-m', 'benchmark.lab'],
        ['-m', 'scripts.verify_exports'],
        ['-m', 'scripts.report'],
        ['-m', 'scripts.audit_docs'],
    ]
    status = []
    for command in commands:
        print('Running:', ' '.join(command), flush=True)
        result = subprocess.run([sys.executable]+command, cwd=ROOT)
        status.append({'command': command, 'exit_code': result.returncode})
        write_json(RUNS / 'reports/finish_status.json', status)
        if result.returncode:
            raise RuntimeError('Stage failed; inspect output before resuming: '+' '.join(command))
    print('All Windows stages finished', flush=True)

if __name__ == '__main__':
    main()
