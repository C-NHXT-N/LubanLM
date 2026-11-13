"""Windows orchestrator: sequential, logged WSL conversions of the two local models."""
import os
import json
import argparse
from scripts.common import WORKSPACE,write_json
from scripts.bootstrap_environment import run

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpu',action='store_true');p.add_argument('--reuse-qparams',action='store_true');p.add_argument('--model',choices=['both','Qwen3.5-2B','Qwen3-4B-Instruct-2507'],default='both');args=p.parse_args()
    report=WORKSPACE/'reports/rk3576-qwen-20261018';rows=[]
    python='/mnt/c/Lab/LubanLM-lab/.conda/rkllm-py310/bin/python'
    if args.gpu:
        compatibility=json.loads((report/'gpu-compatibility.json').read_text(encoding='utf8'))
        if not compatibility['completed']:raise RuntimeError('GPU environment not verified')
        python=compatibility['prefix']+'/bin/python'
    env=dict(os.environ);env.update(PYTHONUTF8='1',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1')
    for name in ['Qwen3.5-2B','Qwen3-4B-Instruct-2507']:
        if args.model!='both' and args.model!=name:continue
        status=report/f'conversion-{name}.json'
        if status.exists() and json.loads(status.read_text(encoding='utf8')).get('completed'):
            rows.append({'model':name,'completed':True,'reused':True});continue
        cmd=['wsl','-d','Ubuntu-22.04','--',
             python,'-B','-u',
             '/mnt/c/Lab/LubanLM-lab/LubanLM/scripts/convert_qwen_rk3576.py',
             '--model',name,'--device','cuda' if args.gpu else 'cpu','--dtype','bfloat16']
        if args.reuse_qparams:cmd.append('--reuse-qparams')
        try:
            run(cmd,env,report,f'convert-{name}'+('-gpu' if args.gpu else ''));rows.append({'model':name,'completed':True})
        except RuntimeError as exc:
            rows.append({'model':name,'completed':False,'error':str(exc)})
        write_json(report/'conversion-summary.json',rows)
    write_json(report/'conversion-summary.json',rows)
    if not all(r['completed'] for r in rows):raise SystemExit(1)

if __name__=='__main__':main()
