"""Isolated Blackwell compatibility environment; the SDK-pinned CPU env is preserved."""
import json
import os
from pathlib import Path
import subprocess
import venv

BASE=Path('/mnt/c/Lab/LubanLM-lab')
PREFIX=Path.home()/'.venvs/lubanlm-rkllm-gpu310'
REPORT=BASE/'reports/rk3576-qwen-20261018'

def main():
    os.environ.update(PIP_CONFIG_FILE='/dev/null',PIP_INDEX_URL='https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple',
        PIP_DISABLE_PIP_VERSION_CHECK='1',PIP_CACHE_DIR=str(BASE/'.cache/pip-linux'),PYTHONDONTWRITEBYTECODE='1',
        PYTHONPATH=str(BASE/'.cache/rkllm-bootstrap'))
    os.environ.pop('PIP_EXTRA_INDEX_URL',None)
    if not (PREFIX/'bin/python').exists():venv.create(PREFIX,with_pip=False)
    py=str(PREFIX/'bin/python')
    def call(*args):
        print('RUN',*args,flush=True);subprocess.run([py,*args],check=True)
    call('-m','pip','install','pip==25.3','setuptools==75.8.2','wheel==0.45.1')
    os.environ.pop('PYTHONPATH',None)
    call('-m','pip','install','torch==2.7.1+cu128','torchvision==0.22.1+cu128','numpy==1.26.4','pillow==10.4.0','Jinja2==3.1.4','fsspec==2025.9.0',
        '--index-url','https://mirrors.nju.edu.cn/pytorch/whl/cu128',
        '--extra-index-url','https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple')
    # Reuse unchanged SDK and non-Torch dependencies after the GPU environment's own packages.
    site=PREFIX/'lib/python3.10/site-packages'
    (site/'rkllm-base.pth').write_text(str(BASE/'.conda/rkllm-py310/lib/python3.10/site-packages')+'\n')
    check=subprocess.run([py,'-m','pip','check'],capture_output=True,text=True)
    (REPORT/'gpu-pip-check.txt').write_text(check.stdout+check.stderr)
    unexpected=[s for s in check.stdout.splitlines() if s and not s.startswith('rkllm-toolkit 1.3.1 has requirement torch')]
    if check.returncode and unexpected:raise RuntimeError('Unexpected GPU dependency conflict: '+str(unexpected))
    call('-c',"import torch;from rkllm.api import RKLLM;torch.cuda.init();a=torch.randn(256,256,device='cuda');b=a@a;torch.cuda.synchronize();print(torch.__version__,torch.cuda.get_device_name(0),torch.cuda.get_device_capability(0));print('RKLLM import and CUDA matmul passed')")
    (REPORT/'gpu-pip-freeze.txt').write_bytes(subprocess.check_output([py,'-m','pip','freeze','--all']))
    (REPORT/'gpu-compatibility.json').write_text(json.dumps({'completed':True,'prefix':str(PREFIX),
        'sdk':'1.3.1','torch':'2.7.1+cu128','torchvision':'0.22.1+cu128',
        'declared_requirement_deviations':['SDK declares torch==2.6.0','SDK declares torchvision==0.21.0'],
        'reason':'Blackwell sm_120 requires CUDA12.8-capable PyTorch; CPU SDK forces FP32'},indent=2)+'\n')

if __name__=='__main__':main()
