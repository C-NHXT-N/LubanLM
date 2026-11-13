"""WSL half of setup; all HTTPS inherited from the temporary physical relay."""
import os
from pathlib import Path
import subprocess
import sys
import venv

BASE=Path('/mnt/c/Lab/LubanLM-lab')
PREFIX=BASE/'.conda/rkllm-py310'
REPORT=BASE/'reports/rk3576-qwen-20261018'
SDK=BASE/'.cache/rkllm-1.3.1/rkllm-toolkit/packages'

def main():
    os.environ.update(PIP_CONFIG_FILE='/dev/null',PIP_INDEX_URL='https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple',
        PIP_DISABLE_PIP_VERSION_CHECK='1',PIP_CACHE_DIR=str(BASE/'.cache/pip-linux'),
        PYTHONPATH=str(BASE/'.cache/rkllm-bootstrap'),PYTHONDONTWRITEBYTECODE='1')
    os.environ.pop('PIP_EXTRA_INDEX_URL',None)
    if not (PREFIX/'bin/python').exists():venv.create(PREFIX,with_pip=False)
    py=str(PREFIX/'bin/python')
    def call(*args):
        print('RUN',*args,flush=True);subprocess.run([py,*args],check=True)
    call('-m','pip','install','pip==25.3','setuptools==75.8.2','wheel==0.45.1')
    os.environ.pop('PYTHONPATH',None)
    # CPU wheel avoids unsupported sm_120 kernels in the SDK-pinned torch 2.6.
    call('-m','pip','install','torch==2.6.0+cpu','torchvision==0.21.0+cpu',
         '--index-url','https://mirrors.nju.edu.cn/pytorch/whl/cpu',
         '--extra-index-url','https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple')
    call('-m','pip','install','-r',str(SDK/'requirements.txt'))
    call('-m','pip','install','--no-deps',str(SDK/'rkllm_toolkit-1.3.1-cp310-cp310-linux_x86_64.whl'))
    call('-m','pip','check')
    call('-c','import torch,transformers; from rkllm.api import RKLLM; print(torch.__version__,transformers.__version__); print(RKLLM)')
    (REPORT/'wsl-pip-freeze.txt').write_bytes(subprocess.check_output([py,'-m','pip','freeze','--all']))

if __name__=='__main__':main()
