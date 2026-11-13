"""Prepare a separate modern Transformers environment without changing the 50M env."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess

from scripts.common import ROOT, WORKSPACE, sha256, write_json
from scripts.bootstrap_environment import run, network_state
from scripts.direct_mirror import PhysicalMirror
from scripts.physical_proxy import PhysicalProxy, INSTALL_HOSTS


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args();report=args.report.resolve();report.mkdir(parents=True,exist_ok=True)
    prefix=WORKSPACE/'.conda/qwen-py311'; python=prefix/'python.exe'
    conda='C:/Tools/miniconda3/Scripts/conda.exe'
    before=network_state();write_json(report/'environment-network-before.json',before)
    mirror=PhysicalMirror(report/'environment-network.jsonl',interface='WLAN',allowed_hosts=INSTALL_HOSTS)
    with PhysicalProxy(mirror) as proxy:
        env=dict(os.environ)
        for key in list(env):
            if key.lower() in {'http_proxy','https_proxy','all_proxy','no_proxy','pip_index_url','pip_extra_index_url','pip_trusted_host','requests_ca_bundle','curl_ca_bundle'}:del env[key]
        env.update(HTTP_PROXY=proxy.url,HTTPS_PROXY=proxy.url,NO_PROXY='localhost,127.0.0.1',
                   PIP_CONFIG_FILE=os.devnull,PIP_INDEX_URL='https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple',
                   PIP_CACHE_DIR=str(WORKSPACE/'.cache/pip'),PIP_DISABLE_PIP_VERSION_CHECK='1',
                   CONDA_PKGS_DIRS=str(WORKSPACE/'.cache/conda/pkgs'),CONDA_ENVS_PATH=str(WORKSPACE/'.conda'),
                   PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(ROOT))
        config=report/'qwen-condarc.json'
        write_json(config,{'channels':['https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge'],
                          'proxy_servers':{'http':proxy.url,'https':proxy.url},'ssl_verify':True,'number_channel_notices':0})
        env['CONDARC']=str(config)
        if not python.exists():
            if prefix.exists() and any(prefix.iterdir()):raise RuntimeError('Refusing to overwrite incomplete Qwen env')
            run([conda,'--no-plugins','create','--solver','classic','--prefix',str(prefix),'--override-channels',
                 '--channel','https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge',
                 '--offline','--no-default-packages','python=3.11','pip','-y'],env,report,'qwen-conda-create')
        version=subprocess.check_output([str(python),'-c','import sys;print("%d.%d"%sys.version_info[:2])'],env=env,text=True).strip()
        if version!='3.11':raise RuntimeError('Existing Qwen environment is not Python 3.11')
        wheel=WORKSPACE/'.cache/bootstrap/torch-2.12.1+cu130-cp311-cp311-win_amd64.whl'
        if sha256(wheel)!='5ff38932260cb4d5a52170d955642f6ede17f565de64e62eaca12a875851471b':
            raise RuntimeError('Cached CUDA wheel checksum mismatch')
        run([str(python),'-m','pip','install',str(wheel),'transformers==5.8.0','accelerate>=1.1.0',
             'numpy==1.26.4','pillow','safetensors>=0.5.3'],env,report,'qwen-install')
        run([str(python),'-m','pip','check'],env,report,'qwen-pip-check')
        env.update(HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1')
        run([str(python),'-c',
             "import torch,transformers;from transformers import Qwen3ForCausalLM,Qwen3_5ForConditionalGeneration;print(torch.__version__,transformers.__version__);print(torch.cuda.get_device_name(0));print('Both model classes imported')"],
             env,report,'qwen-import-check')
        (report/'qwen-pip-freeze.txt').write_text(subprocess.check_output([str(python),'-m','pip','freeze','--all'],env=env,text=True),encoding='utf-8')
    after=network_state();write_json(report/'environment-network-after.json',after)
    if before!=after:raise RuntimeError('Network configuration changed')
    write_json(report/'environment-status.json',{'completed':True,'environment':str(prefix),'network_configuration_unchanged':True})
    print(f'Qwen environment ready: {prefix}',flush=True)


if __name__=='__main__':main()
