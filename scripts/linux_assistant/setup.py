"""Install PEFT into the cloned environment using the existing physical-NIC bridge."""
from pathlib import Path
import os
import subprocess
from scripts.common import ROOT, WORKSPACE, write_json
from scripts.bootstrap_environment import run, network_state
from scripts.direct_mirror import PhysicalMirror
from scripts.physical_proxy import PhysicalProxy, INSTALL_HOSTS


def main():
    report=WORKSPACE/'reports/linux-assistant-v1-20261019/environment'
    report.mkdir(parents=True,exist_ok=True)
    python=WORKSPACE/'.conda/qwen-lora-py311/python.exe'
    if not python.exists(): raise RuntimeError('Clone qwen-py311 into qwen-lora-py311 first')
    original=WORKSPACE/'.conda/qwen-py311/python.exe'
    before_packages=subprocess.check_output([str(original),'-m','pip','freeze','--all'],text=True)
    (report/'original-before.txt').write_text(before_packages,encoding='utf8')
    network_before=network_state();write_json(report/'network-before.json',network_before)
    mirror=PhysicalMirror(report/'install-network.jsonl',interface='WLAN',allowed_hosts=INSTALL_HOSTS)
    try:
        with PhysicalProxy(mirror) as proxy:
            env=dict(os.environ)
            for key in list(env):
                if key.lower() in {'http_proxy','https_proxy','all_proxy','no_proxy','pip_index_url','pip_extra_index_url','pip_trusted_host','requests_ca_bundle','curl_ca_bundle'}: del env[key]
            env.update(HTTP_PROXY=proxy.url,HTTPS_PROXY=proxy.url,NO_PROXY='localhost,127.0.0.1',
                PIP_CONFIG_FILE=os.devnull,PIP_INDEX_URL='https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple',
                PIP_CACHE_DIR=str(WORKSPACE/'.cache/pip'),PIP_DISABLE_PIP_VERSION_CHECK='1',
                PYTHONUTF8='1',PYTHONPATH=str(ROOT))
            # Keep the already-tested Torch/Transformers stack fixed during resolution.
            constraints=report/'constraints.txt'
            constraints.write_text('torch==2.12.1+cu130\ntransformers==5.8.0\ntokenizers==0.22.2\nnumpy==1.26.4\n',encoding='utf8')
            run([str(python),'-m','pip','install','--constraint',str(constraints),'peft','datasets==4.1.1','pyarrow==25.0.1'],env,report,'install')
            run([str(python),'-m','pip','check'],env,report,'check')
            run([str(python),'-c','import peft,torch,transformers,datasets; print(peft.__version__,torch.__version__,transformers.__version__); print(torch.cuda.get_device_name(0))'],env,report,'imports')
            (report/'requirements.lock.txt').write_text(subprocess.check_output([str(python),'-m','pip','freeze','--all'],env=env,text=True),encoding='utf8')
    finally:
        after_packages=subprocess.check_output([str(original),'-m','pip','freeze','--all'],text=True)
        network_after=network_state();write_json(report/'network-after.json',network_after)
        write_json(report/'preservation.json',{'original_environment_unchanged':before_packages==after_packages,'network_unchanged':network_before==network_after})
        if before_packages!=after_packages or network_before!=network_after: raise RuntimeError('Protected environment or network changed')


if __name__=='__main__':main()
