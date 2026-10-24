"""Install the pinned official Toolkit into a workspace WSL venv via physical mirrors."""
import os
import json
import shutil
import subprocess
import argparse
from scripts.common import WORKSPACE, ROOT, write_json
from scripts.direct_mirror import PhysicalMirror
from scripts.physical_proxy import PhysicalProxy,INSTALL_HOSTS
from scripts.bootstrap_environment import network_state,run

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--gpu',action='store_true');args=parser.parse_args()
    report=WORKSPACE/'reports/rk3576-qwen-20261018';report.mkdir(parents=True,exist_ok=True)
    label='gpu-setup' if args.gpu else 'setup'
    before=network_state();write_json(report/f'{label}-network-before.json',before)
    bootstrap=WORKSPACE/'.cache/rkllm-bootstrap'
    import pip
    shutil.copytree(os.path.dirname(pip.__file__),bootstrap/'pip',dirs_exist_ok=True)
    env=dict(os.environ);env['PYTHONUTF8']='1'
    mirror=PhysicalMirror(report/f'{label}-network.jsonl',interface='WLAN',allowed_hosts=INSTALL_HOSTS)
    with PhysicalProxy(mirror) as proxy:
        cmd=['wsl','-d','Ubuntu-22.04','--','python3',
             '/mnt/c/Lab/LubanLM-lab/LubanLM/scripts/wsl_physical_relay.py',
             '--port',str(proxy.server.server_port),'--windows-script',str(ROOT/'scripts/wsl_physical_relay.py').replace('\\','/'),
             '--','python3','/mnt/c/Lab/LubanLM-lab/LubanLM/scripts/'+('setup_rkllm_gpu_linux.py' if args.gpu else 'setup_rkllm_linux.py')]
        run(cmd,env,report,'wsl-gpu-install' if args.gpu else 'wsl-toolkit-install')
    after=network_state();write_json(report/f'{label}-network-after.json',after)
    write_json(report/f'{label}-status.json',{'completed':True,'network_configuration_unchanged':before==after})

if __name__=='__main__':main()
