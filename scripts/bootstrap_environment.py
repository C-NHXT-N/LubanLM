"""Prepare the local Python 3.11 environment through physical-interface mirrors."""
from __future__ import annotations
import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from urllib.parse import urljoin, urlsplit, parse_qs, unquote

from scripts.direct_mirror import PhysicalMirror
from scripts.physical_proxy import PhysicalProxy, INSTALL_HOSTS

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def network_state():
    command = "[Console]::OutputEncoding=[Text.UTF8Encoding]::new(); [ordered]@{adapters=@(Get-NetAdapter|Sort-Object ifIndex|Select-Object Name,Status,ifIndex);routes=@(Get-NetRoute -AddressFamily IPv4|Where-Object DestinationPrefix -eq '0.0.0.0/0'|Sort-Object InterfaceIndex|Select-Object DestinationPrefix,NextHop,InterfaceIndex,RouteMetric);proxy=(Get-ItemProperty 'HKCU:\\Software\\Microsoft\\Windows\\CurrentVersion\\Internet Settings'|Select-Object ProxyEnable,ProxyServer,AutoConfigURL)}|ConvertTo-Json -Depth 5 -Compress"
    return json.loads(subprocess.check_output(['powershell.exe','-NoProfile','-Command',command], encoding='utf-8'))


def run(command, env, report, name):
    print(f'{name}: starting', flush=True)
    started = time.monotonic()
    with (report / (name + '.log')).open('w', encoding='utf-8') as log:
        process = subprocess.Popen(command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                time.sleep(5)
                if int(time.monotonic()-started) % 30 < 5:
                    print(f'{name}: running ({int(time.monotonic()-started)}s)', flush=True)
            if process.returncode:
                tail = (report / (name + '.log')).read_text(encoding='utf-8', errors='replace')[-6000:]
                raise RuntimeError(f'{name} failed ({process.returncode}):\n{tail}')
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=30)
    print(f'{name}: passed ({int(time.monotonic()-started)}s)', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--conda', type=Path, default=Path('C:/Tools/miniconda3/Scripts/conda.exe'))
    parser.add_argument('--environment', type=Path, default=WORKSPACE/'.conda/lubanlm-py311')
    parser.add_argument('--interface', default='WLAN')
    args = parser.parse_args()
    report = WORKSPACE/'reports'/time.strftime('environment-setup-%Y%m%d-%H%M%S')
    report.mkdir(parents=True)
    cache = WORKSPACE/'.cache/bootstrap'
    cache.mkdir(parents=True, exist_ok=True)
    python = args.environment/'python.exe'
    status = {'passed':False, 'environment':str(args.environment), 'report':str(report)}
    before = network_state()
    save(report/'network-before.json',before)
    protected = [WORKSPACE/'runs/pretrain/checkpoints/last.pt',WORKSPACE/'runs/sft/checkpoints/best.pt',WORKSPACE/'runs/sft/checkpoints/last.pt',WORKSPACE/'datasets/general-10b-teaching-v2/tokenizers/vocab-32000/tokenizer.json']
    baseline = {str(p):digest(p) for p in protected}
    save(report/'protected-files-before.json', baseline)
    mirror = PhysicalMirror(report/'network.jsonl', interface=args.interface, allowed_hosts=INSTALL_HOSTS)
    status['physical_adapter'] = mirror.adapter
    try:
        with PhysicalProxy(mirror) as proxy:
            env = dict(os.environ)
            for key in list(env):
                if key.lower() in {'http_proxy','https_proxy','all_proxy','no_proxy','pip_index_url','pip_extra_index_url','pip_trusted_host','requests_ca_bundle','curl_ca_bundle'}:
                    del env[key]
            env.update(HTTP_PROXY=proxy.url, HTTPS_PROXY=proxy.url, NO_PROXY='localhost,127.0.0.1',
                PIP_CONFIG_FILE=os.devnull, PIP_INDEX_URL='https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple',
                PIP_CACHE_DIR=str(WORKSPACE/'.cache/pip'), PIP_DISABLE_PIP_VERSION_CHECK='1',
                CONDA_PKGS_DIRS=str(WORKSPACE/'.cache/conda/pkgs'), CONDA_ENVS_PATH=str(WORKSPACE/'.conda'),
                CONDA_NUMBER_CHANNEL_NOTICES='0', PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1',
                PYTHONPATH=str(ROOT), HF_ENDPOINT='https://hf-mirror.com', HF_HOME=str(WORKSPACE/'.cache/huggingface'),
                HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                LUBAN_DATA=str(WORKSPACE/'datasets/general-10b-teaching-v2'),
                LUBAN_RUNS=str(WORKSPACE/'runs'), CUDA_VISIBLE_DEVICES='0',
                LUBAN_DOWNLOAD_INTERFACE=args.interface)
            config = report/'condarc.json'
            save(config, {'channels':['https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge'],
                'proxy_servers':{'http':proxy.url,'https':proxy.url},'ssl_verify':True,
                'report_errors':False,'number_channel_notices':0,'auto_update_conda':False})
            env['CONDARC']=str(config)
            if python.exists():
                version=subprocess.check_output([str(python),'-c','import sys;print("%d.%d"%sys.version_info[:2])'],env=env,text=True).strip()
                if version!='3.11':
                    raise RuntimeError(f'Existing environment is Python {version}; refusing overwrite')
                print('Reusing existing Python 3.11 environment',flush=True)
            else:
                if args.environment.exists() and any(args.environment.iterdir()):
                    raise RuntimeError('Incomplete existing environment; refusing overwrite')
                run([str(args.conda),'--no-plugins','create','--solver','classic','--prefix',str(args.environment),
                    '--override-channels','--channel','https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge',
                    '--no-default-packages','python=3.11','pip','-y'],env,report,'conda-create')
            installed = subprocess.check_output([str(python),'-c',"import importlib.metadata as m;print(m.version('torch') if any(d.metadata['Name'].lower()=='torch' for d in m.distributions()) else '')"],env=env,text=True).strip()
            if installed and installed != '2.12.1+cu130':
                raise RuntimeError(f'Existing torch {installed}; refusing to replace automatically')
            if not installed:
                index='https://mirrors.nju.edu.cn/pytorch/whl/cu130/torch/'
                page=mirror.bytes(index, maximum=4<<20).decode()
                links=[html.unescape(x) for x in re.findall(r'href="([^"]+)"',page)]
                links=[x for x in links if unquote(urlsplit(x).path).endswith('/torch-2.12.1+cu130-cp311-cp311-win_amd64.whl')]
                if len(links)!=1: raise RuntimeError('Pinned CUDA wheel missing or ambiguous')
                url=urljoin(index,links[0]); sha=parse_qs(urlsplit(url).fragment)['sha256'][0]
                if sha!='5ff38932260cb4d5a52170d955642f6ede17f565de64e62eaca12a875851471b':
                    raise RuntimeError('Pinned wheel digest changed')
                url=url.split('#')[0]
                connection,response=mirror.open(url,{'Range':'bytes=0-0'})
                try:
                    if response.status==206:
                        size=int(response.getheader('Content-Range').split('/')[-1])
                    elif response.status==200:
                        size=int(response.getheader('Content-Length'))
                    else: raise RuntimeError(f'Wheel metadata HTTP {response.status}')
                finally: connection.close()
                wheel=cache/'torch-2.12.1+cu130-cp311-cp311-win_amd64.whl'
                print(f'Downloading pinned CUDA wheel: {size} bytes',flush=True)
                last=[0.0]
                def progress():
                    now=time.monotonic()
                    if now-last[0]>30:
                        part=wheel.with_suffix('.whl.part')
                        print(f'CUDA wheel: {part.stat().st_size if part.exists() else 0}/{size} bytes',flush=True)
                        last[0]=now
                mirror.download(url,wheel,size,sha,progress)
                save(report/'torch-wheel.json',{'url':url,'sha256':sha,'bytes':size,'path':str(wheel)})
                run([str(python),'-m','pip','install',str(wheel)],env,report,'install-torch')
            run([str(python),'-m','pip','install','-r',str(ROOT/'configs/requirements-gpu.txt')],env,report,'install-project')
            run([str(python),'-m','pip','check'],env,report,'pip-check')
            for name, command in [('pip-freeze',[str(python),'-m','pip','freeze','--all']),
                                  ('conda-explicit',[str(args.conda),'--no-plugins','list','--prefix',str(args.environment),'--explicit'])]:
                (report/(name+'.txt')).write_text(subprocess.check_output(command,env=env,text=True),encoding='utf-8')
            run([str(python),'-m','scripts.verify_environment','--output',str(report/'validation.json'),
                '--baseline',str(report/'protected-files-before.json')],env,report,'validate-gpu-and-models')
        after=network_state()
        save(report/'network-after.json',after)
        if before!=after: raise RuntimeError('System network configuration changed during setup; inspect snapshots')
        events=[json.loads(s) for s in (report/'network.jsonl').read_text(encoding='utf-8').splitlines()]
        rejected=[e for e in events if e['event'] in ('host_rejected','proxy_rejected')]
        if rejected: raise RuntimeError('Unapproved download host requested; inspect network audit')
        connections=[e for e in events if e['event']=='physical_tcp_connection']
        if not connections or any(e['interface_index']!=mirror.adapter['index'] or e['local_ip']!=mirror.adapter['ip'] for e in connections):
            raise RuntimeError('Physical connection audit failed')
        status.update(passed=True,network_configuration_unchanged=True,physical_connections=len(connections))
        print('Environment, GPU and three checkpoint checks PASSED',flush=True)
    except BaseException as exc:
        status['error']=str(exc)
        raise
    finally:
        try:
            after = network_state()
            save(report/'network-after.json', after)
            status['network_configuration_unchanged'] = before == after
            status['protected_files_unchanged'] = all(digest(Path(p)) == expected for p, expected in baseline.items())
        except Exception as exc:
            status['final_audit_error'] = str(exc)
        save(report/'summary.json',status)
        print(f'Report: {report}',flush=True)


if __name__=='__main__':
    main()
