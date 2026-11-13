"""Download and hash-lock Python 3.10 ARM64 wheels for offline board installation."""
from email.parser import BytesParser
import argparse
import os
from pathlib import Path
import sys
from zipfile import ZipFile
from packaging.requirements import Requirement
from packaging.markers import default_environment
from packaging.utils import canonicalize_name
from scripts.common import ROOT,WORKSPACE,write_json,sha256
from scripts.bootstrap_environment import run
from scripts.direct_mirror import PhysicalMirror
from scripts.physical_proxy import PhysicalProxy,INSTALL_HOSTS
from scripts.linux_assistant.common import config,DEFAULT_CONFIG


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    c=config(parser.parse_args().config);dest=c['export']/'board-wheels';dest.mkdir(parents=True,exist_ok=True)
    report=c['report']/'board-dependencies';report.mkdir(exist_ok=True)
    if (dest/'requirements.lock.txt').exists():raise FileExistsError('Board dependencies are already locked')
    mirror=PhysicalMirror(report/'network.jsonl',interface='WLAN',allowed_hosts=INSTALL_HOSTS)
    with PhysicalProxy(mirror) as proxy:
        env=dict(os.environ)
        for key in list(env):
            if key.lower() in {'http_proxy','https_proxy','all_proxy','no_proxy','pip_index_url','pip_extra_index_url','pip_trusted_host','requests_ca_bundle','curl_ca_bundle'}:del env[key]
        env.update(HTTP_PROXY=proxy.url,HTTPS_PROXY=proxy.url,NO_PROXY='localhost,127.0.0.1',
            PIP_CONFIG_FILE=os.devnull,PIP_INDEX_URL='https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple',
            PIP_CACHE_DIR=str(WORKSPACE/'.cache/pip'),PIP_DISABLE_PIP_VERSION_CHECK='1')
        command=[sys.executable,'-m','pip','download','--dest',str(dest),'--only-binary=:all:',
             '--platform','manylinux2014_aarch64','--platform','manylinux_2_28_aarch64',
             '--python-version','310','--implementation','cp','--abi','cp310','--abi','abi3']
        run([*command,'-r',str(ROOT/'deploy/linux_assistant/requirements.txt')],env,report,'download')
        # pip --platform chooses wheels, but environment markers can still reflect the host.
        target=default_environment();target.update(python_version='3.10',python_full_version='3.10.12',
            implementation_version='3.10.12',os_name='posix',platform_machine='aarch64',
            platform_system='Linux',sys_platform='linux',platform_python_implementation='CPython',
            platform_release='unknown',platform_version='unknown',extra='')
        for attempt in range(3):
            installed={};requires=[]
            for wheel in dest.glob('*.whl'):
                with ZipFile(wheel) as archive:
                    metadata=BytesParser().parsebytes(archive.read(next(n for n in archive.namelist() if n.endswith('.dist-info/METADATA'))))
                installed[canonicalize_name(metadata['Name'])]=metadata['Version']
                requires.extend(Requirement(r) for r in metadata.get_all('Requires-Dist',[]))
            missing=sorted({str(r).split(';')[0].strip() for r in requires
                if (not r.marker or r.marker.evaluate(target)) and
                (canonicalize_name(r.name) not in installed or installed[canonicalize_name(r.name)] not in r.specifier)})
            if not missing:break
            run([*command,*missing],env,report,f'target-closure-{attempt}')
        else:raise ValueError('Target dependency closure is not resolved; do not package wheels')
        write_json(report/'target-marker-audit.json',{'passed':True,'target':target,'packages':installed,
            'target_profile_not_board_probe':True})
    records=[]
    for wheel in sorted(dest.glob('*.whl')):
        with ZipFile(wheel) as archive:
            names=[n for n in archive.namelist() if n.endswith('.dist-info/METADATA')]
            if len(names)!=1:raise ValueError('Ambiguous wheel metadata')
            metadata=BytesParser().parsebytes(archive.read(names[0]))
        records.append({'name':metadata['Name'],'version':metadata['Version'],'file':wheel.name,'sha256':sha256(wheel)})
    if len({r['name'].lower().replace('_','-') for r in records})!=len(records):raise ValueError('Multiple versions of dependency')
    (dest/'requirements.lock.txt').write_text(''.join(f'{r["name"]}=={r["version"]} --hash=sha256:{r["sha256"]}\n' for r in records),encoding='utf8')
    write_json(report/'manifest.json',{'completed':True,'python':'3.10','platform':'aarch64 Linux glibc>=2.28',
               'packages':records,'board_import_test':None,'contains_torch':any(r['name'].lower()=='torch' for r in records)})
    print(f'Locked {len(records)} packages for offline board installation.')


if __name__=='__main__':main()
