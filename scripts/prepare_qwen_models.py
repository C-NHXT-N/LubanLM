"""Download pinned official Qwen files from mirrors over the physical Windows NIC."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import threading
import time
from urllib.parse import quote

from scripts.common import WORKSPACE, sha256, write_json
from scripts.bootstrap_environment import network_state
from scripts.direct_mirror import PhysicalMirror

REPOSITORIES = ['Qwen/Qwen3-4B-Instruct-2507', 'Qwen/Qwen3.5-2B']
REVISIONS = {'Qwen/Qwen3-4B-Instruct-2507': 'cdbee75f17c01a7cc42f958dc650907174af0554',
             'Qwen/Qwen3.5-2B': '15852e8c16360a2fea060d615a32b45270f8a8fc'}
MODEL_HOSTS = {'dns.alidns.com','hf-mirror.com','modelscope.cn',
               'cdn-lfs-cn-1.modelscope.cn','cdn-lfs-cn-2.modelscope.cn'}


def git_blob(data):
    return hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--interface',default='WLAN')
    parser.add_argument('--model', choices=REPOSITORIES, help='download one pinned model; default: both')
    parser.add_argument('--tokenizer-only', action='store_true', help='fetch text tokenizer/configuration only, without model weights')
    args=parser.parse_args()
    report=args.report.resolve(); report.mkdir(parents=True,exist_ok=True)
    before=network_state();write_json(report/'network-before.json',before)
    mirror=PhysicalMirror(report/'download-network.jsonl',interface=args.interface,allowed_hosts=MODEL_HOSTS)
    status={'completed':False,'physical_adapter':mirror.adapter,'models':[], 'tokenizer_only':args.tokenizer_only}
    jobs=[]
    lock=threading.Lock()
    try:
        for repo in ([args.model] if args.model else REPOSITORIES):
            hf=json.loads(mirror.bytes(f'https://hf-mirror.com/api/models/{repo}/revision/{REVISIONS[repo]}?blobs=true'))
            if hf.get('sha') != REVISIONS[repo]:
                raise RuntimeError('Model metadata revision does not match the textbook baseline')
            ms=json.loads(mirror.bytes(f'https://modelscope.cn/api/v1/models/{repo}/repo/files?Revision=master&Recursive=true'))
            if not ms.get('Success'): raise RuntimeError(f'ModelScope metadata failed for {repo}')
            name=repo.split('/')[-1]; destination=WORKSPACE/'models/Qwen'/name
            destination.mkdir(parents=True,exist_ok=True)
            write_json(report/(name+'-hf-metadata.json'),hf)
            write_json(report/(name+'-modelscope-metadata.json'),ms)
            msfiles={r['Path']:r for r in ms['Data']['Files'] if r['Type']=='blob'}
            model={'repo':repo,'hf_revision':hf['sha'],'directory':str(destination),'files':[],
                   'scope':'tokenizer-only' if args.tokenizer_only else 'full-model'}
            status['models'].append(model)
            for row in hf['siblings']:
                name=row['rfilename']
                if name=='.gitattributes': continue
                if args.tokenizer_only and name not in {'config.json', 'tokenizer.json', 'tokenizer_config.json',
                    'special_tokens_map.json', 'chat_template.jinja', 'generation_config.json',
                    'added_tokens.json', 'vocab.json', 'merges.txt'}:
                    continue
                path=(destination/name).resolve()
                if not path.is_relative_to(destination.resolve()):raise ValueError('Invalid repository filename')
                job={'repo':repo,'name':name,'path':path,'row':row,'model':model}
                if row.get('lfs'):
                    alternate=msfiles.get(name)
                    if not alternate or alternate['Sha256']!=row['lfs']['sha256'] or alternate['Size']!=row['size']:
                        raise RuntimeError(f'Official mirror content mismatch: {repo}/{name}')
                    job.update(url=f'https://modelscope.cn/api/v1/models/{repo}/repo?Revision={alternate["Revision"]}&FilePath={quote(name,safe="/")}',
                               expected_sha256=row['lfs']['sha256'],source='ModelScope official Qwen replica',source_revision=alternate['Revision'])
                else:
                    job.update(url=f'https://hf-mirror.com/{repo}/resolve/{hf["sha"]}/{quote(name,safe="/")}',
                               source='HF-Mirror',source_revision=hf['sha'])
                jobs.append(job)
        total=sum(j['row']['size'] for j in jobs)
        reserve=100*1024**2 if args.tokenizer_only else 30*1024**3
        if shutil.disk_usage(WORKSPACE).free<total+reserve:raise RuntimeError('Insufficient disk reserve')
        print(f'Pinned {len(status["models"])} model(s): {total:,} bytes across {len(jobs)} files',flush=True)
        write_json(report/'download-status.json',status)

        def transfer(job):
            path=job['path'];row=job['row'];last=[0.0]
            if 'expected_sha256' in job:
                def progress():
                    now=time.monotonic()
                    if now-last[0]>30:
                        part=path.with_suffix(path.suffix+'.part')
                        size=path.stat().st_size if path.exists() else (part.stat().st_size if part.exists() else 0)
                        print(f'{job["repo"]}/{job["name"]}: {size:,}/{row["size"]:,} bytes',flush=True)
                        last[0]=now
                mirror.download(job['url'],path,row['size'],job['expected_sha256'],progress)
            else:
                if path.exists(): data=path.read_bytes()
                else: data=mirror.bytes(job['url'],maximum=max(20<<20,row['size']))
                if len(data)!=row['size'] or git_blob(data)!=row['blobId']:
                    raise RuntimeError(f'Git blob checksum mismatch: {job["repo"]}/{job["name"]}')
                if not path.exists():path.write_bytes(data)
            entry={'name':job['name'],'bytes':path.stat().st_size,'sha256':sha256(path),
                   'source':job['source'],'source_revision':job['source_revision'],'source_url':job['url']}
            with lock:
                job['model']['files'].append(entry)
                write_json(report/'download-status.json',status)
            print(f'Verified {job["repo"]}/{job["name"]}',flush=True)

        # Fetch small configuration files before the multi-GB weights.
        jobs.sort(key=lambda j:(j['row']['size']>100_000_000,-j['row']['size']))
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(transfer,job) for job in jobs]
            for future in as_completed(futures): future.result()
        for model in status['models']:
            model['files'].sort(key=lambda r:r['name'])
            write_json(Path(model['directory'])/'download_manifest.json',model)
        after=network_state();write_json(report/'network-after.json',after)
        if before!=after:raise RuntimeError('Network configuration changed; inspect snapshots')
        events=[json.loads(s) for s in (report/'download-network.jsonl').read_text(encoding='utf-8').splitlines()]
        status.update(completed=True,network_configuration_unchanged=True,
                      physical_connections=sum(e['event']=='physical_tcp_connection' for e in events),
                      download_bytes=total)
    except BaseException as exc:
        status['error']=str(exc)
        raise
    finally:
        write_json(report/'download-status.json',status)
        print(f'Download report: {report}',flush=True)


if __name__=='__main__':main()
