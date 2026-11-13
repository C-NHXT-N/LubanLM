"""SHA256-checked SSH upload with prefix-verified resumable .part files; no network reconfiguration."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time
from scripts.common import sha256,write_json
from scripts.linux_assistant.common import config,DEFAULT_CONFIG

REMOTE = r'''
import hashlib,json,os,pathlib,sys
root=pathlib.Path(sys.argv[1]).resolve();relative=pathlib.PurePosixPath(sys.argv[2]);expected=sys.argv[3];total=int(sys.argv[4])
if relative.is_absolute() or '..' in relative.parts:raise ValueError('Invalid relative path')
target=(root/relative).resolve()
if not target.is_relative_to(root):raise ValueError('Path escapes root')
target.parent.mkdir(parents=True,exist_ok=True)
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while chunk:=f.read(8<<20):h.update(chunk)
 return h.hexdigest()
if target.exists():
 if target.stat().st_size!=total or digest(target)!=expected:raise ValueError('Refusing to overwrite different completed file')
 print(json.dumps({'offset':total,'prefix_sha256':expected,'complete':True}),flush=True);sys.exit(0)
partial=target.with_name(target.name+'.part');offset=partial.stat().st_size if partial.exists() else 0
if offset>total:raise ValueError('Partial file is larger than source')
prefix=digest(partial) if partial.exists() else hashlib.sha256(b'').hexdigest()
print(json.dumps({'offset':offset,'prefix_sha256':prefix,'complete':False}),flush=True)
with partial.open('ab') as f:
 remaining=total-offset
 while remaining:
  chunk=sys.stdin.buffer.read(min(1<<20,remaining))
  if not chunk:raise EOFError('Interrupted upload retained as .part')
  f.write(chunk);remaining-=len(chunk)
 f.flush();os.fsync(f.fileno())
if digest(partial)!=expected:raise ValueError('Final SHA256 mismatch; .part retained')
if target.exists():raise FileExistsError('Destination appeared during upload')
partial.rename(target)
print(json.dumps({'verified':True,'sha256':expected,'bytes':total}),flush=True)
'''


def upload(source,relative,digest,remote_root,host,user):
    encoded=base64.b64encode(REMOTE.encode()).decode()
    code='import base64;exec(base64.b64decode('+repr(encoded)+'))'
    command=shlex.join(['python3','-c',code,remote_root,relative,digest,str(source.stat().st_size)])
    process=subprocess.Popen(['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',
        '-o','HostKeyAlias=cat','-o','ConnectTimeout=15',f'{user}@{host}',command],
        stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=None)
    try:
        line=process.stdout.readline()
        if not line:raise RuntimeError('SSH upload handshake failed; configure key login first')
        handshake=json.loads(line);offset=handshake['offset']
        h=hashlib.sha256()
        with source.open('rb') as f:
            left=offset
            while left:
                chunk=f.read(min(8<<20,left))
                if not chunk:raise ValueError('Remote offset exceeds source')
                h.update(chunk);left-=len(chunk)
            if h.hexdigest()!=handshake['prefix_sha256']:raise ValueError('Remote partial prefix mismatch; file retained for inspection')
            if not handshake['complete']:
                while chunk:=f.read(1<<20):process.stdin.write(chunk)
                process.stdin.flush();process.stdin.close()
                result=json.loads(process.stdout.readline())
                if result.get('sha256')!=digest or not result.get('verified'):raise ValueError('Remote verification failed')
        if process.wait(timeout=30)!=0:raise RuntimeError('Remote upload process failed')
        return {'path':relative,'verified':True,'sha256':digest,'resumed_from_bytes':offset}
    finally:
        if process.poll() is None:process.terminate();process.wait(timeout=15)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    p.add_argument('--host',default='100.64.207.34');p.add_argument('--user',default='cat')
    p.add_argument('--remote-root',default='/home/cat/lubanlm-linux-assistant-v1')
    p.add_argument('--verify-only',action='store_true');p.add_argument('--final-round',action='store_true')
    a=p.parse_args();c=config(a.config)
    root=c['export']/'board-bundle';manifest=json.loads((root/'bundle-manifest.json').read_text(encoding='utf8'))
    files={**manifest['files'],'bundle-manifest.json':sha256(root/'bundle-manifest.json')}
    sources={name:root/name for name in files}
    if a.final_round:
        lock_path=c['report']/'candidate-lock.json';lock=json.loads(lock_path.read_text(encoding='utf8'))
        if lock['model_sha256']!=manifest['files']['model.rkllm']:raise ValueError('Final lock does not match board model')
        sources['final-questions.jsonl']=c['report']/'final-questions.jsonl'
        sources['candidate-lock.json']=lock_path
        files.update({'final-questions.jsonl':lock['questions_sha256'],'candidate-lock.json':sha256(lock_path)})
    for name,digest in files.items():
        path=sources[name].resolve()
        allowed=root.resolve() if name not in {'final-questions.jsonl','candidate-lock.json'} else c['report'].resolve()
        if not path.is_relative_to(allowed) or sha256(path)!=digest:raise ValueError('Bundle/locked final input changed: '+name)
    if a.verify_only:print('Local bundle verified; no network operation.');return
    results=[];record=c['report']/('final-transfer.json' if a.final_round else 'transfer.json')
    for name,digest in files.items():
        print('Transfer '+name,flush=True)
        results.append(upload(sources[name],name,digest,a.remote_root,a.host,a.user))
        write_json(record,{'completed':False,'host':a.host,'files':results})
    write_json(record,{'completed':True,'host':a.host,'remote_root':a.remote_root,'files':results,'network_configuration_changed':False})


if __name__=='__main__':main()
