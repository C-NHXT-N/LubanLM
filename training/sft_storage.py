"""Local memory-mapped token cache; JSONL and sampler interfaces stay intact."""
import json
import os
from pathlib import Path
import shutil
import uuid
import numpy as np
from scripts.common import read_json,write_json,sha256
from scripts.worker_health import acquire_lock,release_lock

class MappedRows:
    def __init__(self,path):
        path=Path(path);source_hash=sha256(path)
        folder=path.parent/'.sft-cache'/('i64-v1-'+source_hash)
        folder.mkdir(parents=True,exist_ok=True);manifest=folder/'manifest.json'
        owner=acquire_lock(folder/'build.lock',('training.train','scripts.benchmark_sft_10b'))
        if owner is None:raise RuntimeError('RESOURCE_WAIT: SFT cache is being built')
        try:
            if not manifest.exists():self.build(path,folder,source_hash)
            meta=read_json(manifest)
            if meta['source_sha256']!=source_hash:raise ValueError('SFT cache source mismatch')
            for name in ('ids','labels'):
                f=folder/(name+'.bin')
                if not f.exists() or f.stat().st_size!=meta['tokens']*8 or sha256(f)!=meta[name+'_sha256']:
                    manifest.unlink();self.build(path,folder,source_hash);meta=read_json(manifest);break
            self.lengths=np.asarray(meta['lengths'],dtype=np.int64)
            self.offsets=np.empty(len(self.lengths)+1,dtype=np.int64);self.offsets[0]=0
            np.cumsum(self.lengths,out=self.offsets[1:])
            self.tasks=meta['tasks'];self.version=meta['encoding_version']
            self.ids=np.memmap(folder/'ids.bin',mode='r',dtype=np.int64)
            self.labels=np.memmap(folder/'labels.bin',mode='r',dtype=np.int64)
        finally:release_lock(folder/'build.lock',owner)
    @staticmethod
    def build(path,folder,source_hash):
        lengths=[];tasks=[];versions=set()
        # First pass counts only; no text or token arrays survive the current row.
        with path.open(encoding='utf-8') as f:
            for line in f:
                if not line.strip():continue
                row=json.loads(line);length=len(row['input_ids'])
                if length<2 or len(row['labels'])!=length:raise ValueError('Invalid SFT pair lengths')
                lengths.append(length);tasks.append(row.get('task','public'));versions.add(row.get('encoding_version','legacy-joint-v1'))
        if not lengths or len(versions)!=1:raise ValueError('Empty or mixed SFT encoding versions')
        needed=sum(lengths)*16
        if shutil.disk_usage(folder).free-needed < int(os.environ.get('LUBAN_SFT_CACHE_DISK_FLOOR',20<<30)):
            raise RuntimeError('RESOURCE_WAIT: SFT token cache disk reserve')
        tag=uuid.uuid4().hex;temporary=[folder/(name+'.'+tag+'.tmp') for name in ('ids','labels')]
        try:
            with temporary[0].open('wb') as ids,temporary[1].open('wb') as labels,path.open(encoding='utf-8') as f:
                for line in f:
                    if not line.strip():continue
                    row=json.loads(line)
                    np.asarray(row['input_ids'],dtype=np.int64).tofile(ids)
                    np.asarray(row['labels'],dtype=np.int64).tofile(labels)
                for stream in (ids,labels):stream.flush();os.fsync(stream.fileno())
            if sha256(path)!=source_hash:raise ValueError('SFT source changed while building cache')
            for name,temp in zip(('ids','labels'),temporary):temp.replace(folder/(name+'.bin'))
            write_json(folder/'manifest.json',dict(version=1,source_sha256=source_hash,tokens=sum(lengths),lengths=lengths,tasks=tasks,encoding_version=next(iter(versions)),ids_sha256=sha256(folder/'ids.bin'),labels_sha256=sha256(folder/'labels.bin')))
        finally:
            for temp in temporary:temp.unlink(missing_ok=True)
    def __len__(self):return len(self.lengths)
    def __getitem__(self,index):
        if isinstance(index,slice):return [self[i] for i in range(*index.indices(len(self)))]
        if index<0:index+=len(self)
        if not 0<=index<len(self):raise IndexError(index)
        start,end=self.offsets[index:index+2]
        return dict(task=self.tasks[index],encoding_version=self.version,input_ids=self.ids[start:end],labels=self.labels[start:end])
    def close(self):
        for array in (self.ids,self.labels):array._mmap.close()
