"""Rolling verified raw cache, file-transaction recovery, global lexical dedup."""
from scripts.common import ROOT,read_json,write_json,sha256,utc
from scripts.prepare_general_2b import valid_text,key
from scripts.direct_mirror import PhysicalMirror
from eval.chat_suite import normalize,excluded_questions
from sft.template import CHAT_TEMPLATE
from scripts.corpus_quality import clean_document
from scripts.policy_versions import pretrain_identity as policy_identity
from tokenizers import Tokenizer
from pathlib import Path
from collections import Counter,deque
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import random
import shutil
import sqlite3
import time
import hashlib
import os
import numpy as np
import pyarrow.parquet as pq

class Writer:
    def __init__(self,folder,tokenizer_hash,state=None):
        self.folder=folder;folder.mkdir(parents=True,exist_ok=True)
        self.tokenizer_hash=tokenizer_hash
        self.shards=list((state or {}).get('shards',[]))
        self.tokens=(state or {}).get('tokens',0)
        self.documents=(state or {}).get('documents',0)
        self.parts=[];self.buffer=0
        if self.tokens!=sum(int(r['tokens']) for r in self.shards):raise ValueError('Committed writer extent mismatch')
        for row in self.shards:
            path=Path(row['path'])
            if path.stat().st_size!=2*row['tokens'] or sha256(path)!=row['sha256']:raise ValueError('Committed shard changed')
        committed={Path(r['path']).name for r in self.shards}
        for p in folder.glob('*.bin'):
            if p.name not in committed:p.unlink()
    def add(self,ids):
        if min(ids)<0 or max(ids)>=32000:raise ValueError('Invalid token range')
        self.parts.append(np.asarray(ids,dtype=np.uint16));self.buffer+=len(ids)
        self.tokens+=len(ids);self.documents+=1
        if self.buffer>=64000000:self.flush()
    def flush(self):
        if not self.parts:return
        path=self.folder/f'{len(self.shards):05d}.bin'
        temporary=path.with_suffix('.bin.tmp')
        with temporary.open('wb') as stream:
            np.concatenate(self.parts).tofile(stream);stream.flush();os.fsync(stream.fileno())
        temporary.replace(path)
        self.shards.append(dict(path=str(path),tokens=self.buffer,sha256=sha256(path)))
        self.parts=[];self.buffer=0
    def state(self):
        self.flush()
        return dict(dtype='uint16',tokenizer_sha256=self.tokenizer_hash,shards=self.shards,tokens=self.tokens,documents=self.documents)

def fingerprint(text):
    # 128 evenly spaced character shingles; lexical similarity, not semantics.
    if len(text)<5:return 0
    positions=np.linspace(0,len(text)-5,min(128,len(text)-4),dtype=np.int64)
    values=np.array([int.from_bytes(hashlib.blake2b(text[int(i):int(i)+5].encode(),digest_size=8).digest(),'little') for i in positions],dtype=np.uint64)
    bits=((values[:,None]>>np.arange(64,dtype=np.uint64))&1).sum(axis=0)>=len(values)/2
    return sum(int(v)<<i for i,v in enumerate(bits))

class Preparation:
    def __init__(self,plan_path):
        self.plan_path=Path(plan_path);self.plan=read_json(plan_path)
        self.policy=policy_identity(plan_path)
        self.data,self.runs=Path(self.plan['data_root']),Path(self.plan['runs_root'])
        self.data.mkdir(parents=True,exist_ok=True);self.runs.mkdir(parents=True,exist_ok=True)
        self.start=time.monotonic();self.report=self.runs/'reports/preparation_progress.json'
        self.last_cache_check=0
        self.mirror=PhysicalMirror(self.runs/'reports/network.jsonl')
        self.db=sqlite3.connect(self.data/'dedup.sqlite',timeout=60)
        self.db.execute('PRAGMA journal_mode=WAL');self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('PRAGMA cache_size=-65536')
        self.db.executescript('CREATE TABLE IF NOT EXISTS docs(hash TEXT PRIMARY KEY,lead TEXT UNIQUE,split TEXT); CREATE TABLE IF NOT EXISTS near(bucket INTEGER,fp TEXT,hash TEXT); CREATE TABLE IF NOT EXISTS checkpoints(name TEXT PRIMARY KEY,state TEXT);')
        self.db.execute('CREATE INDEX IF NOT EXISTS near_covering ON near(bucket,fp)')
        self.db.execute('DROP INDEX IF EXISTS near_bucket');self.db.commit()
        self.db.execute('CREATE TABLE IF NOT EXISTS metadata(name TEXT PRIMARY KEY,value TEXT)')
        saved=self.db.execute("SELECT value FROM metadata WHERE name='policy'").fetchone()
        if saved and json.loads(saved[0])!=self.policy:raise ValueError('Cleaning policy changed; use a new data version')
        self.db.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)',('policy',json.dumps(self.policy)));self.db.commit()
        frozen=ROOT/'docs/run-results/general-10b-20261005/frozen-test.json'
        self.blocked=set(excluded_questions())
        if frozen.exists():self.blocked.update(normalize(r['question']) for r in read_json(frozen)['questions'])
        self.blocked=sorted(q for q in self.blocked if len(q)>=12)
        legacy=Path(self.plan['legacy_data'])/'tokenizers/vocab-32000'
        if sha256(legacy/'tokenizer.json')!=self.plan['tokenizer_sha256']:raise ValueError('Tokenizer changed')
        shutil.copytree(legacy,self.data/'tokenizers/vocab-32000',dirs_exist_ok=True)
        tokenizer_config=self.data/'tokenizers/vocab-32000/tokenizer_config.json'
        if tokenizer_config.exists():
            config=read_json(tokenizer_config);config['chat_template']=CHAT_TEMPLATE;write_json(tokenizer_config,config)
        self.tokenizer=Tokenizer.from_file(str(legacy/'tokenizer.json'))

    def guard(self):
        from scripts.resource_guard import check_resources
        check_resources(self.plan,'pretrain')
        if time.monotonic()-self.start>self.plan['preparation_hours_cap']*3600:raise RuntimeError('Preparation time cap')
        if shutil.disk_usage(self.data).free<self.plan['disk_free_bytes_floor']:raise RuntimeError('Disk reserve reached')
        if time.monotonic()-self.last_cache_check>=5:
            raw_bytes=0
            for folder in (self.data/'raw-cache',Path(self.plan.get('sft_raw_root',str(self.data/'sft-raw')))):
                if not folder.exists():continue
                for path in folder.rglob('*'):
                    try:
                        if path.is_file():raw_bytes+=path.stat().st_size
                    except FileNotFoundError:
                        pass  # A committed raw file may be removed concurrently.
            if raw_bytes>self.plan['raw_cache_bytes_cap']:raise RuntimeError('Combined raw/partial cache exceeds 5 GiB')
            self.last_cache_check=time.monotonic()

    def progress(self,**values):
        write_json(self.report,dict(updated_utc=utc(),elapsed_seconds=round(time.monotonic()-self.start),**values))

    def previous(self,name):
        row=self.db.execute('SELECT state FROM checkpoints WHERE name=?',(name,)).fetchone()
        return json.loads(row[0]) if row else None

    def commit(self,name,value):
        value['policy_identity']=self.policy
        self.db.execute('INSERT OR REPLACE INTO checkpoints VALUES (?,?)',(name,json.dumps(value,ensure_ascii=False)))
        self.db.commit()

    def wiki(self):
        previous=self.previous('wiki')
        if previous and previous.get('complete'):return previous
        legacy=Path(self.plan['legacy_data'])
        previous=previous or {};writers={s:Writer(self.data/f'processed/wiki/{s}',self.plan['tokenizer_sha256'],previous.get(s)) for s in ('train','validation')}
        cursors=previous.get('cursors',{});counts=Counter(previous.get('counts',{}));samples=previous.get('samples',[])
        for split in ('train','validation'):
            with (legacy/f'processed/wiki-{split}.jsonl').open(encoding='utf-8') as stream:
                self.db.execute('BEGIN')
                for i,line in enumerate(stream):
                    if i<cursors.get(split,0):continue
                    if writers[split].tokens >= (self.plan['wiki_train_tokens_target'] if split=='train' else self.plan['validation_tokens_per_source_cap']):break
                    row=json.loads(line);text,reason=clean_document(row['text']);cursors[split]=i+1
                    if reason:counts[reason]+=1
                    else:
                        norm=normalize(text);digest,lead=key(text),key(norm[:160]);fp=fingerprint(norm)
                        if not any(q in norm for q in self.blocked) and self.accept_doc(digest,lead,fp,split,counts):
                            writers[split].add(self.tokenizer.encode(text,add_special_tokens=False).ids+[2]);counts[split+'_documents']+=1
                            self.sample(samples,dict(text=text,split=split,source='wiki',group=lead),sum(counts[s+'_documents'] for s in ('train','validation')))
                    if (i+1)%4096==0:
                        self.guard();value={s:w.state() for s,w in writers.items()};value.update(cursors=cursors,counts=dict(counts),samples=samples,complete=False)
                        self.commit('wiki',value);self.db.execute('BEGIN');self.progress(stage='cleaning_wiki',documents=i+1,train_tokens=writers['train'].tokens)
                value={s:w.state() for s,w in writers.items()};value.update(cursors=cursors,counts=dict(counts),samples=samples,complete=False);self.commit('wiki',value)
        value.update(complete=True,source_revision='b04c8d1ceb2f5cd4588862100d08de323dccfbaa',replay_of_parent=True);self.commit('wiki',value)
        self.progress(stage='wiki_complete',train_tokens=value['train']['tokens'])
        return value

    def sample(self,samples,row,seen):
        # Deterministic reservoir; not simply the first documents in a source.
        if len(samples)<200:samples.append(row)
        else:
            slot=int(key(f"{self.plan['seed']}:{row['group']}:{seen}")[:16],16)%seen
            if slot<200:samples[slot]=row

    def accept_doc(self,digest,lead,fp,split,counts):
        if self.db.execute('SELECT 1 FROM docs WHERE hash=? OR lead=?',(digest,lead)).fetchone():counts['exact_or_same_lead']+=1;return False
        buckets=[(i<<16)|((fp>>(i*16))&65535) for i in range(4)]
        # No candidate cap: all fingerprints sharing a bucket must be compared.
        if any((fp^int(row[0],16)).bit_count()<=3 for row in self.db.execute('SELECT DISTINCT fp FROM near WHERE bucket IN (?,?,?,?)',buckets)):
            counts['near_duplicate']+=1;return False
        self.db.execute('INSERT INTO docs VALUES (?,?,?)',(digest,lead,split))
        self.db.executemany('INSERT INTO near VALUES (?,?,?)',[(b,format(fp,'016x'),digest) for b in buckets])
        return True

    def cached_download(self,source,row):
        path=self.data/'raw-cache'/source['repo'].replace('/','--')/row['rfilename']
        for root in (Path('D:/LubanLM-Data/general-2b-v1/raw'),):
            old=root/source['repo'].replace('/','--')/source['revision']/row['rfilename']
            if old.exists() and old.stat().st_size==row['size'] and sha256(old)==row['lfs']['sha256']:
                return old
        return self.mirror.download(f"{self.mirror.endpoint}/datasets/{source['repo']}/resolve/{source['revision']}/{row['rfilename']}",path,row['size'],row['lfs']['sha256'],self.guard)

    def files(self,source,files):
        executor=ThreadPoolExecutor(max_workers=self.plan['download_workers']);queued=deque();index=0
        # Reservations include the current file, partial files and pending work.
        def fill(current=0):
            nonlocal index
            resident=sum(p.stat().st_size for p in Path(self.plan.get('sft_raw_root',self.data/'sft-raw')).rglob('*') if p.is_file())
            cap=min(self.plan['download_prefetch_bytes'],self.plan['raw_cache_bytes_cap']-resident)
            def file_bytes(row):
                target=self.data/'raw-cache'/source['repo'].replace('/','--')/row['rfilename']
                return sum(p.stat().st_size for p in (target,target.with_name(target.name+'.part')) if p.exists())
            raw_resident=sum(p.stat().st_size for p in (self.data/'raw-cache').rglob('*') if p.is_file())
            booked=max(current,raw_resident)+sum(max(0,r['size']-file_bytes(r)) for r,_ in queued)
            while index<len(files) and len(queued)<self.plan['download_workers']:
                row=files[index]
                if row['size']>self.plan['raw_cache_bytes_cap']:raise ValueError('Raw file exceeds rolling cache cap')
                additional=max(0,row['size']-file_bytes(row))
                if booked+additional>cap:
                    if not queued and not current:raise RuntimeError('No cache capacity for next raw file')
                    break
                queued.append((row,executor.submit(self.cached_download,source,row)));index+=1;booked+=additional
        try:
            fill()
            while queued:
                row,future=queued.popleft()
                if not future.done():self.progress(stage='waiting_for_download',source=source['name'],file=row['rfilename'])
                while not future.done():self.guard();time.sleep(1)
                path=future.result();fill(row['size'])
                self.progress(stage='reading_file',source=source['name'],file=row['rfilename'])
                yield row,path
                fill()
        finally:
            for _,future in queued:future.cancel()
            executor.shutdown(wait=True,cancel_futures=True)

    def source(self,source,target=None):
        target=target or source['train_tokens_target'];old=self.previous(source['name']) or {}
        metadata=read_json(ROOT/source['metadata'])
        if metadata['sha']!=source['revision']:raise ValueError('Source revision mismatch')
        files=[r for r in metadata['siblings'] if r['rfilename'].startswith(source['prefix']) and r['rfilename'].endswith('.parquet')]
        random.Random(f"{self.plan['seed']}:{source['name']}").shuffle(files)
        done=set(old.get('done_files',[]));files=[r for r in files if r['rfilename'] not in done]
        cursors=dict(old.get('file_rows',{}))
        writers={s:Writer(self.data/f"processed/{source['name']}/{s}",self.plan['tokenizer_sha256'],old.get(s)) for s in ('train','validation')}
        counts=Counter(old.get('counts',{}));previews=old.get('samples',[])
        index_path=self.data/f"processed/{source['name']}/document-index.jsonl"
        with index_path.open('a+b') as index:
            if index_path.stat().st_size<old.get('index_bytes',0):raise ValueError('Committed document index was truncated')
            index.truncate(old.get('index_bytes',0));index.seek(0,2)
            if writers['train'].tokens>=target:return old
            stream=self.files(source,files)
            parquet=None
            try:
                for file,path in stream:
                    self.db.execute('BEGIN')
                    parquet=pq.ParquetFile(path);row_position=0
                    for batch in parquet.iter_batches(batch_size=256):
                        pending=[]
                        for row in batch.to_pylist():
                            row_position+=1
                            if row_position<=cursors.get(file['rfilename'],0):continue
                            counts['raw_documents']+=1
                            if row.get('score',0)<source['score_min']:counts['score']+=1;continue
                            if source.get('formats') and (row.get('data_format') not in source['formats'] or row.get('source') in source['exclude_seed_sources']):counts['style']+=1;continue
                            text,reason=clean_document(row['text'])
                            if reason:counts[reason]+=1;continue
                            if source.get('formats') and len(text)>6000:counts['synthetic_long']+=1;continue
                            norm=normalize(text)
                            if any(q in norm for q in self.blocked):counts['heldout_question']+=1;continue
                            digest,lead=key(text),key(norm[:160])
                            fp=fingerprint(norm);buckets=[(i<<16)|((fp>>(i*16))&65535) for i in range(4)]
                            split='validation' if int(lead[:8],16)%100<2 else 'train'
                            if writers[split].tokens>=(target if split=='train' else self.plan['validation_tokens_per_source_cap']):continue
                            pending.append((text,digest,lead,fp,buckets,split,row['score'],row_position))
                        encoded=self.tokenizer.encode_batch([r[0] for r in pending],add_special_tokens=False)
                        for item,encoding in zip(pending,encoded):
                            text,digest,lead,fp,buckets,split,score,source_row=item;ids=encoding.ids+[2]
                            if len(ids)<128:counts['too_short']+=1;continue
                            if writers[split].tokens>=(target if split=='train' else self.plan['validation_tokens_per_source_cap']):continue
                            # Pending rows can also duplicate another row in this batch.
                            if not self.accept_doc(digest,lead,fp,split,counts):continue
                            offset=writers[split].tokens
                            writers[split].add(ids);counts[split+'_documents']+=1
                            index.write((json.dumps(dict(hash=digest,group=lead,split=split,file=file['rfilename'],source_row=source_row,tokens=len(ids),token_offset=offset,score=score))+'\n').encode())
                            self.sample(previews,dict(text=text,score=score,split=split,group=lead,source=source['name']),sum(counts[s+'_documents'] for s in ('train','validation')))
                        self.guard()
                        if counts['raw_documents']%4096<256:self.progress(stage='encoding',source=source['name'],train_tokens=writers['train'].tokens,counts=dict(counts))
                        # If the quota is reached mid-batch, reread the batch on
                        # extension; committed documents are deduplicated.
                        if writers['train'].tokens>=target:
                            row_position-=len(batch);break
                    cursors[file['rfilename']]=max(cursors.get(file['rfilename'],0),row_position)
                    if row_position>=parquet.metadata.num_rows:done.add(file['rfilename'])
                    parquet.close();parquet=None
                    index.flush();os.fsync(index.fileno())
                    value={s:w.state() for s,w in writers.items()}
                    value.update(done_files=sorted(done),file_rows=cursors,counts=dict(counts),samples=previews,index_bytes=index.tell(),source=source)
                    self.commit(source['name'],value)
                    write_json(self.runs/f"reports/{source['name']}.json",value)
                    cache_root=(self.data/'raw-cache').resolve()
                    if path.resolve().is_relative_to(cache_root):path.unlink()
                    self.progress(stage='file_committed',source=source['name'],train_tokens=writers['train'].tokens,files=len(done))
                    if writers['train'].tokens>=target:break
            finally:
                if parquet is not None:parquet.close()
                stream.close()
                # Leave verified prefetched files and Range-resumable partials.
                # They are not token checkpoints and are reused on the next run.
        return self.previous(source['name']) or old

    def compact(self,results):
        # File transactions can leave many small durable shards. Compact one
        # source at a time; old shards remain until the new checkpoint commits.
        for name,result in results.items():
            updated=dict(result);obsolete=[]
            for split in ('train','validation'):
                manifest=result[split]
                if not manifest['shards']:continue
                if all('compacted' in Path(r['path']).parts for r in manifest['shards']):
                    # Recover a stop after checkpoint commit but before cleanup.
                    original_folder=self.data/f'processed/{name}/{split}'
                    if not original_folder.resolve().is_relative_to((self.data/'processed').resolve()):raise ValueError('Cleanup outside owned data')
                    for original in original_folder.glob('*.bin'):obsolete.append(original)
                    continue
                folder=self.data/f'processed/{name}/{split}/compacted';folder.mkdir(parents=True,exist_ok=True)
                shards=[];handle=None;digest=None;size=0
                def finish():
                    nonlocal handle,size
                    if handle is None:return
                    handle.flush();os.fsync(handle.fileno());handle.close()
                    final=Path(handle.name).with_suffix('');Path(handle.name).replace(final)
                    shards.append(dict(path=str(final),tokens=size//2,sha256=digest.hexdigest()))
                    handle=None;size=0
                try:
                    for row in manifest['shards']:
                        original=Path(row['path'])
                        if not original.resolve().is_relative_to((self.data/'processed').resolve()):raise ValueError('Compaction input outside owned processed data')
                        if sha256(original)!=row['sha256']:raise ValueError('Input shard changed before compaction')
                        with original.open('rb') as reader:
                            while True:
                                self.guard()
                                if handle is None:
                                    handle=(folder/f'{len(shards):05d}.bin.tmp').open('wb');digest=hashlib.sha256()
                                block=reader.read(min(1<<20,128000000-size))
                                if not block:break
                                handle.write(block);digest.update(block);size+=len(block)
                                if size==128000000:finish()
                        obsolete.append(original)
                    if size:finish()
                    elif handle is not None:
                        name_empty=handle.name;handle.close();Path(name_empty).unlink();handle=None
                finally:
                    if handle is not None:handle.close()
                if sum(r['tokens'] for r in shards)!=manifest['tokens']:raise ValueError('Compaction changed token count')
                updated[split]=dict(manifest,shards=shards)
                self.progress(stage='compacting_tokens',source=name,split=split,tokens=manifest['tokens'])
            self.commit(name,updated)
            write_json(self.runs/f'reports/{name}.json',updated)
            for original in obsolete:original.unlink()
            results[name]=updated
        return results

    def run(self):
        complete=self.runs/'reports/preparation_complete.json'
        if complete.exists():
            value=read_json(complete)
            if value.get('policy_identity')!=self.policy:raise ValueError('Prepared policy changed')
            return value
        wiki=self.wiki()
        results={s['name']:self.source(s) for s in self.plan['sources']}
        total=wiki['train']['tokens']+sum(r.get('train',{}).get('tokens',0) for r in results.values())
        if total<self.plan['additional_tokens']+1:
            high=self.plan['sources'][0]
            target=results[high['name']]['train']['tokens']+self.plan['additional_tokens']+1-total
            results[high['name']]=self.source(high,target)
        results=self.compact(results)
        pieces={'wiki':wiki,**results}
        for split in ('train','validation'):
            manifests=[r[split] for r in pieces.values()]
            value=dict(dtype='uint16',tokenizer_sha256=self.plan['tokenizer_sha256'],policy_identity=self.policy,shards=[s for r in manifests for s in r['shards']],tokens=sum(r['tokens'] for r in manifests),source_tokens={n:r[split]['tokens'] for n,r in pieces.items()},dedup='exact + normalized lead + 128-shingle SimHash <=3; exhaustive bucket candidates',packing='EOS between documents; cross-document attention permitted')
            write_json(self.data/f'processed/pretrain-{split}/manifest.json',value)
        train=read_json(self.data/'processed/pretrain-train/manifest.json')
        if train['tokens']<self.plan['additional_tokens']+1:raise RuntimeError('10B usable train corpus not achieved')
        if any(r['validation']['tokens']<100000 for r in pieces.values()):raise RuntimeError('Heldout source coverage too small')
        samples=[]
        for name,result in pieces.items():
            pool=result.get('samples',[]);random.Random(f"{self.plan['seed']}:{name}").shuffle(pool);samples.extend(pool[:50])
        write_json(self.runs/'reports/pretrain-content-samples.json',dict(policy_identity=self.policy,samples=samples))
        value=dict(created_utc=utc(),policy_identity=self.policy,plan_sha256=sha256(self.plan_path),train_tokens=train['tokens'],training_target_tokens=self.plan['additional_tokens'],source_tokens=train['source_tokens'],raw_cache_cap=self.plan['raw_cache_bytes_cap'],quality_status='structural_passed_content_sample_pending',parent_overlap='Wiki replay known; other parent overlap not fully measured')
        write_json(complete,value);self.progress(stage='complete',**value)
        return value

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',default='configs/50m-general-pretrain-10b.json')
    args=parser.parse_args()
    from scripts.worker_health import WorkerHealth
    with WorkerHealth(args.plan,'preparing_10b_cpu'):
        preparation=Preparation(args.plan)
        try:preparation.run()
        finally:preparation.db.close()

if __name__=='__main__':main()
