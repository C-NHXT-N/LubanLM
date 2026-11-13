"""One-time, journalled removal of inspected documents using exact token indexes."""
import argparse
import copy
from collections import defaultdict
import json
from pathlib import Path
import shutil
import numpy as np
from scripts.common import ROOT,read_json,write_json,sha256,utc
from scripts.prepare_general_10b import Preparation
from scripts.worker_health import WorkerHealth

def without_intervals(array,start,intervals):
    end=start+len(array);position=0;pieces=[]
    for a,b in intervals:
        lo=max(a,start);hi=min(b,end)
        if lo>=hi:continue
        pieces.append(array[position:lo-start]);position=hi-start
    pieces.append(array[position:]);return pieces

def read_tokens(manifest,offset,number):
    result=[];start=0
    for shard in manifest['shards']:
        end=start+shard['tokens'];lo=max(start,offset);hi=min(end,offset+number)
        if lo<hi:
            with Path(shard['path']).open('rb') as f:f.seek((lo-start)*2);result.extend(np.frombuffer(f.read((hi-lo)*2),dtype=np.uint16).tolist())
        start=end
    if len(result)!=number:raise ValueError('Indexed record extent mismatch')
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',default='configs/50m-general-pretrain-10b-teaching.json');p.add_argument('--notes',required=True);a=p.parse_args()
    plan=read_json(a.plan);data=Path(plan['data_root']);runs=Path(plan['runs_root']);backup=runs/'recovery/content-repair-20261007'
    backup.mkdir(parents=True,exist_ok=True);journal=backup/'journal.json'
    if journal.exists():
        if read_json(journal).get('stage')=='completed':print('Content repair already completed');return
        raise RuntimeError('Incomplete repair journal: restore saved indexes/checkpoints before retry')
    original=read_json(runs/'reports/pretrain-content-samples.json');samples=original['samples'];notes=read_json(a.notes)
    rejected={samples[int(i)]['group']:reason for i,reason in notes['reject'].items()}
    write_json(backup/'original-samples.json',original);write_json(backup/'inspection-notes.json',notes)
    states={};by_source=defaultdict(dict);removed=[]
    for sample in samples:
        if sample['group'] in rejected:by_source[sample['source']][sample['group']]=sample
    for source in by_source:
        index=data/f'processed/{source}/document-index.jsonl';found={}
        with index.open(encoding='utf-8') as f:
            for line in f:
                row=json.loads(line)
                if row['group'] in by_source[source]:found[row['group']]=row
        if set(found)!=set(by_source[source]):raise ValueError('Rejected record missing from document index')
        removed.extend(dict(row,source=source,reason=rejected[group]) for group,row in found.items())
    prep=Preparation(a.plan)
    try:
        for source in by_source:
            states[source]=prep.previous(source);shutil.copy2(data/f'processed/{source}/document-index.jsonl',backup/(source+'-index.jsonl'))
        for row in removed:
            sample=by_source[row['source']][row['group']]
            expected=prep.tokenizer.encode(sample['text'],add_special_tokens=False).ids+[2]
            if read_tokens(states[row['source']][row['split']],row['token_offset'],row['tokens'])!=expected:
                raise ValueError('Indexed rejected text does not match stored tokens')
        write_json(backup/'checkpoints.json',states)
        write_json(journal,dict(stage='prepared',created_utc=utc(),removed=removed,restore='Restore backed-up indexes and SQLite checkpoints; original shards retained'))
        with WorkerHealth(a.plan,'preparing_10b_cpu'):
            for source,old in states.items():
                state=copy.deepcopy(old);intervals=defaultdict(list)
                for row in removed:
                    if row['source']==source:intervals[row['split']].append((row['token_offset'],row['token_offset']+row['tokens']))
                for split,ranges in intervals.items():
                    ranges.sort();manifest=state[split];new=[];start=0
                    for shard in manifest['shards']:
                        prep.guard();end=start+shard['tokens']
                        if any(lo<end and hi>start for lo,hi in ranges):
                            array=np.memmap(shard['path'],dtype=np.uint16,mode='r');parts=without_intervals(array,start,ranges)
                            folder=data/f'processed/{source}/{split}/review-repaired';folder.mkdir(parents=True,exist_ok=True)
                            dest=folder/Path(shard['path']).name;temp=dest.with_suffix('.bin.tmp')
                            with temp.open('wb') as f:
                                for part in parts:part.tofile(f)
                            del parts,array;temp.replace(dest)
                            if dest.stat().st_size:new.append(dict(path=str(dest),tokens=dest.stat().st_size//2,sha256=sha256(dest)))
                        else:new.append(shard)
                        start=end
                    manifest['shards']=new;manifest['tokens']=sum(r['tokens'] for r in new);manifest['documents']-=sum(1 for r in removed if r['source']==source and r['split']==split)
                    state['counts'][split+'_documents']=manifest['documents']
                    expected=old[split]['tokens']-sum(hi-lo for lo,hi in ranges)
                    if manifest['tokens']!=expected:raise ValueError('Removal token accounting mismatch')
                index=data/f'processed/{source}/document-index.jsonl';temp=index.with_suffix('.repaired.tmp')
                with index.open(encoding='utf-8') as f,temp.open('w',encoding='utf-8') as out:
                    for line in f:
                        row=json.loads(line)
                        if row['group'] in rejected:continue
                        row['token_offset']-=sum(hi-lo for lo,hi in intervals[row['split']] if hi<=row['token_offset'])
                        out.write(json.dumps(row)+'\n')
                temp.replace(index);state['index_bytes']=index.stat().st_size
                state['samples']=[r for r in state.get('samples',[]) if r['group'] not in rejected]
                prep.commit(source,state);write_json(runs/f'reports/{source}.json',state)
                write_json(journal,dict(stage='removed_'+source,removed=removed,created_utc=utc()))
            # Retain rejected hashes in global dedup so extension cannot reaccept them.
            pieces={'wiki':prep.previous('wiki'),**{s['name']:prep.previous(s['name']) for s in plan['sources']}}
            old_ends={name:r['train']['tokens'] for name,r in pieces.items()}
            for source in plan['sources']:
                if source['name'] in states:
                    prep.source(source,states[source['name']]['train']['tokens']+32000)
                    pieces[source['name']]=prep.previous(source['name'])
            for split in ('train','validation'):
                path=data/f'processed/pretrain-{split}/manifest.json';manifest=read_json(path)
                manifest.update(shards=[s for r in pieces.values() for s in r[split]['shards']],tokens=sum(r[split]['tokens'] for r in pieces.values()),source_tokens={n:r[split]['tokens'] for n,r in pieces.items()})
                write_json(path,manifest)
            # Use fresh real documents for replacement inspection, preserving 200 slots.
            replacements=[];by_replacement={}
            for source,groups in by_source.items():
                items=[];index=data/f'processed/{source}/document-index.jsonl'
                with index.open(encoding='utf-8') as f:
                    for line in f:
                        row=json.loads(line)
                        if row['split']=='train' and row['token_offset']>=old_ends[source]:
                            ids=read_tokens(pieces[source]['train'],row['token_offset'],row['tokens'])
                            if ids[-1]!=2:raise ValueError('Replacement lacks EOS')
                            items.append(dict(text=prep.tokenizer.decode(ids[:-1]),group=row['group'],source=source,split='train',score=row['score']))
                            if len(items)>=len(groups):break
                if len(items)<len(groups):raise RuntimeError('Insufficient fresh replacement review records')
                replacements.extend(items);by_replacement[source]=iter(items)
            new_samples=[next(by_replacement[r['source']]) if r['group'] in rejected else r for r in samples]
            write_json(runs/'reports/pretrain-content-samples.json',dict(policy_identity=prep.policy,samples=new_samples,replacement_provenance=str(journal)))
            complete=read_json(runs/'reports/preparation_complete.json');train=read_json(data/'processed/pretrain-train/manifest.json')
            complete.update(train_tokens=train['tokens'],source_tokens=train['source_tokens'],content_repair=str(journal));write_json(runs/'reports/preparation_complete.json',complete)
            if train['tokens']<plan['additional_tokens']+1:raise ValueError('10B target no longer met')
            write_json(journal,dict(stage='completed',created_utc=utc(),removed=removed,replacements=replacements,train_tokens=train['tokens']))
            print(json.dumps(dict(removed=len(removed),replacement_samples=len(replacements),train_tokens=train['tokens'])),flush=True)
    finally:prep.db.close()

if __name__=='__main__':main()
