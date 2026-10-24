"""Read-only bounded ZIP range research; no pipeline policy changes."""
import json,struct,zlib
from collections import Counter
from pathlib import Path
from scripts.common import ROOT,read_json,write_json
from scripts.direct_mirror import PhysicalMirror
from scripts.prepare_sft_100k import candidates
from scripts.corpus_quality import rejected_groups
from tokenizers import Tokenizer

def strip_tag(text,role,end):
    if not isinstance(text,str):raise ValueError('invalid tag content')
    text=text.strip();start='<|'+role+'|>:'
    if not text.startswith(start) or not text.endswith(end):raise ValueError('unexpected MOSS framing')
    return text[len(start):-len(end)].strip()

def adapt(row):
    ms=[]
    for n in range(1,row['num_turns']+1):
        turn=row['chat']['turn_'+str(n)]
        if any(strip_tag(turn[k],role,end)!='None' for k,role,end in [('Commands','Commands','<eoc>'),('Tool Responses','Results','<eor>')]):return {}
        ms.extend([dict(role='user',content=strip_tag(turn['Human'],'Human','<eoh>')),dict(role='assistant',content=strip_tag(turn['MOSS'],'MOSS','<eom>'))])
    return dict(conversations=ms)

def main():
    repo='OpenMOSS-Team/moss-003-sft-data';filename='moss-003-sft-no-tools.jsonl.zip'
    plan=read_json(ROOT/'configs/50m-general-pretrain-10b-teaching.json')
    mirror=PhysicalMirror(None,interface='WLAN');meta=mirror.metadata(repo)
    write_json(ROOT/'.cache/general-10b/OpenMOSS-Team--moss-003-sft-data.json',meta)
    blob=next(v for v in meta['siblings'] if v['rfilename']==filename)
    c,r=mirror.open(f'{mirror.endpoint}/datasets/{repo}/resolve/{meta["sha"]}/{filename}',{'Range':'bytes=0-8388607'})
    try:
        if r.status not in (200,206):raise ValueError(r.status)
        data=r.read(8<<20)
    finally:c.close()
    if data[:4]!=b'PK\x03\x04':raise ValueError('Unexpected ZIP header')
    header=struct.unpack('<IHHHHHIIIHH',data[:30]);method=header[3];name_len,extra_len=header[-2:]
    name=data[30:30+name_len].decode('utf-8');offset=30+name_len+extra_len
    if method!=8 or not name.endswith('.jsonl'):raise ValueError('Expected deflated jsonl member')
    content=zlib.decompressobj(-15).decompress(data[offset:],32<<20).decode('utf-8',errors='ignore').splitlines()[:-1]
    tk=Tokenizer.from_file(str(Path(plan['legacy_data'])/'tokenizers/vocab-32000/tokenizer.json'))
    source=dict(repo=repo,revision=meta['sha'],allowed_tasks=['daily'],daily_only=True,first_daily_only=True,exclude_context_regex='MOSS|复旦|Fudan')
    counts=Counter();samples=[];bad=rejected_groups()
    for i,line in enumerate(content[:2000],1):
        row=json.loads(line);counts['raw_rows']+=1
        if i==1:print('First raw:',str(row)[:3000],flush=True)
        transformed=adapt(row);best=None
        for v in candidates(transformed,source,filename,repo+'/'+meta['sha']+'/'+filename,i,blob,plan,tk,set(),bad,counts):
            if best is None or len(v['input_ids'])>len(best['input_ids']):best=v
        if best:
            counts['accepted_daily']+=1
            if len(samples)<40:samples.append(best['messages'])
    write_json(ROOT/'docs/run-results/teaching-supplement-20261007/moss-range-pilot.json',dict(repo=repo,revision=meta['sha'],blob=blob,zip_member=name,counts=dict(counts),samples=samples,physical_interface=mirror.adapter))
    print(json.dumps(dict(counts),ensure_ascii=False))

if __name__=='__main__':main()
