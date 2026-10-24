"""Finite range probe, using the same physical route and candidate rules."""
import argparse,json
from collections import Counter
from pathlib import Path
from scripts.common import ROOT,read_json,write_json
from scripts.direct_mirror import PhysicalMirror
from scripts.prepare_sft_100k import candidates
from scripts.corpus_quality import rejected_groups
from eval.chat_suite import normalize,excluded_questions
from tokenizers import Tokenizer

def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--file',required=True);p.add_argument('--offset',type=int,default=0);a=p.parse_args()
    plan=read_json(ROOT/'configs/50m-general-pretrain-10b-teaching.json')
    mirror=PhysicalMirror(None,interface='WLAN');meta=mirror.metadata(a.repo)
    write_json(ROOT/'.cache/general-10b'/ (a.repo.replace('/','--')+'.json'),meta)
    blob=next(r for r in meta['siblings'] if r['rfilename']==a.file)
    source=next((s for s in plan['sft']['raw_sources'] if s['repo']==a.repo),dict(repo=a.repo,revision=meta['sha'],allowed_tasks=['daily'],daily_only=True))
    c,r=mirror.open(f'{mirror.endpoint}/datasets/{a.repo}/resolve/{meta["sha"]}/{a.file}',{'Range':f'bytes={a.offset}-{a.offset+8388607}'})
    try:
        if r.status not in (200,206):raise ValueError(r.status)
        if a.offset and (r.status!=206 or not r.getheader('Content-Range','').startswith(f'bytes {a.offset}-')):raise ValueError('Range ignored')
        data=r.read(8<<20).decode('utf-8',errors='ignore').splitlines()[:-1]
        if a.offset:data=data[1:]
    finally:c.close()
    tk=Tokenizer.from_file(str(Path(plan['legacy_data'])/'tokenizers/vocab-32000/tokenizer.json'))
    blocked={normalize(q) for q in excluded_questions()};bad=rejected_groups();counts=Counter();samples=[]
    for i,line in enumerate(data[:2000],1):
        row=json.loads(line);counts['raw_rows']+=1
        best=None
        for v in candidates(row,source,a.file,a.repo+'/'+meta['sha']+'/'+a.file,i,blob,plan,tk,blocked,bad,counts):
            if best is None or len(v['input_ids'])>len(best['input_ids']):best=v
        if best:
            counts['accepted_'+best['task']]+=1
            if len(samples)<40:samples.append(dict(task=best['task'],messages=best['messages']))
    report=dict(repo=a.repo,revision=meta['sha'],file=a.file,byte_offset=a.offset,blob=blob,counts=dict(counts),samples=samples,physical_interface=mirror.adapter)
    write_json(ROOT/'docs/run-results/teaching-supplement-20261007'/(a.repo.replace('/','--')+f'-range-pilot-{a.offset}.json'),report)
    print(json.dumps(dict(repo=a.repo,offset=a.offset,counts=dict(counts)),ensure_ascii=False))

if __name__=='__main__':main()
