"""Pinned physical-interface downloads and bounded SFT supplementation probes."""
import argparse
from collections import Counter
from pathlib import Path
import shutil
import json
from scripts.common import ROOT,read_json,write_json,sha256,utc
from scripts.resource_guard import check_resources

def cache_bytes(plan):
    roots={Path(plan['sft_raw_root']),Path(plan['data_root'])/'raw-cache'}
    return sum(p.stat().st_size for root in roots if root.exists() for p in root.rglob('*') if p.is_file())

def ensure_file(plan,source,filename,blob):
    path=Path(plan['sft_raw_root'])/source['repo'].replace('/','--')/filename
    if path.exists():
        if path.stat().st_size!=blob['size'] or sha256(path)!=blob['lfs']['sha256']:raise ValueError('SFT raw integrity failed')
        return path
    if not source.get('supplement'):raise FileNotFoundError(path)
    from scripts.direct_mirror import PhysicalMirror
    partial=path.with_suffix(path.suffix+'.part')
    remaining=max(0,blob['size']-(partial.stat().st_size if partial.exists() else 0))
    cap=plan.get('raw_cache_bytes_cap',5<<30)
    if cache_bytes(plan)+remaining>cap:raise RuntimeError('RESOURCE_WAIT: raw cache would exceed cap')
    if shutil.disk_usage(plan['data_root']).free-remaining<plan['disk_free_bytes_floor']:raise RuntimeError('RESOURCE_WAIT: download would consume disk reserve')
    def guard():
        check_resources(plan,'sft')
        if cache_bytes(plan)>cap:raise RuntimeError('RESOURCE_WAIT: raw cache cap reached')
    mirror=PhysicalMirror(Path(plan['runs_root'])/'reports/network.jsonl',interface='WLAN')
    mirror.download(f"{mirror.endpoint}/datasets/{source['repo']}/resolve/{source['revision']}/{filename}",path,blob['size'],blob['lfs']['sha256'],guard)
    return path

def quota_ready(db,plan):
    from scripts.sft_disk import assign_splits
    from scripts.balanced_sft import soft_allocations
    from scripts.policy_versions import digest
    size=db.execute('SELECT COUNT(*) FROM candidates').fetchone()[0]
    # This preparation pool is append-only. A policy change invalidates the
    # cache; a newly accepted candidate changes its extent. Final export still
    # performs a fresh family/split calculation unconditionally.
    binding=dict(size=size,policy=db.execute("SELECT value FROM meta WHERE name='policy'").fetchone(),settings=digest(dict(seed=plan['seed'],sft=plan['sft'])))
    saved=db.execute("SELECT value FROM meta WHERE name='quota_cache'").fetchone()
    if saved:
        cached=json.loads(saved[0])
        if cached['binding']==json.loads(json.dumps(binding)):return cached['ready'],cached['gaps']
    assign_splits(db,plan)
    counts={(t,s):n for t,s,n in db.execute("SELECT task,source,COUNT(*) FROM candidates WHERE split='train' GROUP BY task,source")}
    _,gaps=soft_allocations(counts,plan['sft']['source_fraction_caps'],plan['sft']['train_minimum'],plan['sft']['daily_minimum'],plan['sft']['concept_maximum'])
    for split in ('validation','test'):
        gaps[split+'_gap']=plan['sft'][split+'_examples']-db.execute('SELECT COUNT(*) FROM candidates WHERE split=?',(split,)).fetchone()[0]
    ready=not any(n>0 for n in gaps.values())
    db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)',('quota_cache',json.dumps(dict(binding=binding,ready=ready,gaps=gaps))));db.commit()
    return ready,gaps

def pilot(plan_path,repo=None):
    from tokenizers import Tokenizer
    from scripts.prepare_sft_100k import rows,candidates
    from scripts.corpus_quality import rejected_groups
    from eval.chat_suite import normalize,excluded_questions
    from scripts.worker_health import WorkerHealth
    plan=read_json(plan_path);tokenizer=Tokenizer.from_file(str(Path(plan['legacy_data'])/'tokenizers/vocab-32000/tokenizer.json'))
    blocked={normalize(q) for q in excluded_questions()}
    frozen=ROOT/'docs/run-results/general-10b-20261005/frozen-test.json'
    if frozen.exists():blocked.update(normalize(r['question']) for r in read_json(frozen)['questions'])
    reports=[];bad_groups=rejected_groups()
    with WorkerHealth(plan_path,'preparing_sft_cpu'):
        for source in plan['sft']['raw_sources']:
            if not source.get('supplement') or (repo and source['repo']!=repo):continue
            filename=source['files'][0];metadata=read_json(ROOT/source['metadata'])
            if metadata['sha']!=source['revision']:raise ValueError('SFT source pin changed')
            blob=next(r for r in metadata['siblings'] if r['rfilename']==filename)
            path=ensure_file(plan,source,filename,blob);counts=Counter();samples=[]
            key=source['repo']+'/'+source['revision']+'/'+filename
            for i,row in enumerate(rows(path),1):
                if i>2000:break
                counts['raw_rows']+=1;best=None
                for item in candidates(row,source,filename,key,i,blob,plan,tokenizer,blocked,bad_groups,counts):
                    if best is None or len(item['input_ids'])>len(best['input_ids']):best=item
                if best:
                    counts['accepted_'+best['task']]+=1
                    if len(samples)<25:samples.append(dict(task=best['task'],messages=best['messages']))
            report=dict(repo=source['repo'],revision=source['revision'],file=filename,counts=dict(counts),samples=samples,global_dedup_not_yet_applied=True)
            reports.append(report)
            name='sft_supplement_pilot.json' if not repo else 'sft_supplement_pilot-'+repo.replace('/','--')+'.json'
            write_json(Path(plan['runs_root'])/'reports'/name,dict(created_utc=utc(),sources=reports))
            print(source['repo'],dict(counts),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',default='configs/50m-general-pretrain-10b-teaching.json');p.add_argument('--repo');a=p.parse_args();pilot(a.plan,a.repo)
