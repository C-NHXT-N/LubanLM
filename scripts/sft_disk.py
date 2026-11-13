"""Bounded-memory, transactional SFT selection from pinned public source files."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sqlite3
import time
import psutil
from scripts.common import ROOT,read_json,write_json,sha256,utc
from scripts.policy_versions import sft_identity,digest
from scripts.resource_guard import check_resources

def connect(path):
    db=sqlite3.connect(path,timeout=60)
    db.executescript('''PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL;
      PRAGMA cache_size=-65536; PRAGMA temp_store=FILE;
      CREATE TABLE IF NOT EXISTS meta(name TEXT PRIMARY KEY,value TEXT);
      CREATE TABLE IF NOT EXISTS cursors(file TEXT PRIMARY KEY,position INTEGER,complete INTEGER,counts TEXT);
      CREATE TABLE IF NOT EXISTS candidates(id TEXT PRIMARY KEY,context TEXT UNIQUE,exact TEXT UNIQUE,
        family TEXT,task TEXT,source TEXT,rank TEXT,payload TEXT,split TEXT,chosen INTEGER DEFAULT 0);
      CREATE INDEX IF NOT EXISTS selection ON candidates(split,task,source,rank);
      CREATE INDEX IF NOT EXISTS family_lookup ON candidates(family);
      CREATE TABLE IF NOT EXISTS near(bucket INTEGER,fp TEXT);
      CREATE INDEX IF NOT EXISTS near_covering ON near(bucket,fp);
      DROP INDEX IF EXISTS near_bucket;
      CREATE TABLE IF NOT EXISTS lineage_info(id TEXT PRIMARY KEY,turns INTEGER,chars INTEGER);
      CREATE TABLE IF NOT EXISTS prefixes(key TEXT,id TEXT,PRIMARY KEY(key,id));
      CREATE INDEX IF NOT EXISTS prefix_owner ON prefixes(id);
      CREATE TABLE IF NOT EXISTS legacy(id INTEGER PRIMARY KEY,payload TEXT);
      DROP INDEX IF EXISTS family_idx;''')
    return db

def checkpoint_records(path):
    """Read the selected array without loading the 1 GB legacy JSON object."""
    decoder=json.JSONDecoder()
    with Path(path).open(encoding='utf-8') as stream:
        prefix=''
        while '"selected":' not in prefix:
            part=stream.read(1<<20)
            if not part:raise ValueError('Missing legacy selected array')
            prefix+=part
            if len(prefix)>8<<20:raise ValueError('Legacy header too large')
        buffer=prefix.split('"selected":',1)[1].lstrip()
        if not buffer.startswith('['):raise ValueError('Legacy array malformed')
        buffer=buffer[1:];need=True
        while True:
            buffer=buffer.lstrip()
            if not buffer:
                buffer=stream.read(1<<20)
                if not buffer:raise ValueError('Truncated legacy checkpoint')
                continue
            if buffer[0]==']':return
            if not need:
                if buffer[0]!=',':raise ValueError('Legacy separator missing')
                buffer=buffer[1:];need=True;continue
            while True:
                try:record,end=decoder.raw_decode(buffer);break
                except json.JSONDecodeError:
                    more=stream.read(1<<20)
                    if not more:raise
                    buffer+=more
                    if len(buffer)>16<<20:raise ValueError('Oversized legacy record')
            yield record;buffer=buffer[end:];need=False

def migrate_legacy(db,path,report):
    if not path.exists():return
    if db.execute("SELECT 1 FROM meta WHERE name='legacy_complete'").fetchone():return
    already=db.execute('SELECT COUNT(*) FROM legacy').fetchone()[0]
    count=0
    for count,row in enumerate(checkpoint_records(path),1):
        if count<=already:continue
        db.execute('INSERT OR IGNORE INTO legacy VALUES (?,?)',(count,json.dumps(row,ensure_ascii=False)))
        if count%2048==0:db.commit()
    db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)',('legacy_complete',str(count)));db.commit()
    write_json(report,dict(created_utc=utc(),legacy_path=str(path),imported=count,
        reuse='Audit archive; changed conversation identities require rescanning all affected raw records. No download.',
        rss_bytes=psutil.Process().memory_info().rss))

def insert_candidate(db,item,seed):
    from eval.chat_suite import normalize
    from scripts.prepare_general_10b import fingerprint
    # The whole context is the identity; a shared generic greeting is not a family.
    prompts=[normalize(m['content']) for m in item['messages'] if m['role']=='user']
    family=digest([item['system'],*prompts])
    norm='\n'.join(prompts)
    exact=digest([item['system'],*[(m['role'],normalize(m['content'])) for m in item['messages']]])
    if db.execute('SELECT 1 FROM candidates WHERE context=? OR exact=? OR family=?',(item['question_sha256'],exact,family)).fetchone():return 'exact_or_context_duplicate'
    if len(norm)>=40:
        fp=fingerprint(norm);buckets=[(i<<16)|((fp>>(i*16))&65535) for i in range(4)]
        if any((fp^int(r[0],16)).bit_count()<=3 for r in db.execute('SELECT DISTINCT fp FROM near WHERE bucket IN (?,?,?,?)',buckets)):return 'near_duplicate_conversation'
        db.executemany('INSERT INTO near VALUES (?,?)',[(b,format(fp,'016x')) for b in buckets])
    item['family']=family
    db.execute('INSERT INTO candidates VALUES (?,?,?,?,?,?,?,?,NULL,0)',
        (item['group'],item['question_sha256'],exact,family,item['task'],item['source'],digest([seed,item['group']]),json.dumps(item,ensure_ascii=False)))
    pairs=[]
    for message in item['messages']:
        pairs.append((message['role'],normalize(message['content'])))
        if message['role']=='assistant':
            generic=(len(pairs)==2 and pairs[0][1] in ('你好','您好','嗨','哈喽','早上好','晚上好','早安','午安','晚安','hello','hi') and len(pairs[1][1])<=80 and pairs[1][1].startswith(('你好','您好','嗨','哈喽','hello','hi')))
            if not generic:db.execute('INSERT OR IGNORE INTO prefixes VALUES (?,?)',(digest([item['system'],*pairs]),item['group']))
    db.execute('INSERT INTO lineage_info VALUES (?,?,?)',(item['group'],len(pairs)//2,sum(len(text) for _,text in pairs)))
    return None

class DataShortfall(RuntimeError):pass

def collapse_lineage(db):
    """Only a COMPLETE candidate matching another's prefix links their families.
    Two longer conversations sharing a generic first turn remain independent.
    """
    db.executescript('DROP TABLE IF EXISTS lineage_edges; CREATE TABLE lineage_edges(a TEXT,b TEXT,PRIMARY KEY(a,b)); CREATE INDEX lineage_reverse ON lineage_edges(b,a);')
    db.execute('INSERT OR IGNORE INTO lineage_edges SELECT p.id,c.id FROM prefixes p JOIN candidates c ON c.exact=p.key WHERE p.id<>c.id')
    db.execute('INSERT OR IGNORE INTO lineage_edges SELECT b,a FROM lineage_edges')
    while True:
        changed=db.execute('UPDATE candidates AS c SET family=(SELECT MIN(n.family) FROM lineage_edges e JOIN candidates n ON n.id=e.b WHERE e.a=c.id) WHERE EXISTS(SELECT 1 FROM lineage_edges e JOIN candidates n ON n.id=e.b WHERE e.a=c.id AND n.family<c.family)').rowcount
        if not changed:break
    db.executescript('DROP TABLE IF EXISTS representatives; CREATE TABLE representatives(id TEXT PRIMARY KEY);')
    db.execute("INSERT INTO representatives SELECT id FROM (SELECT c.id,ROW_NUMBER() OVER(PARTITION BY c.family ORDER BY COALESCE(i.turns,0) DESC,COALESCE(i.chars,0) DESC,c.rank,c.id) AS n FROM candidates c LEFT JOIN lineage_info i ON i.id=c.id) WHERE n=1")
    db.commit()

def assign_splits(db,plan):
    collapse_lineage(db)
    db.execute("UPDATE candidates SET split='excluded',chosen=0")
    db.execute("UPDATE candidates SET split='train' WHERE id IN (SELECT id FROM representatives)")
    sizes={'test':plan['sft']['test_examples'],'validation':plan['sft']['validation_examples']}
    counts=dict(db.execute("SELECT task,COUNT(*) FROM candidates WHERE split='train' GROUP BY task"))
    coverage={}
    for task,n in counts.items():
        coverage[task]='independent val/test' if n>=20 else 'train only; insufficient independent examples'
        if n>=20:
            for split in sizes:
                db.execute('UPDATE candidates SET split=? WHERE id IN (SELECT id FROM candidates WHERE split=\'train\' AND task=? ORDER BY rank LIMIT 5)',(split,task))
    for split,target in sizes.items():
        actual=db.execute('SELECT COUNT(*) FROM candidates WHERE split=?',(split,)).fetchone()[0]
        # Sparse categories stay in train; do not invent meaningless tiny test sets.
        eligible=[task for task,n in counts.items() if n>=20]
        placeholders=','.join('?' for _ in eligible)
        if eligible:
            db.execute(f"UPDATE candidates SET split=? WHERE id IN (SELECT id FROM candidates WHERE split='train' AND task IN ({placeholders}) ORDER BY rank LIMIT ?)",(split,*eligible,max(0,target-actual)))
    db.commit();return coverage

def finalize(db,plan,policy,out,runs,count):
    from scripts.balanced_sft import soft_allocations
    coverage=assign_splits(db,plan)
    counts={(t,s):n for t,s,n in db.execute("SELECT task,source,COUNT(*) FROM candidates WHERE split='train' GROUP BY task,source")}
    allocation,gaps=soft_allocations(counts,plan['sft']['source_fraction_caps'],plan['sft']['train_minimum'],plan['sft']['daily_minimum'],plan['sft']['concept_maximum'])
    for (task,source),number in allocation.items():
        db.execute("UPDATE candidates SET chosen=1 WHERE id IN (SELECT id FROM candidates WHERE split='train' AND task=? AND source=? ORDER BY rank LIMIT ?)",(task,source,number))
    db.execute("UPDATE candidates SET chosen=1 WHERE split IN ('validation','test')");db.commit()
    db.execute('CREATE INDEX IF NOT EXISTS output_order ON candidates(chosen,split,rank)');db.commit()
    report=dict(created_utc=utc(),policy_identity=policy,plan_sha256=digest(plan),template_sha256=sha256(ROOT/'sft/template.py'),recipe_sha256=sha256(ROOT/'scripts/prepare_sft_100k.py'),counts=dict(count),examples={},distinct_conversations={},categories={},sources={},quality_status='structural_filters_passed_content_sample_pending',split_group_disjoint=True,tokenizer_sha256=plan['tokenizer_sha256'],category_evaluation_coverage=coverage)
    for split in ('train','validation','test'):
        report['examples'][split]=db.execute('SELECT COUNT(*) FROM candidates WHERE chosen=1 AND split=?',(split,)).fetchone()[0]
        report['distinct_conversations'][split]=report['examples'][split]
        report['categories'][split]=dict(db.execute('SELECT task,COUNT(*) FROM candidates WHERE chosen=1 AND split=? GROUP BY task',(split,)))
    report['sources']=dict(db.execute("SELECT source,COUNT(*) FROM candidates WHERE chosen=1 AND split='train' GROUP BY source"))
    report['available_candidates']=dict(db.execute('SELECT task,COUNT(*) FROM candidates GROUP BY task'))
    report['rss_bytes']=psutil.Process().memory_info().rss
    if db.execute('SELECT 1 FROM candidates WHERE chosen=1 GROUP BY family HAVING COUNT(DISTINCT split)>1 LIMIT 1').fetchone():raise RuntimeError('Conversation family leakage')
    if db.execute('SELECT 1 FROM prefixes p JOIN candidates a ON a.id=p.id JOIN candidates b ON b.exact=p.key WHERE a.chosen=1 AND b.chosen=1 AND a.id<>b.id AND a.split<>b.split LIMIT 1').fetchone():raise RuntimeError('Conversation prefix leakage')
    for source,cap in plan['sft']['source_fraction_caps'].items():
        if report['sources'].get(source,0)>int(cap*plan['sft']['train_minimum']):raise RuntimeError('Source cap violated')
    if report['categories']['train'].get('concept',0)>plan['sft']['concept_maximum']:raise RuntimeError('Concept maximum violated')
    gaps.update({split+'_gap':plan['sft'][split+'_examples']-report['examples'][split] for split in ('validation','test')})
    write_json(runs/'reports/sft_filter_report.json',report)
    if any(n>0 for n in gaps.values()):
        write_json(runs/'reports/sft_shortfall.json',dict(created_utc=utc(),policy_identity=policy,gaps=gaps,report=report,action='Supplement verified public data; do not generate, relabel or relax hard totals/source caps'))
        raise DataShortfall('DATA_SHORTFALL: '+json.dumps(gaps))
    # Stream complete outputs atomically. No 100k list of token arrays in memory.
    for split in ('train','validation','test'):
        final=out/f'{split}-tokens.jsonl';temp=final.with_suffix('.jsonl.tmp')
        with temp.open('w',encoding='utf-8') as stream:
            for (payload,) in db.execute('SELECT payload FROM candidates WHERE chosen=1 AND split=? ORDER BY rank',(split,)):stream.write(payload+'\n')
            stream.flush();os.fsync(stream.fileno())
        temp.replace(final);report[split+'_sha256']=sha256(final)
    samples=[];seen=set()
    def sample_where(where,args,limit,split):
        if limit<=0:return
        for payload,identity in db.execute('SELECT payload,id FROM candidates WHERE chosen=1 AND '+where+' ORDER BY rank LIMIT ?',(*args,limit+len(seen))):
            if identity in seen:continue
            samples.append(dict(json.loads(payload),split=split));seen.add(identity);limit-=1
            if limit<=0:break
    for split in ('validation','test'):sample_where('split=?',(split,),90,split)
    for task in report['categories']['train']:sample_where("split='train' AND task=?",(task,),50,'train')
    for source in report['sources']:sample_where("split='train' AND source=?",(source,),10,'train')
    sample_where("split='train'",(),600-len(samples),'train')
    if len(samples)!=600:raise RuntimeError('Content sample count mismatch')
    write_json(runs/'reports/sft_semantic_samples.json',samples)
    write_json(runs/'reports/sft_filter_report.json',report);write_json(out/'manifest.json',report)
    shortfall=runs/'reports/sft_shortfall.json'
    if shortfall.exists():shortfall.unlink()

def run(plan_path):
    from tokenizers import Tokenizer
    from eval.chat_suite import normalize,excluded_questions
    from scripts.prepare_sft_100k import rows,candidates
    from scripts.corpus_quality import rejected_groups
    plan=read_json(plan_path);policy=sft_identity(plan_path)
    data,runs=Path(plan['data_root']),Path(plan['runs_root'])
    out=data/'processed'/plan['sft']['data_version'];out.mkdir(parents=True,exist_ok=True)
    manifest=out/'manifest.json'
    if manifest.exists():
        value=read_json(manifest)
        if value['policy_identity']!=policy:raise ValueError('SFT policy changed')
        for split in ('train','validation','test'):
            if sha256(out/f'{split}-tokens.jsonl')!=value[split+'_sha256']:raise ValueError('SFT output changed')
        return
    check_resources(plan,'sft');db=connect(out/'candidates.sqlite')
    try:
        saved=db.execute("SELECT value FROM meta WHERE name='policy'").fetchone()
        if saved and json.loads(saved[0])!=policy:raise ValueError('SFT candidate policy changed')
        db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)',('policy',json.dumps(policy)));db.commit()
        if not plan['sft'].get('skip_legacy_archive',False):
            migrate_legacy(db,data/'processed/sft-100k-teaching-v2/candidates-checkpoint.json',runs/'reports/sft_legacy_migration.json')
        tokenizer_path=Path(plan['legacy_data'])/'tokenizers/vocab-32000/tokenizer.json'
        if sha256(tokenizer_path)!=plan['tokenizer_sha256']:raise ValueError('Tokenizer changed')
        tokenizer=Tokenizer.from_file(str(tokenizer_path));blocked={normalize(q) for q in excluded_questions()}
        frozen=ROOT/'docs/run-results/general-10b-20261005/frozen-test.json'
        if frozen.exists():blocked.update(normalize(r['question']) for r in read_json(frozen)['questions'])
        bad_groups=rejected_groups();count=Counter();last_progress=0
        from scripts.sft_supplement import ensure_file,quota_ready
        enough=False;accepted_since_quota=0
        for source in plan['sft']['raw_sources']:
            if enough:break
            for filename in source['files']:
                if source.get('supplement'):
                    enough,gaps=quota_ready(db,plan)
                    accepted_since_quota=0
                    write_json(runs/'reports/sft_supplement_quota.json',dict(updated_utc=utc(),ready=enough,gaps=gaps,next_file=filename))
                    if enough:break
                if filename=='role-playing.parquet':continue
                file_key=source['repo']+'/'+source['revision']+'/'+filename
                cursor=db.execute('SELECT position,complete,counts FROM cursors WHERE file=?',(file_key,)).fetchone()
                start=cursor[0] if cursor else 0
                if cursor:count.update(json.loads(cursor[2]))
                if cursor and cursor[1]:continue
                metadata=read_json(ROOT/source['metadata'])
                if metadata['sha']!=source['revision']:raise ValueError('SFT source pin changed')
                blob=next(r for r in metadata['siblings'] if r['rfilename']==filename)
                path=ensure_file(plan,source,filename,blob)
                file_count=Counter(json.loads(cursor[2]) if cursor else {});position=start;stopped_early=False
                for position,row in enumerate(rows(path),1):
                    if position<=start:continue
                    file_count['raw_rows']+=1
                    best=None
                    for item in candidates(row,source,filename,file_key,position,blob,plan,tokenizer,blocked,bad_groups,file_count):
                        if best is None or len(item['input_ids'])>len(best['input_ids']):best=item
                    if best is not None:
                        reason=insert_candidate(db,best,plan['seed'])
                        file_count[reason or 'accepted_'+best['task']]+=1
                        if source.get('supplement') and reason is None:accepted_since_quota+=1
                    if position%2048==0:
                        db.execute('INSERT OR REPLACE INTO cursors VALUES (?,?,0,?)',(file_key,position,json.dumps(dict(file_count))));db.commit()
                        resource=check_resources(plan,'sft')
                        if source.get('supplement') and accepted_since_quota>=10000:
                            enough,gaps=quota_ready(db,plan)
                            accepted_since_quota=0
                            if enough:stopped_early=True;break
                        if time.monotonic()-last_progress>=30:
                            write_json(runs/'reports/sft_preparation_progress.json',dict(updated_utc=utc(),pid=os.getpid(),file=file_key,position=position,candidates=db.execute('SELECT COUNT(*) FROM candidates').fetchone()[0],**resource));last_progress=time.monotonic()
                db.execute('INSERT OR REPLACE INTO cursors VALUES (?,?,?,?)',(file_key,position,int(not stopped_early),json.dumps(dict(file_count))));db.commit()
                if source.get('rolling_cache'):
                    # Cursor and all accepted payloads are durable before deleting raw.
                    path.unlink()
                # Recompute counters from durable per-file accounting after resume.
                count=Counter()
                for (counts_json,) in db.execute('SELECT counts FROM cursors'):count.update(json.loads(counts_json))
                print(json.dumps(dict(file=filename,candidates=db.execute('SELECT COUNT(*) FROM candidates').fetchone()[0],counts=dict(count)),ensure_ascii=False),flush=True)
                if enough:break
        finalize(db,plan,policy,out,runs,count)
    finally:db.close()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',default='configs/50m-general-pretrain-10b-teaching.json');args=parser.parse_args()
    from scripts.worker_health import WorkerHealth
    plan=read_json(args.plan);outcome=Path(plan['runs_root'])/'reports/sft_outcome.json'
    policy=sft_identity(args.plan)
    write_json(outcome,dict(status='preparing',pid=os.getpid(),policy_identity=policy,updated_utc=utc()))
    try:
        with WorkerHealth(args.plan,'preparing_sft_cpu'):run(args.plan)
    except BaseException as exc:
        shortfall=isinstance(exc,DataShortfall)
        write_json(outcome,dict(status='data_shortfall' if shortfall else 'failed',pid=os.getpid(),policy_identity=policy,updated_utc=utc(),error=str(exc)))
        raise
    write_json(outcome,dict(status='completed',pid=os.getpid(),policy_identity=policy,updated_utc=utc()))

if __name__=='__main__':main()
