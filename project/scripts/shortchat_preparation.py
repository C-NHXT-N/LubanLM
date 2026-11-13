"""Versioned reuse and independently resumable, fail-closed split exports."""
from collections import Counter,defaultdict
import json
from pathlib import Path
import sqlite3
from scripts.common import read_json,sha256,utc
from scripts.shortchat_io import write_json
from scripts.shortchat_common import CATEGORIES,SPLITS,TASKS,SYSTEM,classify,invalid,digest,norm,skeleton
from scripts.shortchat_allocation import allocate_quality,stratified_subset
from sft.template import encode_messages


def invalidate(builder,reason):
    """Invalidate completion and its semantic decisions before changing data."""
    for name in ('manifest.json','review-samples.json','review-marks.json'):
        path=builder.folder/name
        if path.exists():
            archive=builder.folder/'superseded';archive.mkdir(exist_ok=True)
            path.replace(archive/(sha256(path)[:16]+'-'+name))
    for name in ('preparation_complete.json','data_audit.json','preflight.json'):
        path=builder.runs/'reports'/name
        if path.exists():path.unlink()
    write_json(builder.folder/'invalidation.json',dict(reason=reason,policy=builder.policy,updated_utc=utc()))


def validate_boundary_review(folder,policy):
    samples=read_json(folder/'public-boundary-review-samples.json')
    marks=read_json(folder/'public-boundary-review-marks.json')
    if marks.get('policy')!=policy or marks.get('samples_sha256')!=sha256(folder/'public-boundary-review-samples.json'):
        raise ValueError('Public boundary review binding changed')
    expected={r['id'] for r in samples};items=marks.get('marks',[])
    if len(items)!=len(expected) or {r['id'] for r in items}!=expected:raise ValueError('Public boundary review incomplete')
    for row in items:
        if any(type(row.get(k)) is not bool for k in ('severe_error','category_correct','usable')) or not row.get('reason'):
            raise ValueError('Explicit public boundary decisions required')
        if row['severe_error'] or not row['category_correct'] or not row['usable']:raise ValueError('Rejected public boundary family remains')
    return dict(samples=len(items),status='passed')


def reuse_candidates(b):
    name='reuse-qualified-v2-v3'
    if b.db.execute('SELECT 1 FROM cursors WHERE name=? AND complete=1',(name,)).fetchone():return
    invalidate(b,'reclassification')
    source=Path(b.plan['old_pool'])
    if source.resolve()==(b.data/'candidates.sqlite').resolve():raise ValueError('Cannot mutate reuse source')
    db=sqlite3.connect('file:'+source.as_posix()+'?mode=ro&immutable=1',uri=True)
    db.row_factory=sqlite3.Row
    saved=b.db.execute('SELECT position FROM cursors WHERE name=?',(name,)).fetchone();start=saved[0] if saved else 0
    stats=Counter(json.loads(b.db.execute("SELECT value FROM meta WHERE name='correction_stats'").fetchone()[0])) if start else Counter()
    if sha256(source)!=b.plan['reuse_candidates_sha256']:raise ValueError('Pinned candidate snapshot changed')
    corrections_path=Path(b.plan['known_corrections'])
    if sha256(corrections_path)!=b.plan['known_corrections_sha256']:raise ValueError('Known semantic corrections changed')
    corrections=read_json(corrections_path)
    for record in corrections:
        if not record['usable']:
            b.db.execute('INSERT OR IGNORE INTO blocked VALUES(?)',(record['family'],));b.db.execute('DELETE FROM candidates WHERE family=?',(record['family'],))
    write_json(b.runs/'reports/known_semantic_corrections.json',dict(policy=b.policy,records=corrections))
    from scripts.shortchat_source_quality import message_reason
    try:
        last_position=start
        for row in db.execute('SELECT rowid AS position,* FROM candidates WHERE rowid>? ORDER BY rowid',(start,)):
            last_position=row['position']
            r=json.loads(row['payload']);ms=r['messages'];reason=None
            category=r['category'] if r['generated'] else classify(ms,r.get('file',''))
            if category is None:reason='uncertain_category'
            elif r['generated'] and row['split']=='dev' and category=='boundary':reason='replaced_dev_boundary_group'
            else:
                reason=invalid(ms,category) or (None if r['generated'] else message_reason(ms,human=r['source']=='silver/lccc'))
            if any(norm(m['content']) in b.blocked_questions for m in ms[::2]):reason='old_test_overlap'
            if b.db.execute('SELECT 1 FROM blocked WHERE family=?',(r['family_id'],)).fetchone():reason='removed_family'
            if reason:
                stats['removed:'+reason]+=1
                b.db.execute('INSERT OR IGNORE INTO blocked VALUES(?)',(r['family_id'],))
                b.db.execute('DELETE FROM candidates WHERE family=?',(r['family_id'],))
            else:
                # The read-only v2 snapshot, tokenizer and encoding are pinned.
                # Preserve its exact labels; re-encode every selected export in
                # the mandatory structural audit rather than all unused rows.
                if len(r['input_ids'])!=len(r['labels']) or len(r['input_ids'])>512 or max(r['answer_tokens'])>128 or sum(v>=0 for v in r['labels'])!=r['assistant_tokens']:
                    raise ValueError('Reused sample encoding metadata changed: '+r['id'])
                if category!=r['category']:stats['reclassified:'+r['category']+'->'+category]+=1
                r.update(category=category,task=TASKS[category]);stats['kept']+=1
                b.db.execute('INSERT OR IGNORE INTO candidates(id,original,qhash,exact,family,split,category,source,short,payload,rank,answer_cost) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                    (r['id'],row['original'],row['qhash'],row['exact'],row['family'],row['split'],category,row['source'],row['short'],json.dumps(r,ensure_ascii=False),row['rank'],row['answer_cost']))
            if row['position']%1000==0:
                b.db.execute('INSERT OR REPLACE INTO cursors VALUES(?,?,0)',(name,row['position']))
                b.db.execute("INSERT OR REPLACE INTO meta VALUES('correction_stats',?)",(json.dumps(stats),));b.db.commit()
                b.progress('reclassifying',position=row['position'],corrections=dict(stats))
        # Retain the original global prompt near-dedup index for any new records.
        b.db.execute('DELETE FROM near')
        near=db.execute('SELECT bucket,fp FROM near')
        while chunk:=near.fetchmany(4000):b.db.executemany('INSERT INTO near VALUES(?,?)',chunk)
        # Raw sources are already ingested; no repeat downloads on continuation.
        for row in db.execute("SELECT name,position,complete FROM cursors WHERE name NOT LIKE 'old-qualified%' AND name!='rules-v1'"):
            b.db.execute('INSERT OR REPLACE INTO cursors VALUES(?,?,?)',tuple(row))
        b.db.execute("INSERT OR REPLACE INTO cursors VALUES('rules-v1',0,1)")
        b.db.execute('INSERT OR REPLACE INTO cursors VALUES(?,?,1)',(name,last_position))
        b.db.execute("INSERT OR REPLACE INTO meta VALUES('correction_stats',?)",(json.dumps(stats),));b.db.commit()
        write_json(b.runs/'reports/classification_corrections.json',dict(status='reclassified',source=str(source),source_policy=db.execute("SELECT value FROM meta WHERE name='policy'").fetchone()[0],policy=b.policy,statistics=dict(stats),retained=b.db.execute('SELECT count(*) FROM candidates').fetchone()[0],updated_utc=utc()))
    finally:db.close()


def curated_dev(b):
    fixture=Path(b.plan['curated_dev_boundary'])
    if sha256(fixture)!=b.plan['curated_dev_boundary_sha256']:raise ValueError('Curated development questions changed')
    if b.db.execute("SELECT 1 FROM cursors WHERE name='curated-dev-v3' AND complete=1").fetchone():return
    invalidate(b,'replace entire development boundary category')
    from scripts.prepare_general_10b import fingerprint
    questions=read_json(fixture);removed=set();evidence=[]
    if len(questions)!=10 or Counter(q['kind'] for q in questions)!={k:2 for k in ('greeting','identity','physical','perception','network')}:
        raise ValueError('Development boundary coverage changed')
    fps=[fingerprint(skeleton(q['question'])) for q in questions]
    for q,fp in zip(questions,fps):
        if norm(q['question']) in b.blocked_questions or any(len(old)>=20 and (fingerprint(skeleton(old))^fp).bit_count()<=3 for old in b.blocked_questions):
            raise ValueError('Curated question overlaps an old test; stop instead of rewriting')
    replaced=[dict(id=id,family=family,reason='entire development boundary group replaced by authored questions') for id,family in b.db.execute("SELECT id,family FROM candidates WHERE split='dev' AND category='boundary'")]
    removed.update(r['family'] for r in replaced)
    for family,payload in b.db.execute('SELECT family,payload FROM candidates'):
        r=json.loads(payload)
        for user in r['messages'][::2]:
            question=norm(user['content']);fp=fingerprint(skeleton(question)) if len(question)>=20 else None
            if any(question==norm(q['question']) or (fp is not None and (fp^p).bit_count()<=3) for q,p in zip(questions,fps)):removed.add(family)
    for family in removed:
        b.db.execute('INSERT OR IGNORE INTO blocked VALUES(?)',(family,));b.db.execute('DELETE FROM candidates WHERE family=?',(family,))
    for q in questions:
        rule=dict(category='boundary',family='v3-exclusive-dev:'+q['key'],kind=q['kind'],expected=q['reference'],reference_points=q['points'],evaluation_only=True)
        b.add(dict(source='rule-basics-v1',revision='v3-manual-dev',file='curated-dev-boundary',raw_sha256=sha256(fixture),source_row=digest(q['question'])),
            [dict(role='user',content=q['question']),dict(role='assistant',content=q['reference'])],rule=rule)
        b.db.execute("UPDATE candidates SET split='dev' WHERE family=?",(rule['family'],))
        evidence.append(dict(key=q['key'],family=rule['family'],question_sha256=digest(norm(q['question']))))
    b.db.execute("INSERT OR REPLACE INTO cursors VALUES('curated-dev-v3',10,1)");b.db.commit()
    write_json(b.runs/'reports/curated_dev_isolation.json',dict(policy=b.policy,fixture_sha256=sha256(fixture),replaced_candidates=replaced,removed_overlapping_families=sorted(removed),questions=evidence,old_test_overlap=0))


def audit_sample_ids(b,clause,total):
    """Largest remainders across category/source/answer-length strata."""
    selected=set()
    for category,n in CATEGORIES.items():
        take=n*total//100000
        bins=list(b.db.execute(f'SELECT source,short,count(*) FROM candidates WHERE {clause} AND category=? GROUP BY source,short',(category,)))
        denominator=sum(n for s,l,n in bins);alloc={(s,l):n*take//denominator for s,l,n in bins}
        remainder=take-sum(alloc.values())
        order=sorted(bins,key=lambda r:(-(r[2]*take%denominator),digest(str(b.plan['selection_seed'])+str(r[:2]))))
        for s,l,n in order[:remainder]:alloc[s,l]+=1
        for (source,short),amount in alloc.items():
            selected.update(r[0] for r in b.db.execute(f'SELECT id FROM candidates WHERE {clause} AND category=? AND source=? AND short=? ORDER BY rank LIMIT ?',(category,source,short,amount)))
    return selected


def export_selection(b,prepare_splits=None):
    requested=list(prepare_splits or [*SPLITS,'subset'])
    if any(s not in {*SPLITS,'subset'} for s in requested):raise ValueError('Unknown preparation split')
    if 'subset' in requested and 'train' not in requested:
        if b.db.execute("SELECT count(*) FROM candidates WHERE chosen=1 AND split='train'").fetchone()[0]!=100000:raise ValueError('Subset requires determined 100,000 train records')
    state_path=b.folder/'preparation-state.json'
    state=read_json(state_path) if state_path.exists() else dict(policy=b.policy,completed={})
    if state['policy']!=b.policy:raise ValueError('Partial artifacts use a different candidate policy')
    gaps=b.feasible();changed=False
    for split in requested:
        if split in gaps:continue
        path=b.folder/(split+'-tokens.jsonl');previous=state['completed'].get(split)
        if previous and path.exists() and sha256(path)==previous['sha256']:continue
        if not changed:invalidate(b,'split selection or refill');changed=True
        if split!='subset':
            b.db.execute('UPDATE candidates SET chosen=0 WHERE split=?',(split,))
            counts={(s,c,cost):n for s,c,cost,n in b.db.execute('SELECT source,category,answer_cost,count(*) FROM candidates WHERE split=? GROUP BY source,category,answer_cost',(split,))}
            flow,allocation,cost=allocate_quality(counts,SPLITS[split])
            if flow!=SPLITS[split] or cost>0:raise ValueError('Infeasible selected split')
            for (source,category,cost),amount in allocation.items():
                b.db.execute('UPDATE candidates SET chosen=1 WHERE id IN (SELECT id FROM candidates WHERE split=? AND source=? AND category=? AND answer_cost=? ORDER BY rank LIMIT ?)',(split,source,category,cost,amount))
            if split=='train':
                state['completed'].pop('subset',None);b.db.execute('UPDATE candidates SET subset=0')
        else:
            if b.db.execute("SELECT count(*) FROM candidates WHERE chosen=1 AND split='train'").fetchone()[0]!=100000:
                gaps['subset']=dict(missing=20000,reason='training split not complete');continue
            b.db.execute('UPDATE candidates SET subset=0')
            bins={(s,c,l,v):n for s,c,l,v,n in b.db.execute("SELECT source,category,short,answer_cost,count(*) FROM candidates WHERE chosen=1 AND split='train' GROUP BY source,category,short,answer_cost")}
            for (s,c,l,v),amount in stratified_subset(bins).items():
                b.db.execute("UPDATE candidates SET subset=1 WHERE id IN (SELECT id FROM candidates WHERE chosen=1 AND split='train' AND source=? AND category=? AND short=? AND answer_cost=? ORDER BY rank LIMIT ?)",(s,c,l,v,amount))
        b.db.commit();clause='subset=1' if split=='subset' else 'chosen=1 AND split='+repr(split)
        count=targets=short=turns=0;cats=Counter();sources=Counter();suite=[]
        tmp=path.with_suffix('.tmp')
        with tmp.open('w',encoding='utf-8') as stream:
            for id,payload in b.db.execute(f'SELECT id,payload FROM candidates WHERE {clause} ORDER BY rank'):
                r=json.loads(payload);stream.write(payload+'\n');count+=1;targets+=r['assistant_tokens'];cats[r['category']]+=1;sources[r['source']]+=1
                short+=sum(v<=64 for v in r['answer_tokens']);turns+=len(r['answer_tokens'])
                if split in ('dev','test'):suite.append(dict(id=id,category=r['category'],messages=r['messages'][:-1],reference=r['messages'][-1]['content'],family_id=r['family_id'],rule=r['rule']))
        expected=20000 if split=='subset' else SPLITS[split]
        if count!=expected or short/turns<.8:raise ValueError('Incomplete split or short-answer shortfall')
        tmp.replace(path)
        if suite:write_json(b.folder/(split+'-suite.json'),dict(split=split,questions=suite))
        state['completed'][split]=dict(sha256=sha256(path),stats=dict(examples=count,categories=dict(cats),sources=dict(sources),short_answer_fraction=short/turns,supervised_tokens=targets))
        state.update(updated_utc=utc(),missing=gaps);write_json(state_path,state);b.progress('exported_split',split=split,completed=list(state['completed']),shortfall=gaps)
    missing=[s for s in [*SPLITS,'subset'] if s not in state['completed']]
    state.update(missing_splits=missing,shortfall=gaps,status='partial' if missing else 'structural_check_pending');write_json(state_path,state)
    write_json(b.runs/'reports/data_shortfall.json',dict(status=state['status'],missing_splits=missing,gaps=gaps,inventory=b.inventories()))
    # Public identity answers require their own per-record semantic audit.
    boundary=[dict(json.loads(p),split=s) for p,s in b.db.execute("SELECT payload,split FROM candidates WHERE category='boundary' AND source!='rule-basics-v1' ORDER BY rank")]
    write_json(b.folder/'public-boundary-review-samples.json',boundary)
    if missing:return False
    hashes={s:r['sha256'] for s,r in state['completed'].items()};stats={s:r['stats'] for s,r in state['completed'].items()}
    artifacts=dict(hashes=hashes,policy=b.policy,tokenizer_sha256=b.plan['tokenizer_sha256'],suite_hashes={s:sha256(b.folder/(s+'-suite.json')) for s in ('dev','test')})
    samples=[]
    for split,total in (('train',600),('validation',200)):
        clause='chosen=1 AND split='+repr(split);ids=audit_sample_ids(b,clause,total)
        samples.extend(dict(json.loads(p),split=split) for id,p in b.db.execute(f'SELECT id,payload FROM candidates WHERE {clause} ORDER BY rank') if id in ids)
    manifest=dict(status='content_review_pending',artifacts=artifacts,stats=stats,selection_seed=b.plan['selection_seed'],created_utc=utc())
    # Full structural audit is performed against an unpublished proposed manifest.
    write_json(b.folder/'manifest-proposed.json',manifest)
    from scripts.compare_shortchat import check_data
    check_data(b.plan,require_review=False,manifest_path=b.folder/'manifest-proposed.json')
    write_json(b.folder/'review-samples.json',samples);write_json(b.folder/'manifest.json',manifest)
    write_json(b.runs/'reports/preparation_complete.json',dict(status='content_review_pending',manifest_sha256=sha256(b.folder/'manifest.json'),stats=stats))
    return True
