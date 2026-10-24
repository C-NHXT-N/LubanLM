"""Bounded, resumable, independent short-chat candidate preparation."""
import argparse
from collections import Counter, defaultdict, deque
import json
import os
from pathlib import Path
import random
import re
import shutil
import sqlite3
import time
import psutil
from tokenizers import Tokenizer
from scripts.common import ROOT, read_json, sha256, utc
from scripts.shortchat_io import write_json
from scripts.shortchat_common import CATEGORIES, SPLITS, SYSTEM, TASKS, digest, norm, skeleton, partition, classify, invalid
from sft.template import encode_messages
from scripts.shortchat_allocation import allocate_quality,stratified_subset

def allocate(counts, total):
    """Integral source/category max flow, with hard provenance caps."""
    cap=defaultdict(dict); original={}
    def edge(a,b,n):
        cap[a][b]=n; cap[b].setdefault(a,0); original[a,b]=n
    sources=sorted({s for s,c in counts})
    for s in sources:
        limit=total//10 if s=='rule-basics-v1' else int(total*(.15 if s=='BelleGroup/train_1M_CN' else .4))
        edge('START','S:'+s,limit)
        for c in CATEGORIES:edge('S:'+s,'C:'+c,counts.get((s,c),0))
    for c,n in CATEGORIES.items():edge('C:'+c,'END',n*total//100000)
    flow=0
    while True:
        parent={'START':None}; queue=deque(['START'])
        while queue and 'END' not in parent:
            a=queue.popleft()
            for b,n in cap[a].items():
                if n>0 and b not in parent:parent[b]=a;queue.append(b)
        if 'END' not in parent:break
        amount=total; b='END'
        while parent[b] is not None:amount=min(amount,cap[parent[b]][b]);b=parent[b]
        b='END'
        while parent[b] is not None:
            a=parent[b];cap[a][b]-=amount;cap[b][a]+=amount;b=a
        flow+=amount
    allocations={(s,c):original['S:'+s,'C:'+c]-cap['S:'+s]['C:'+c] for s in sources for c in CATEGORIES}
    return flow,allocations

class Builder:
    def __init__(self, config):
        self.config=Path(config);self.plan=read_json(config)
        self.data=Path(self.plan['data_root']);self.runs=Path(self.plan['runs_root'])
        self.data.mkdir(parents=True,exist_ok=True);(self.runs/'reports').mkdir(parents=True,exist_ok=True)
        self.folder=self.data/'processed';self.folder.mkdir(exist_ok=True)
        self.tokenizer=Tokenizer.from_file(self.plan['tokenizer'])
        if sha256(self.plan['tokenizer'])!=self.plan['tokenizer_sha256']:raise ValueError('Tokenizer changed')
        if sha256(self.plan['source_plan'])!=self.plan['source_plan_sha256']:raise ValueError('Source plan changed')
        self.source_plan=read_json(self.plan['source_plan']);self.counts=Counter();self.process=psutil.Process()
        self.db=sqlite3.connect(self.data/'candidates.sqlite',timeout=60)
        self.db.executescript('''PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL; PRAGMA cache_size=-32768; PRAGMA temp_store=MEMORY;
        CREATE TABLE IF NOT EXISTS candidates(id TEXT PRIMARY KEY,original TEXT UNIQUE,qhash TEXT UNIQUE,exact TEXT UNIQUE,family TEXT,split TEXT,category TEXT,source TEXT,short INTEGER,payload TEXT,rank TEXT,chosen INTEGER DEFAULT 0,subset INTEGER DEFAULT 0,answer_cost INTEGER);
        CREATE INDEX IF NOT EXISTS choose ON candidates(split,category,source,short,rank);
        CREATE TABLE IF NOT EXISTS cursors(name TEXT PRIMARY KEY,position INTEGER,complete INTEGER);
        CREATE TABLE IF NOT EXISTS near(bucket TEXT,fp TEXT);
        CREATE INDEX IF NOT EXISTS near_lookup ON near(bucket,fp);
        CREATE TABLE IF NOT EXISTS blocked(family TEXT PRIMARY KEY);
        CREATE TABLE IF NOT EXISTS meta(name TEXT PRIMARY KEY,value TEXT);''')
        policy=digest({'config':sha256(config),'builder':sha256(__file__),
            'code':{f:sha256(ROOT/f) for f in ('scripts/shortchat_preparation.py','scripts/shortchat_common.py','scripts/shortchat_io.py','scripts/shortchat_allocation.py','sft/template.py',
                'scripts/prepare_sft_100k.py','scripts/public_sft_formats.py','scripts/prepare_general_10b.py','scripts/corpus_quality.py','scripts/direct_mirror.py','scripts/shortchat_source_quality.py','eval/chat_suite.py')},
            'metadata':{s['repo']:sha256(ROOT/s['metadata']) for s in self.source_plan['sft']['raw_sources']}})
        saved=self.db.execute("SELECT value FROM meta WHERE name='policy'").fetchone()
        if saved and saved[0]!=policy:raise ValueError('Preparation policy changed; use a new dataset version')
        self.db.execute("INSERT OR IGNORE INTO meta VALUES('policy',?)",(policy,));self.db.commit();self.policy=policy
        self.blocked_questions=set()
        for path in [ROOT/'docs/run-results/general-10b-20261005/frozen-test.json']:
            if path.exists():
                for r in read_json(path).get('questions',[]):
                    self.blocked_questions.add(norm(r['question']))
                    self.blocked_questions.update(norm(m['content']) for m in r.get('messages',[]) if m.get('role')=='user')
        from scripts.corpus_quality import rejected_groups
        self.rejected=rejected_groups();self.last_progress=0

    def progress(self,stage,**extra):
        if self.process.memory_info().rss>3<<30:raise MemoryError('Preparation exceeded 3 GiB')
        if shutil.disk_usage(self.data).free<20<<30:raise RuntimeError('Disk reserve reached')
        write_json(self.runs/'reports/preparation_progress.json',dict(stage=stage,updated_utc=utc(),counts=dict(self.counts),rss_bytes=self.process.memory_info().rss,**extra))

    def add(self, meta, messages, *, rule=None):
        if self.plan.get('source_quality_v2') and not rule:
            from scripts.shortchat_source_quality import message_reason
            # Check only the prospective initial prefix; a later long turn cannot
            # invalidate an otherwise complete initial exchange.
            reason=message_reason(messages[:2],human=meta.get('source')=='silver/lccc')
            if reason:self.counts[reason]+=1;return
        if meta.get('group') in self.rejected:return
        if messages and not rule:
            from scripts.prepare_general_2b import key
            from eval.chat_suite import normalize
            if key(normalize(messages[0].get('content',''))) in self.rejected:self.counts['known_rejected_family']+=1;return
        original=digest([meta.get('source'),meta.get('raw_sha256'),meta.get('source_row',meta.get('group'))])
        if rule:
            category=rule['category'];chosen=messages
        else:
            if not messages or len(messages)<2:return
            # Public initial turns are self-contained; never extract a middle turn.
            chosen=messages[:2];category=classify(chosen,meta.get('file',''))
            if category=='daily_single':
                variants=[]
                for end in (8,6,4):
                    ms=messages[:end]
                    if len(ms)!=end:continue
                    if self.plan.get('source_quality_v2') and message_reason(ms,human=meta.get('source')=='silver/lccc'):continue
                    if encode_messages(self.tokenizer,ms,512,128,system=SYSTEM):
                        if invalid(ms,'daily_multi'):continue
                        lengths=[len(self.tokenizer.encode(m['content'],add_special_tokens=False).ids) for m in ms[1::2]]
                        variants.append((max(lengths)<=64,end,ms))
                if variants:chosen=max(variants,key=lambda v:(v[0],v[1]))[2];category='daily_multi'
            if category is None:self.counts['uncertain_category']+=1;return
        if any(norm(m['content']) in self.blocked_questions for m in chosen[::2]):self.counts['old_test_overlap']+=1;return
        reason=invalid(chosen,category)
        if reason:self.counts[reason]+=1;return
        text='\n'.join(m['content'] for m in chosen)
        if re.search(r'[\u3040-\u30ff\uac00-\ud7af\u0400-\u04ff]',text):self.counts['script']+=1;return
        if category not in ('basic','instruction') and sum('\u4e00'<=c<='\u9fff' for c in text)/max(1,len(norm(text)))<.4:self.counts['language']+=1;return
        encoded=encode_messages(self.tokenizer,chosen,512,128,system=SYSTEM)
        if not encoded:self.counts['length_or_reserved_token']+=1;return
        lengths=[len(self.tokenizer.encode(m['content'],add_special_tokens=False).ids) for m in chosen[1::2]]
        first=skeleton(chosen[0]['content']);family=rule['family'] if rule else digest('public-question:'+first)
        if self.db.execute('SELECT 1 FROM blocked WHERE family=?',(family,)).fetchone():return
        qhash=digest([SYSTEM,*[(m['role'],norm(m['content'])) for m in chosen[:-1]]])
        exact=digest([(m['role'],norm(m['content'])) for m in chosen])
        old=self.db.execute('SELECT category,payload FROM candidates WHERE original=?',(original,)).fetchone()
        replacing=False
        if old and category=='daily_multi':
            previous=json.loads(old[1])
            replacing=old[0]=='daily_single' or (old[0]=='daily_multi' and (max(lengths)<=64,len(chosen))>(max(previous['answer_tokens'])<=64,len(previous['messages'])))
        if old and not replacing:return
        # Simhash prompt-near dedup. Rule variations share explicit isolated template families.
        fp=None
        if not rule and not replacing and len(first)>=20:
            from scripts.prepare_general_10b import fingerprint
            fp=fingerprint(first)
            for shift in (0,16,32,48):
                bucket=f'{shift}:{(fp>>shift)&65535}'
                for (value,) in self.db.execute('SELECT fp FROM near WHERE bucket=?',(bucket,)):
                    if (fp^int(value)).bit_count()<=3:self.counts['near_duplicate']+=1;return
        item=dict(encoded,id=digest([original,qhash]),messages=chosen,system=SYSTEM,category=category,
            task=TASKS[category],family_id=family,source=meta['source'],revision=meta.get('revision'),
            file=meta.get('file'),raw_sha256=meta.get('raw_sha256'),source_row=meta.get('source_row'),
            answer_tokens=lengths,generated=bool(rule),rule=rule)
        try:
            self.db.execute('SAVEPOINT candidate')
            if replacing:self.db.execute('DELETE FROM candidates WHERE original=?',(original,))
            self.db.execute('INSERT INTO candidates(id,original,qhash,exact,family,split,category,source,short,payload,rank,answer_cost) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                (item['id'],original,qhash,exact,family,partition(family),category,meta['source'],int(max(lengths)<=64),json.dumps(item,ensure_ascii=False),digest(str(self.plan['selection_seed'])+item['id']),8*len(lengths)-10*sum(v<=64 for v in lengths)))
            self.db.execute('RELEASE candidate')
        except sqlite3.IntegrityError:
            self.db.execute('ROLLBACK TO candidate');self.db.execute('RELEASE candidate')
            self.counts['exact_duplicate']+=1;return
        if fp is not None:
            for shift in (0,16,32,48):self.db.execute('INSERT INTO near VALUES(?,?)',(f'{shift}:{(fp>>shift)&65535}',str(fp)))
        self.counts['accepted_'+category]+=1

    def ingest(self,name,iterator):
        saved=self.db.execute('SELECT position,complete FROM cursors WHERE name=?',(name,)).fetchone()
        if saved and saved[1]:return
        start=saved[0] if saved else 0
        for index,(meta,ms) in enumerate(iterator):
            if index<start:continue
            self.add(meta,ms);self.counts['scanned']+=1
            if index%1000==0:
                self.db.execute('INSERT OR REPLACE INTO cursors VALUES(?,?,0)',(name,index+1));self.db.commit()
                self.progress('ingesting',file=name,position=index+1)
        self.db.execute('INSERT OR REPLACE INTO cursors VALUES(?,?,1)',(name,index+1 if 'index' in locals() else 0));self.db.commit()

    def old_pool(self):
        path=Path(self.plan['old_pool']);db=sqlite3.connect('file:'+path.as_posix()+'?mode=ro&immutable=1',uri=True)
        try:
            for (payload,) in db.execute('SELECT payload FROM candidates ORDER BY rowid'):
                item=json.loads(payload)
                if item.get('generated'):continue
                yield item,item['messages']
        finally:db.close()

    def rules(self):
        rng=random.Random(self.plan['selection_seed'])
        prefixes=['请直接回答：','我想问一下：','帮我确认一下：','用一句话告诉我：','请简短说明：','我有个小问题：','请明确回答：','我想了解：','请给我一个简短答案：','可以告诉我吗：','麻烦回答这个问题：','请不要展开解释：']
        suffixes=['',' 请用一句话回答。',' 回答简短一些。',' 请直接给出结论。',' 不需要长篇解释。',' 请准确回答。']
        actions=['整理书包','收拾桌面','去商店买东西','开门','给朋友送礼物','到公园散步','搬动椅子','帮我洗衣服','倒一杯水','照顾花草','到学校接我','把书放进抽屉','替我打扫房间','替我跑腿','来我家帮忙','给我做饭','替我遛狗','替我搬书','帮我叠衣服','把垃圾拿出去','帮我拿快递','替我寄信','替我修理台灯','帮我擦窗户','把鞋放到门口','陪我去图书馆','替我整理衣柜','帮我浇花','替我去排队','到现场布置房间','替我洗碗','替我整理床铺','帮我搬行李','替我取外卖','替我把门关上','帮我折纸','替我擦桌子','替我晒衣服','帮我把袋子提回家','替我把杯子洗干净']
        for pi,prefix in enumerate(prefixes):
            for si,suffix in enumerate(suffixes):
                for ai,action in enumerate(actions):
                    for phrasing in ['你能真的{a}吗？','你能实际完成{a}这件事吗？','我需要你{a}，你能实际做到吗？']:
                        q=prefix+phrasing.format(a=action)+suffix
                        a='我没有身体，不能替你实际行动，但可以帮你想办法或列出步骤。'
                        family=f'boundary-template:{pi}:{si}:{phrasing}'
                        rule=dict(category='boundary',family=family,kind='physical_boundary',expected='cannot_act_physically')
                        yield dict(source='rule-basics-v1',revision='1',file='boundary',source_row=digest(q)),[dict(role='user',content=q),dict(role='assistant',content=a)],rule
                for form in ['你是什么类型的助手？','你的名字是什么？','你有自己的身体吗？','你能够帮我整理文字吗？','你能直接看到我的房间吗？','你现在能直接查询网络吗？']:
                    # Identity variants remain complete answers and use different template families.
                    answers={'你是什么类型的助手？':'我是LubanLM，一个中文AI助手。','你的名字是什么？':'我叫LubanLM，是一个中文AI助手。','你有自己的身体吗？':'我没有身体，只能通过文字与你交流。','你能够帮我整理文字吗？':'可以，你把文字发给我，我可以帮你整理。','你能直接看到我的房间吗？':'我不能直接看到你的房间，你可以用文字描述。','你现在能直接查询网络吗？':'我不能直接查询网络，可以根据已有信息和你提供的内容回答。'}
                    q=prefix+form+suffix;rule=dict(category='boundary',family=f'identity:{pi}:{si}:{form}',kind='identity',expected=answers[form])
                    yield dict(source='rule-basics-v1',revision='1',file='identity',source_row=digest(q)),[dict(role='user',content=q),dict(role='assistant',content=answers[form])],rule
        for pi,prefix in enumerate(prefixes):
            for si,suffix in enumerate(suffixes):
                for op in ['+','-','×']:
                    for _ in range(160):
                        a,b=rng.randrange(100),rng.randrange(100)
                        q=f'{prefix}计算 {a}{op}{b} 的结果。{suffix}'
                        expected=a+b if op=='+' else a-b if op=='-' else a*b
                        rule=dict(category='basic',family=f'math-template:{pi}:{si}:{op}',kind='integer',expected=str(expected))
                        yield dict(source='rule-basics-v1',revision='1',file='arithmetic',source_row=digest(q)),[dict(role='user',content=q),dict(role='assistant',content=str(expected))],rule

    def ingest_rules(self):
        if self.db.execute("SELECT 1 FROM cursors WHERE name='rules-v1' AND complete=1").fetchone():return
        for i,(meta,ms,rule) in enumerate(self.rules()):
            self.add(meta,ms,rule=rule)
            if i%1000==0:self.db.commit();self.progress('rules',position=i)
        self.db.execute("INSERT OR REPLACE INTO cursors VALUES('rules-v1',0,1)");self.db.commit()

    def inventories(self):
        return [dict(split=s,source=src,category=c,count=n) for s,src,c,n in self.db.execute('SELECT split,source,category,count(*) FROM candidates GROUP BY split,source,category')]

    def feasible(self):
        available=self.inventories();gaps={}
        for split,n in SPLITS.items():
            counts={(r['source'],r['category']):r['count'] for r in available if r['split']==split}
            quality={(s,c,cost):amount for s,c,cost,amount in self.db.execute('SELECT source,category,answer_cost,count(*) FROM candidates WHERE split=? GROUP BY source,category,answer_cost',(split,))}
            flow,alloc,cost=allocate_quality(quality,n)
            if flow<n or cost>0:gaps[split]=dict(required=n,flow=flow,missing=n-flow,minimum_answer_penalty=cost,categories={c:sum(v for (s,k),v in counts.items() if k==c) for c in CATEGORIES})
        return gaps

    def supplements(self):
        from scripts.prepare_sft_100k import rows,messages
        from scripts.direct_mirror import PhysicalMirror
        mirror=None;raw=self.data/'raw';raw.mkdir(exist_ok=True)
        # Only these pinned sources may fill the initial cache's shortfall.
        repos=self.plan.get('supplement_repos',['shareAI/ShareGPT-Chinese-English-90k','allenai/WildChat-1M','OpenMOSS-Team/moss-003-sft-data'])
        cache=Path(self.source_plan['sft_raw_root'])
        sources=self.source_plan['sft']['raw_sources']
        for repo in repos:
            source=next(s for s in sources if s['repo']==repo)
            info=read_json(ROOT/source['metadata'])
            if info.get('sha')!=source['revision']:raise ValueError('Source metadata revision mismatch')
            blobs={r.get('path',r.get('rfilename')):r for r in info['siblings']}
            for filename in source['files']:
                name=repo+'@'+source['revision']+':'+filename
                if self.db.execute('SELECT 1 FROM cursors WHERE name=? AND complete=1',(name,)).fetchone():continue
                blob=blobs[filename];expected=blob['lfs']['sha256'];size=blob.get('size') or blob['lfs']['size']
                paths=list(cache.glob('**/'+Path(filename).name));path=next((p for p in paths if p.stat().st_size==size and sha256(p)==expected),None)
                fresh=False
                if path is None:
                    if size>4<<30:raise RuntimeError('Archive exceeds rolling cache limit')
                    self.progress('downloading',repo=repo,file=filename)
                    if mirror is None:mirror=PhysicalMirror(self.runs/'reports/network.jsonl')
                    path=raw/(repo.replace('/','--')+'--'+Path(filename).name)
                    path=mirror.download(f'{mirror.endpoint}/datasets/{repo}/resolve/{source["revision"]}/{filename}',path,size,expected,guard=lambda:self.progress('downloading',repo=repo,file=filename))
                    fresh=True
                def iterator():
                    for i,row in enumerate(rows(path)):
                        if self.plan.get('source_quality_v2'):
                            from scripts.shortchat_source_quality import row_reason
                            reason=row_reason(row,source)
                            if reason:self.counts[reason]+=1;continue
                        if repo=='allenai/WildChat-1M' and (row.get('language')!='Chinese' or row.get('toxic') or row.get('redacted') or any(m.get('toxic') or m.get('redacted') for m in row.get('conversation',[]))):continue
                        ms=messages(row)
                        if ms and ms[0].get('role')=='system':ms=ms[1:]
                        yield dict(source=repo,revision=source['revision'],file=filename,raw_sha256=expected,source_row=i),ms
                self.ingest(name,iterator())
                write_json(self.runs/'reports/availability.json',dict(inventory=self.inventories(),shortfall=self.feasible()))
                if fresh:
                    if not path.resolve().is_relative_to(raw.resolve()):raise ValueError('Unsafe rolling cleanup target')
                    path.unlink()
                if not self.feasible():return

    def select(self, prepare_splits=None):
        from scripts.shortchat_preparation import export_selection
        return export_selection(self,prepare_splits)

    def run(self, download, prepare_splits=None):
        if self.plan.get('reuse_candidates_v3'):
            from scripts.shortchat_preparation import reuse_candidates,curated_dev
            reuse_candidates(self);curated_dev(self)
        else:
            self.ingest('old-qualified-pool-v1',self.old_pool());self.ingest_rules()
        if download and self.feasible():self.supplements()
        self.select(prepare_splits);self.progress('preparation_finished',shortfall=self.feasible());self.db.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--download',action='store_true');p.add_argument('--prepare-splits',nargs='+',choices=[*SPLITS,'subset']);args=p.parse_args()
    Builder(args.plan).run(args.download,args.prepare_splits)

if __name__=='__main__':main()
