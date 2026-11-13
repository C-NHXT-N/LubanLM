"""流式扩展语料；保留 tokenizer，不改动 v1。通过独立 LUBAN_DATA/RUNS 运行。"""
from __future__ import annotations
from scripts.common import DATA, RUNS, write_json, read_json, sha256, utc
from data.prepare import clean, load_jsonl, save_jsonl
from eval.chat_suite import excluded_questions, normalize
from sft.template import encode_pair
from tokenizers import Tokenizer
from huggingface_hub import HfApi
from datasets import load_dataset
from pathlib import Path
import argparse
import hashlib
import json
import re
import shutil
import numpy as np

def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def records(path):
    with Path(path).open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

class Shards:
    def __init__(self, directory, tokenizer_hash):
        self.directory=directory
        directory.mkdir(parents=True,exist_ok=True)
        self.shards=[]
        self.chunk=[]
        self.tokens=0
        self.documents=0
        self.tokenizer_hash=tokenizer_hash
    def add(self,ids):
        self.chunk.extend(ids)
        self.tokens+=len(ids)
        self.documents+=1
        if len(self.chunk)>=2_000_000:
            self.flush()
    def flush(self):
        if self.chunk:
            path=self.directory/f'{len(self.shards):04d}.bin'
            np.asarray(self.chunk,dtype=np.uint16).tofile(path)
            self.shards.append(dict(path=str(path),tokens=len(self.chunk),sha256=sha256(path)))
            self.chunk=[]
    def finish(self):
        self.flush()
        manifest=dict(dtype='uint16',shards=self.shards,tokens=self.tokens,documents=self.documents,tokenizer_sha256=self.tokenizer_hash,packing='EOS between documents; cross-document attention allowed')
        write_json(self.directory/'manifest.json',manifest)
        return manifest

def prepare_wiki(legacy, tokenizer, limit):
    manifest_path=RUNS/'reports/wiki-v2.json'
    if manifest_path.exists():
        return read_json(manifest_path)
    source=read_json(legacy/'raw/wikipedia-zh.source.json')
    original_valid={r['sha256'] for r in records(legacy/'processed/wiki-validation.jsonl')}
    writers={s:Shards(DATA/f'processed/pretrain-{s}',sha256(DATA/'tokenizers/vocab-32000/tokenizer.json')) for s in ('train','validation')}
    handles={s:(DATA/f'processed/wiki-{s}.jsonl').open('w',encoding='utf-8') for s in writers}
    seen=set()
    raw_count=0
    blocked=excluded_questions()
    def consume(row):
        nonlocal raw_count
        raw_count+=1
        text=clean(row['text'])
        if not 80<=len(text)<=100000 or any(t in text for t in ('<bos>','<eos>','<pad>')):
            return
        key=digest(text)
        if key in seen:
            return
        seen.add(key)
        split='validation' if key in original_valid or int(key[:8],16)%100<2 else 'train'
        # 对文章仅排除完整问题可见拼写；普通知识内容可自然出现。
        normalized_text=normalize(text)
        if split=='train' and any(q in normalized_text for q in blocked if len(q)>=12):
            return
        record=dict(text=text,sha256=key,id=row['id'],title=row.get('title',''),split=split)
        handles[split].write(json.dumps(record,ensure_ascii=False)+'\n')
        writers[split].add(tokenizer.encode(text,add_special_tokens=False).ids+[tokenizer.token_to_id('<eos>')])
        if raw_count%5000==0:
            print('wiki',raw_count,{s:w.tokens for s,w in writers.items()},flush=True)
            write_json(RUNS/'reports/preparation_progress.json',dict(stage='wiki',documents=raw_count,tokens={s:w.tokens for s,w in writers.items()},updated_utc=utc()))
    try:
        for row in records(legacy/'raw/wikipedia-zh.jsonl'):
            if raw_count>=limit:
                break
            consume(row)
        already=raw_count
        if raw_count<limit:
            ds=load_dataset(source['repo'],source['config'],revision=source['revision'],split='train',streaming=True)
            for index,row in enumerate(ds):
                if index<already:
                    continue
                consume(row)
                if raw_count>=limit:
                    break
    finally:
        for h in handles.values():
            h.close()
    result=dict(source={k:source[k] for k in ('repo','revision','config','license')},legacy_seed=source,requested_documents=limit,raw_documents=raw_count,selection='first requested records of fixed source revision, streaming; legacy prefix reused',processed_sha256={s:sha256(DATA/f'processed/wiki-{s}.jsonl') for s in writers},original_validation_preserved=True,train=writers['train'].finish(),validation=writers['validation'].finish(),created_utc=utc())
    write_json(manifest_path,result)
    return result

FACTS=[
 ('电脑','电脑是一种处理信息的工具，可以用来学习、工作和交流。'),
 ('键盘','键盘用来向电脑输入文字、数字和指令。'),
 ('鼠标','鼠标用来移动指针、点击选项和操作电脑。'),
 ('浏览器','浏览器是一种用来访问网页的软件。'),
 ('互联网','互联网连接许多计算机，让人们分享信息和交流。'),
 ('文件夹','文件夹用来整理和存放文件。'),
 ('Linux','Linux是一种开源操作系统内核，日常也用来指基于它的操作系统。'),
 ('操作系统','操作系统管理电脑的硬件和软件，帮助用户运行程序。'),
 ('Token','Token是模型处理文字时使用的基本单元，可以是字、词或词的一部分。'),
 ('语言模型','语言模型学习文字中的规律，并根据前面的内容预测后面的文字。'),
 ('手机充电','给手机充电可以补充电池的电量。'),
 ('冰融化','冰受热会融化成水。'),
 ('阳光','植物通过光合作用利用光能制造有机物。'),
 ('雨','云中的水滴逐渐增大，降落到地面就形成了雨。'),
 ('学习','学习时可以先明确一个小目标，再集中注意力完成它。'),
 ('休息','累了可以暂时放下工作，喝点水，并适当休息。'),
 ('阅读','可以每天安排一小段固定时间，先从感兴趣的书开始。'),
 ('整理','可以先清理不需要的东西，再把同类物品放在一起。'),
 ('感谢','得到帮助后，可以真诚地向对方说谢谢。'),
 ('睡眠','保持规律的睡眠时间，有助于休息和恢复精力。'),
]

def authored_pairs():
    result=[]
    # 600 个算术样本，答案由整数运算核查，不声称为人工逐题创作。
    for a in range(1,21):
        for b in range(1,11):
            for op,value in [('+',a+b),('-',a-b),('乘',a*b)]:
                result.append(dict(user=f'请计算{a}{op}{b}，告诉我结果。',assistant=f'{a}{op}{b}={value}。',source='authored-arithmetic'))
    for topic,answer in FACTS:
        for wording in ['请介绍{}。','我想了解{}，请简单解释。','用简短的话说说{}。','关于{}，你能解释一下吗？','简单讲讲{}是什么。','谈谈{}的基本作用。','初学者怎样理解{}？','解释一下{}这个概念。','我不太懂{}，请说明。','说说{}，不要太长。','请用一句话介绍{}。','给我一个关于{}的简单说明。','{}是怎么回事？','我想知道{}的含义。','请讲一下{}的基础知识。']:
            result.append(dict(user=wording.format(topic),assistant=answer,source='authored-facts'))
    examples=[('你好啊','你好！很高兴和你聊天。'),('早安朋友','早安！祝你今天顺利。'),('晚安朋友','晚安，祝你睡个好觉。'),('谢谢你呀','不客气，很高兴能帮到你。'),('再会','再见！有需要随时来聊。'),('你是哪位助手','我是LubanLM，一个中文助手，可以和你聊天并回答简单问题。'),('你有名字么','我的名字是LubanLM。'),('你能走路么','我没有身体，不能走路。我通过文字和你交流。'),('你是一个人么','我是一个语言模型，不是真人。'),('陪我说一句话','当然可以。今天过得怎么样？')]
    for q,a in examples:
        for prefix in ['','朋友，','请问，','我想问一下，','现在，','助手，','小助手，','打扰一下，','可以问你吗，','和你聊聊，']:
            result.append(dict(user=prefix+q,assistant=a,source='authored-chat'))
    assert len(result)==1000
    return result

def prepare_sft(legacy,tokenizer,limit):
    manifest_path=RUNS/'reports/sft-v2.json'
    if manifest_path.exists():
        return read_json(manifest_path)
    blocked=excluded_questions()
    seen={}
    rows={'train':[],'validation':[]}
    counters={'excluded_suite':0,'duplicate':0,'length':0,'filter':0}
    def add(row,force_split=None):
        q,a=clean(row['user']),clean(row['assistant'])
        key=normalize(q)
        if key in blocked:
            counters['excluded_suite']+=1
            return False
        if not key or key in seen:
            counters['duplicate']+=1
            return False
        if any(t in q+a for t in ('<bos>','<eos>','<pad>')):
            return False
        pair=encode_pair(tokenizer,q,a,512)
        if pair is None:
            counters['length']+=1
            return False
        split=force_split or ('validation' if int(digest(key)[:8],16)%100<2 else 'train')
        seen[key]=split
        record=dict(row,user=q,assistant=a,question_sha256=digest(key),split=split)
        rows[split].append(dict(pair,question_sha256=digest(key),source=row['source'],tree_id=row.get('tree_id',digest(key)),message_id=row.get('message_id',digest(q+a))))
        return record
    raw_out={'train':[],'validation':[]}
    # 验证优先去重，保留 OASST 原始树划分。
    for split in ('validation','train'):
        for r in records(legacy/f'processed/sft-{split}.jsonl'):
            r['source']='oasst1'
            result=add(r,split)
            if result:
                raw_out[split].append(result)
    for r in authored_pairs():
        result=add(r)
        if result:
            raw_out[result['split']].append(result)
    repo='BelleGroup/train_0.5M_CN'
    source_path=DATA/'raw/belle.source.json'
    if source_path.exists():
        source=read_json(source_path)
    else:
        info=HfApi().dataset_info(repo)
        source=dict(repo=repo,revision=info.sha,license='see dataset card: GPL-3.0 and additional restrictions',synthetic=True)
        write_json(source_path,source)
    accepted=0
    with (DATA/'raw/belle-selected.jsonl').open('w',encoding='utf-8') as raw:
        for index,r in enumerate(load_dataset(repo,revision=source['revision'],split='train',streaming=True)):
            q=clean(str(r.get('instruction',''))+'\n'+str(r.get('input','')))
            a=clean(str(r.get('output','')))
            if not 2<=len(q)<=350 or not 2<=len(a)<=280 or len(re.findall(r'[\u4e00-\u9fff]',a))<max(2,len(a)*.2) or re.search(r'(.{3,30}?)\1\1',a):
                counters['filter']+=1
                continue
            # 简单四则运算回答可以自动核查；发现矛盾即剔除。
            arithmetic=re.search(r'(-?\d+)\s*([+\-×*])\s*(-?\d+)\s*=\s*(-?\d+)',a)
            if arithmetic:
                x,op,y,z=arithmetic.groups(); x,y,z=int(x),int(y),int(z)
                expected=x+y if op=='+' else x-y if op=='-' else x*y
                if z!=expected:
                    counters['filter']+=1
                    continue
            result=add(dict(user=q,assistant=a,source='belle-synthetic',source_index=index))
            if result:
                accepted+=1
                raw_out[result['split']].append(result)
                raw.write(json.dumps(result,ensure_ascii=False)+'\n')
            if accepted and accepted%5000==0:
                print('belle',accepted,flush=True)
                write_json(RUNS/'reports/preparation_progress.json',dict(stage='belle',accepted=accepted,updated_utc=utc()))
            if accepted>=limit:
                break
    for split in rows:
        save_jsonl(DATA/f'processed/sft-{split}-tokens.jsonl',rows[split])
        save_jsonl(DATA/f'processed/sft-{split}.jsonl',raw_out[split])
    train_keys={r['question_sha256'] for r in rows['train']}
    val_keys={r['question_sha256'] for r in rows['validation']}
    assert not train_keys & val_keys
    assert not {normalize(r['user']) for split in raw_out for r in raw_out[split]} & blocked
    result=dict(source=source,belle_accepted=accepted,authored_generated=1000,authored_validation='template facts reviewed; arithmetic computed exactly',statistics={s:dict(examples=len(rs),input_tokens=sum(len(r['input_ids']) for r in rs),supervised_tokens=sum(r['assistant_tokens'] for r in rs)) for s,rs in rows.items()},rejected=counters,question_split_disjoint=True,suite_exact_questions_excluded=True,created_utc=utc())
    write_json(manifest_path,result)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--legacy-data',type=Path,default=Path('D:/LubanLM-Data'))
    p.add_argument('--documents',type=int,default=250000)
    p.add_argument('--belle-pairs',type=int,default=50000)
    args=p.parse_args()
    if DATA.resolve()==args.legacy_data.resolve():
        p.error('use a separate LUBAN_DATA directory; preserve v1')
    (DATA/'raw').mkdir(parents=True,exist_ok=True)
    (DATA/'processed').mkdir(parents=True,exist_ok=True)
    target=DATA/'tokenizers/vocab-32000'
    if not target.exists():
        shutil.copytree(args.legacy_data/'tokenizers/vocab-32000',target)
    assert sha256(target/'tokenizer.json')==sha256(args.legacy_data/'tokenizers/vocab-32000/tokenizer.json')
    tokenizer=Tokenizer.from_file(str(target/'tokenizer.json'))
    wiki=prepare_wiki(args.legacy_data,tokenizer,args.documents)
    sft=prepare_sft(args.legacy_data,tokenizer,args.belle_pairs)
    write_json(RUNS/'reports/data-v2.json',dict(wiki=wiki,sft=sft,tokenizer_sha256=sha256(target/'tokenizer.json'),created_utc=utc()))
    print('DATA READY',flush=True)

if __name__=='__main__':
    main()
