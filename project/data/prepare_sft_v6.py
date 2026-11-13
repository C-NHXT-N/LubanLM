"""仅重新编码既有v5语义数据并隔离模板/负载家族，不读取任何验收输出。"""
from scripts.common import DATA,RUNS,ROOT,read_json,write_json,sha256,utc
from data.prepare import load_jsonl,save_jsonl
from data.prepare_chat_v2 import FACTS
from eval.chat_suite import normalize,excluded_questions
from sft.template import encode_pair,prompt,PREFIX_ENCODING
from tokenizers import Tokenizer
from collections import Counter
import hashlib
import re

def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def task_for(row):
    source,answer=row['source'],row['assistant']
    if not source.startswith('authored-'):return 'public'
    if 'arithmetic' in source:return 'arithmetic'
    if 'identity' in source:return 'identity'
    if 'daily' in source:return 'daily'
    if any(s in source for s in ('copy','keyword','naming','instruction')):return 'instruction'
    if 'chat' in source:
        return 'identity' if any(s in answer for s in ('LubanLM','语言模型','身体','基础知识','中文')) else 'greeting'
    daily_answers={a for _,a in FACTS[14:]}
    if answer in daily_answers:return 'daily'
    if source=='authored-basic-v5':
        return 'concept' if answer.startswith('冰块') else 'greeting' if answer.startswith('先友善') else 'daily'
    return 'concept'

def family_for(row):
    if 'arithmetic_group' in row:return 'arithmetic:'+row['arithmetic_group']
    if 'copy' in row['source'] or ('抄' in row['user'] or '原样' in row['user']):
        return 'copy-payload:'+digest(normalize(row['assistant']))
    # 前缀变体全部归同一问法家族，不把加一个请问当作独立验证问题。
    q=row['user']
    prefixes=('请问，','朋友，','助手，','我想了解一下，','打扰一下，','我想问一下，','现在，','小助手，','可以问你吗，','和你聊聊，','给我一个小建议：')
    while any(q.startswith(p) for p in prefixes):
        q=next(q[len(p):] for p in prefixes if q.startswith(p))
    return 'question-family:'+digest(normalize(q))

def main():
    target=DATA/'processed/sft-v6'
    if target.exists():raise ValueError('v6 already exists; preserve immutable data')
    parent=DATA/'processed/sft-v5'
    tok=Tokenizer.from_file(str(DATA/'tokenizers/vocab-32000/tokenizer.json'))
    blocked=excluded_questions()
    fresh=ROOT/'eval/suites/chat-v6.json'
    if fresh.exists():
        for values in read_json(fresh).values():
            if isinstance(values,list):blocked.update(normalize(r['question']) for r in values)
    originals={'train':load_jsonl(parent/'train-unique.jsonl'),'validation':load_jsonl(parent/'validation.jsonl')}
    valid_families={family_for(r) for r in originals['validation']}
    rows={'train':[],'validation':[]};rejected=Counter();mismatches=0
    for old_split in ('validation','train'):
        for r in originals[old_split]:
            if normalize(r['user']) in blocked:
                rejected['frozen_suite_question']+=1;continue
            # 单轮任务不能把内嵌角色模板或特殊控制符误当普通指令。
            if any(t in r['user']+r['assistant'] for t in ('<bos>','<eos>','<pad>','系统：','用户：','助手：')):
                rejected['embedded_control_or_role']+=1;continue
            pair=encode_pair(tok,r['user'],r['assistant'],512,encoding=PREFIX_ENCODING)
            if pair is None:rejected['length_or_empty']+=1;continue
            legacy=encode_pair(tok,r['user'],r['assistant'],512)
            prefix=tok.encode(prompt(r['user']),add_special_tokens=False).ids
            mismatches+=int(legacy is not None and legacy['input_ids'][:len(prefix)]!=prefix)
            assert pair['input_ids'][:len(prefix)]==prefix
            assert pair['labels'][:len(prefix)]==[-100]*len(prefix)
            assert pair['labels'][len(prefix):]==pair['input_ids'][len(prefix):]
            assert tok.decode(pair['input_ids'][len(prefix):-1],skip_special_tokens=False)==r['assistant']
            family=family_for(r)
            split='validation' if old_split=='validation' or family in valid_families or (r['source'].startswith('authored-') and int(digest(family)[:8],16)%100<5) else 'train'
            rows[split].append(dict(r,**pair,task=task_for(r),family_group=family,split=split))
    assert not {r['family_group'] for r in rows['train']} & {r['family_group'] for r in rows['validation']}
    assert not {normalize(r['user']) for r in rows['train']} & {normalize(r['user']) for r in rows['validation']}
    assert not {r.get('tree_id') for r in rows['train'] if r['source']=='oasst1'} & {r.get('tree_id') for r in rows['validation'] if r['source']=='oasst1'}
    assert set(r['task'] for r in rows['train'])=={'public','greeting','identity','daily','concept','instruction','arithmetic'}
    target.mkdir(parents=True)
    for split in rows:
        save_jsonl(target/f'{split}.jsonl',rows[split])
        save_jsonl(target/f'{split}-tokens.jsonl',[{k:r[k] for k in ('input_ids','labels','assistant_tokens','encoding_version','prompt_tokens','source','task','family_group','question_sha256')} for r in rows[split]])
    manifest=dict(created_utc=utc(),encoding_version=PREFIX_ENCODING,parent_manifest_sha256=sha256(parent/'manifest.json'),
                  tokenizer_sha256=sha256(DATA/'tokenizers/vocab-32000/tokenizer.json'),
                  counts={s:len(v) for s,v in rows.items()},tasks={s:dict(Counter(r['task'] for r in v)) for s,v in rows.items()},
                  legacy_prefix_mismatches=mismatches,rejected=dict(rejected),prompt_prefix_verified_all=True,
                  question_and_family_disjoint=True,oasst_tree_disjoint=True,
                  train_sha256=sha256(target/'train-tokens.jsonl'),validation_sha256=sha256(target/'validation-tokens.jsonl'),
                  policy='Existing v5 semantic pairs only; no final outputs read; unique rows; split by copy payload and question family. Some held-out families were seen by ancestor checkpoints: this validation is not an independent capability test. Public data remains heuristic, not fully fact-certified.')
    write_json(target/'manifest.json',manifest);write_json(RUNS/'reports/sft-v6.json',manifest)
    print(manifest)

if __name__=='__main__':main()
