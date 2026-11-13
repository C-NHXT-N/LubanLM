"""Build frozen authored task data, provenance ledger and explicit variant counts."""
from __future__ import annotations
import argparse
from collections import Counter
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
from scripts.linux_assistant.catalog import cases, SOURCES
from scripts.linux_assistant.common import config, DEFAULT_CONFIG, write_jsonl, fingerprint, screen
from scripts.common import ROOT, sha256, write_json

PREFIXES = ['', '请给出准确而简洁的解释：', '我在整理部署笔记，请回答：',
            '面向Linux初学者回答：', '请说明操作依据：', '请帮我确认：',
            '我需要核对这一步：', '请指出关键检查项：', '请按实际可确认的信息回答：',
            '请帮我复核下面的问题：']


def general(split):
    """Eight independently authored families per split, five concrete tasks per family."""
    result = []
    dev = split == 'dev'
    for i in range(40):
        group, variant = divmod(i, 5)
        if group == 0:
            a, b = (23, 18) if dev else (47, 26)
            a += variant * 3
            q, answer, rubric = f'只输出{a}+{b}的结果。', str(a+b), f'EXACT:{a+b}'
        elif group == 1:
            a, b = (8, 7) if dev else (12, 6)
            b += variant
            q, answer, rubric = f'只输出{a}×{b}的结果。', str(a*b), f'EXACT:{a*b}'
        elif group == 2:
            words = ['河流','橘子','雨伞','灯塔','茶杯'] if dev else ['海湾','竹林','石桥','麦田','书签']
            answer = words[variant]
            q, rubric = f'只输出“{answer}”两个字，不加引号和解释。', f'EXACT:{answer}'
        elif group == 3:
            n = (10 if dev else 30) + variant
            q = f'只输出JSON对象，字段count为{n}，ready为true。'
            answer, rubric = json.dumps({'count':n,'ready':True}, ensure_ascii=False), f'JSON:count={n},ready=true'
        elif group == 4:
            names = ['小林','小周','小吴'] if dev else ['小赵','小陈','小孙']
            q = f'{names[0]}比{names[1]}高，{names[1]}比{names[2]}高。只输出最高者的名字。'
            # Five wording variants share one family and are explicitly counted as such.
            q = PREFIXES[variant] + q
            answer, rubric = names[0], f'EXACT:{names[0]}'
        elif group == 5:
            items = [('水结冰通常属于什么物态变化？','凝固'),('冰变成液态水叫什么？','熔化'),
                     ('液态水变成水蒸气叫什么？','汽化'),('水蒸气变成小水滴叫什么？','液化'),
                     ('光合作用通常需要哪种气体？','二氧化碳')] if dev else [
                     ('植物根部主要吸收什么液体？','水'),('地球绕太阳运动叫什么？','公转'),
                     ('地球绕自身轴运动叫什么？','自转'),('声音在真空中可以传播吗？','不能'),
                     ('磁铁通常可以吸引铁制品吗？','可以')]
            q, answer = items[variant]; rubric = re.escape(answer)
        elif group == 6:
            n = (6 if dev else 15) + variant
            q = f'有{n}盒彩笔，每盒4支。只输出彩笔总支数。'
            answer, rubric = str(n*4), f'EXACT:{n*4}'
        else:
            n = (40 if dev else 90) + variant
            q = f'只输出{n}减去17的结果。'
            answer, rubric = str(n-17), f'EXACT:{n-17}'
        result.append({'id':f'general-{split}-{i:03d}', 'group':f'general/{split}/{group}',
                       'kind':'general','category':'general','messages':[{'role':'user','content':q}],
                       'reference':answer,'rubric':rubric,'sources':['authored elementary task'],
                       'facts':[answer], 'source_review':'elementary fact checked at authoring'})
    return result


def main():
    raise RuntimeError('Legacy authored-data generator is retired. Build only from pinned, independently reviewed source candidates with the rebuild workflow.')
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    a = p.parse_args(); c = config(a.config)
    out, report = c['data'], c['report']; out.mkdir(parents=True, exist_ok=True); report.mkdir(parents=True, exist_ok=True)
    if (out/'manifest.json').exists():
        raise FileExistsError('Frozen dataset exists; use a new data version rather than overwrite')
    seeds = cases()
    expected = Counter({(s,k):n for s,n in [('train',10),('dev',4),('test',4)]
                        for k in ['environment','linux','deployment','troubleshooting','interaction']})
    actual = Counter((r['split'],r['category']) for r in seeds)
    if actual != expected:
        raise ValueError(f'Unexpected seed counts {actual}')
    groups = [r['group'] for r in seeds]
    if len(groups) != len(set(groups)): raise ValueError('Duplicate scenario group')
    ledger, data, warnings = [], {s:[] for s in ['train','dev','test']}, []
    for seed in seeds:
        for key in seed['source_keys']:
            if key not in SOURCES: raise ValueError(key)
        # Required-condition rules must at least recognize the independently authored reference.
        if not screen(seed, seed['answer'])['screen_pass']:
            raise ValueError(f'Reference fails its own rubric: {seed["group"]}')
        ledger.append({'group':seed['group'],'split':seed['split'],'question':seed['question'],
                       'reference':seed['answer'],'source_urls':[SOURCES[k] for k in seed['source_keys']],
                       'review_status':'pending_source_verification','reviewer':None,'evidence':None})
        for v in range(10 if seed['split']=='train' else 5):
            exact = seed['rubric'].startswith(('EXACT:', 'JSON:'))
            question = seed['question'] if v == 0 else PREFIXES[v] + seed['question']
            messages = [{'role':'user','content':question}]
            if seed['split'] == 'train' and v < 3:
                messages = [{'role':'user','content':'我在学习Linux与开发板部署，请用中文帮助我核对问题。'},
                            {'role':'assistant','content':'请给出具体问题及已知条件。'}, *messages]
            row = {'id':seed['group'].replace('/','-')+f'-{v:02d}', 'group':seed['group'],
                   'kind':'task','category':seed['category'],'messages':messages,
                   'reference':seed['answer'],'rubric':seed['rubric'],
                   'facts':[seed['answer']],'sources':[SOURCES[k] for k in seed['source_keys']],
                   'variant':v,'source_review':'see source-review.jsonl'}
            if seed['split']=='train': messages.append({'role':'assistant','content':seed['answer']})
            data[seed['split']].append(row)
    for i, left in enumerate(seeds):
        for right in seeds[i+1:]:
            if left['split'] == right['split']: continue
            ratio = SequenceMatcher(None,left['question'],right['question']).ratio()
            if ratio >= .62: warnings.append({'left':left['group'],'right':right['group'],'similarity':ratio,'status':'needs_review'})
    for split, rows in data.items(): write_jsonl(out/f'{split}.jsonl',rows)
    for split in ('dev','test'): write_jsonl(out/f'general-{split}.jsonl',general(split))
    write_jsonl(out/'source-review.jsonl',ledger)
    write_json(report/'data-audit.json',{'counts':{s:len(v) for s,v in data.items()},
        'independent_scenarios':dict(Counter(r['split'] for r in seeds)),
        'training_multiturn':sum(len(r['messages'])>2 for r in data['train']),
        'group_overlap':False,'near_duplicate_warnings':warnings,
        'source_review':'pending','semantic_overlap_review':'pending',
        'warning':'500 train records are 10 wording variants of 50 authored scenarios, not 500 independent facts.'})
    names=['train.jsonl','dev.jsonl','test.jsonl','general-dev.jsonl','general-test.jsonl']
    write_json(out/'manifest.json',{'version':out.name,'files':{n:sha256(out/n) for n in names},
        'seed_catalog_sha256':sha256(Path(__file__).with_name('catalog.py')),
        'source_snapshots':{'project_status':sha256(ROOT/'PROJECT_STATUS.md')},
        'independent_scenarios':{'train':50,'dev':20,'test':20},
        'final_test_policy':'Do not evaluate before candidate lock; no tuning against final outputs',
        'source_review_file':'source-review.jsonl','source_review_required':True})
    print(json.dumps({'dataset':str(out),'counts':{s:len(v) for s,v in data.items()},'review_pending':len(ledger)},ensure_ascii=False))


if __name__ == '__main__': main()
