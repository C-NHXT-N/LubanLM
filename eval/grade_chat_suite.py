"""汇总逐题人工/审阅者评分；缺评分或不完整题目必须拒绝验收。"""
from __future__ import annotations
from scripts.common import write_json, read_json, sha256, utc
from eval.chat_suite import CATEGORIES, suite, load_suite
from pathlib import Path
import argparse

def grade(raw_path,marks_path):
    raw=read_json(raw_path)
    marks=read_json(marks_path)['marks']
    expected={(r['id'],r['seed']) for r in raw['results']}
    provided={(r['id'],r['seed']) for r in marks}
    if expected!=provided or len(marks)!=len(expected):
        raise ValueError('each raw answer needs exactly one grade')
    target=60 if raw['split'] in ('dev','test','sampling') else 5
    if len(expected)!=target:
        raise ValueError('suite incomplete')
    custom=raw.get('suite_file')
    if custom and sha256(custom)!=raw.get('suite_definition_sha256'):
        raise ValueError('frozen suite definition hash changed')
    definition=load_suite(custom,raw['split']) if custom else suite(raw['split'])
    if raw['split'] in ('dev','test') and {(r['id'],r['question'],r['category']) for r in raw['results']}!={(r['id'],r['question'],r['category']) for r in definition}:
        raise ValueError('suite IDs differ from frozen suite')
    lookup={(r['id'],r['seed']):r for r in marks}
    scores=[]
    for row in raw['results']:
        mark=lookup[(row['id'],row['seed'])]
        if any(type(mark.get(key)) is not bool for key in ('relevant_readable','correct','looping')) or not str(mark.get('reason','')).strip():
            raise ValueError('explicit booleans and reason required for every answer')
        scores.append(dict(row,grading=mark,qualified=mark['relevant_readable'] and mark['correct'] and not mark['looping']))
    n=len(scores)
    rate=sum(r['qualified'] for r in scores)/n
    readable=sum(r['grading']['relevant_readable'] for r in scores)/n
    loops=sum(r['grading']['looping'] for r in scores)/n
    categories={c:sum(r['qualified'] for r in scores if r['category']==c)/sum(r['category']==c for r in scores) for c in CATEGORIES if any(r['category']==c for r in scores)}
    if raw['split'] in ('dev','test'):
        passed=rate>=.8 and readable>=.9 and loops<=.05 and min(categories.values())>=.6 and categories['arithmetic']>=.8
    elif raw['split']=='core':
        passed=rate==1.
    else:
        passed=readable>=.8 and loops<=.05
    result=dict(answers_sha256=sha256(raw_path),marks_sha256=sha256(marks_path),review_complete=True,split=raw['split'],qualified_rate=rate,relevant_readable_rate=readable,looping_rate=loops,eos_rate=sum(r['eos'] for r in scores)/n,category_rates=categories,passed=passed,development_ready=passed if raw['split']=='dev' else None,created_utc=utc(),results=scores)
    output=Path(raw_path).with_suffix('.review.json')
    write_json(output,result)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--answers',type=Path,required=True)
    p.add_argument('--marks',type=Path,required=True)
    args=p.parse_args()
    result=grade(args.answers,args.marks)
    print({k:v for k,v in result.items() if k!='results'})

if __name__=='__main__':
    main()
