"""Attach explicit documented reviews; never turn keyword-screen passes into semantic passes."""
import argparse
import json
from pathlib import Path
from scripts.common import write_json,sha256
from scripts.linux_assistant.common import config,DEFAULT_CONFIG,jsonl,write_jsonl,reviewed_scores,screen


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    p.add_argument('--answers',type=Path);p.add_argument('--decisions',type=Path)
    p.add_argument('--baseline',action='store_true');a=p.parse_args();c=config(a.config)
    if a.baseline:
        models={}
        for name in ('qwen35','qwen3'):
            path=c['report']/f'{name}-dev.reviewed.jsonl';rows=jsonl(path)
            expected=jsonl(c['data']/'dev.jsonl')+jsonl(c['data']/'general-dev.jsonl')
            if [r['id'] for r in rows]!=[r['id'] for r in expected]:raise ValueError('Baseline requires all current development answers in order')
            models[name]={'scores':reviewed_scores(rows),'sha256':sha256(path),'file':str(path)}
        write_json(c['report']/'baseline-reviewed.json',{'completed':True,'models':models})
        return
    if not a.answers or not a.decisions:raise ValueError('Both --answers and --decisions are required')
    rows=jsonl(a.answers);decisions=jsonl(a.decisions)
    index={r['id']:r for r in decisions}
    if len(index)!=len(decisions) or set(index)!={r['id'] for r in rows}:raise ValueError('Review IDs must exactly cover outputs')
    reused_sources={}
    for row in rows:
        row.update(screen(row,row['answer']))
        d=index[row['id']]
        if d.get('reuse'):
            from scripts.linux_assistant.review_template import key
            source=Path(d['reuse']['file'])
            if str(source) not in reused_sources:
                if sha256(source)!=d['reuse']['sha256']:raise ValueError('Reused review source changed')
                reused_sources[str(source)]={key(r):r for r in jsonl(source)}
            prior=reused_sources[str(source)].get(key(row))
            if prior is None or prior['accepted']!=d['accepted'] or (row['kind']=='general' and prior['content_correct']!=d.get('content_correct')):
                raise ValueError('Review reuse requires identical question, answer and verdict')
            row['review_reused_from']=d['reuse']
        if d.get('answer_sha256')!=__import__('hashlib').sha256(row['answer'].encode()).hexdigest():
            raise ValueError('Decision does not bind the actual answer: '+row['id'])
        if type(d.get('accepted')) is not bool or not d.get('reason') or not d.get('reviewer'):
            raise ValueError('Missing explicit semantic decision/evidence/reviewer')
        row.update(accepted=d['accepted'],review_reason=d['reason'],reviewer=d['reviewer'],semantic_review='complete')
        if 'grading' in row:
            from scripts.linux_assistant.semantic_grading import validate_model_review
            row['axes']=validate_model_review(row,d)
            for axis in ('content_correct','steps_complete','format_compliant'):row[axis]=row['axes'][axis]
        row['equivalents']=d.get('equivalents',{})
        if row['kind']=='general':
            if type(d.get('content_correct')) is not bool:raise ValueError('Missing separate general content decision: '+row['id'])
            if 'grading' in row and d['content_correct']!=row['axes']['content_correct']:
                raise ValueError('General content decision conflicts with semantic axis: '+row['id'])
            row['content_correct']=d['content_correct']
    output=a.answers.with_name(a.answers.stem+'.reviewed.jsonl')
    write_jsonl(output,rows);write_json(output.with_suffix('.scores.json'),reviewed_scores(rows))


if __name__=='__main__':main()
