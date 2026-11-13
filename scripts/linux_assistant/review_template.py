"""Reuse documented reviews only for exactly identical questions and raw answers.

New answers remain undecided. This is evidence reuse, not automatic semantic grading.
"""
import argparse
import hashlib
import json
from pathlib import Path
from scripts.common import sha256
from scripts.linux_assistant.common import config,DEFAULT_CONFIG,jsonl,write_jsonl,fingerprint,reviewed_scores


def key(row):
    return fingerprint({name:row.get(name) for name in ('id','messages','reference','rubric','grading','kind','category','answer')})


def template(rows,sources):
    index={};conflicts=set()
    for path,digest,previous in sources:
        for row in previous:
            signature=key(row);verdict=(row['accepted'],row.get('content_correct'),fingerprint(row.get('axes')))
            if signature in index and index[signature][3]!=verdict:conflicts.add(signature)
            index[signature]=(path,digest,row,verdict)
    decisions=[]
    for row in rows:
        signature=key(row)
        decision={'id':row['id'],'answer_sha256':hashlib.sha256(row['answer'].encode()).hexdigest(),
                  'accepted':None,'reason':'','reviewer':''}
        if row['kind']=='general':decision['content_correct']=None
        if 'grading' in row:decision['axes']={k:None for k in ('content_correct','steps_complete','format_compliant')}
        if signature in index and signature not in conflicts:
            path,digest,source,_=index[signature]
            decision.update(accepted=source['accepted'],reviewer=source['reviewer'],
                reason='同一问题、必要条件与原始回答逐字相同，沿用已有记录：'+source['review_reason'],
                equivalents=source.get('equivalents',{}),reuse={'file':str(path),'sha256':digest})
            if row['kind']=='general':decision['content_correct']=source['content_correct']
            if 'grading' in row:decision['axes']=dict(source['axes'])
        decisions.append(decision)
    return decisions


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    p.add_argument('--answers',type=Path,required=True);p.add_argument('--output',type=Path)
    a=p.parse_args();c=config(a.config);output=a.output or a.answers.with_name(a.answers.stem+'.decisions.draft.jsonl')
    if output.exists():raise FileExistsError('Do not overwrite an existing review draft')
    sources=[]
    for path in [*c['report'].glob('qwen*-dev.reviewed.jsonl'),*c['runs'].glob('lr-*/dev-*.reviewed.jsonl')]:
        rows=jsonl(path);reviewed_scores(rows)
        if not rows:raise ValueError('Empty review source')
        sources.append((path.resolve(),sha256(path),rows))
    decisions=template(jsonl(a.answers),sources);write_jsonl(output,decisions)
    print(json.dumps({'reused':sum('reuse' in d for d in decisions),'manual_review_pending':sum(d['accepted'] is None for d in decisions),'output':str(output)}))


if __name__=='__main__':main()
