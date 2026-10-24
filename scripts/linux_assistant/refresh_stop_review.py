"""Carry a documented factual review only when native-stop content is unchanged.

Strict EXACT/JSON conditions are recalculated on the actual assistant prefix.
Changed substantive answers always require a new explicit review.
"""
import argparse
import hashlib
from pathlib import Path
from scripts.common import sha256
from scripts.linux_assistant.common import jsonl,write_jsonl,screen


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old-answers',type=Path,required=True);p.add_argument('--old-decisions',type=Path,required=True)
    p.add_argument('--answers',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();old={r['id']:r for r in jsonl(a.old_answers)};decisions={r['id']:r for r in jsonl(a.old_decisions)}
    results=[]
    for row in jsonl(a.answers):
        before=old[row['id']];decision=decisions[row['id']]
        if before['answer'].strip()!=row['answer'].strip():raise ValueError('Substantive answer changed; re-review '+row['id'])
        if hashlib.sha256(before['answer'].encode()).hexdigest()!=decision['answer_sha256']:raise ValueError('Old review does not bind old answer')
        accepted=decision['accepted']
        if row['kind']=='general' and row['rubric'].startswith(('EXACT:','JSON:')):
            accepted=decision['content_correct'] and screen(row,row['answer'])['screen_pass']
        results.append({**decision,'answer_sha256':hashlib.sha256(row['answer'].encode()).hexdigest(),
            'accepted':accepted,'reason':'原生助手结束前的语义与旧轨迹一致；沿用已记录事实审阅，严格格式重新核对。'+decision['reason'],
            'previous_decisions_sha256':sha256(a.old_decisions)})
    write_jsonl(a.output,results)


if __name__=='__main__':main()
