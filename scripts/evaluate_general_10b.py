"""Frozen final responses and mechanical flags; semantic acceptance stays pending."""
from scripts.common import ROOT,read_json,write_json,sha256,utc,GPUBudget,require_gpu
from eval.chat_suite import repeated
from sft.template import SYSTEM,prompt
from inference.generate import load_model,complete
from training.execution import configure_model
import argparse
import torch

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();suite=ROOT/'docs/run-results/general-10b-20261005/frozen-test.json'
    result=[]
    with GPUBudget('general-10b-final-evaluation',30) as budget:
        model,tokenizer=load_model(args.checkpoint,require_gpu());configure_model(model,'sdpa')
        for row in read_json(suite)['questions']:
            if budget.remaining()<10:raise RuntimeError('Incomplete final evaluation')
            text=prompt(row['question'])
            if row.get('messages'):
                text=f'<bos>系统：{SYSTEM}\n'
                for m in row['messages']:
                    text+=('用户：' if m['role']=='user' else '助手：')+m['content']+('\n' if m['role']=='user' else '<eos>\n')
                text+='助手：'
            answer,ids=complete(model,tokenizer,text,max_new_tokens=128,temperature=0,repetition_penalty=1)
            result.append(dict(row,answer=answer,eos=bool(ids and ids[-1]==2),replacement_character='\ufffd' in answer,repeated=repeated(answer),semantic_correct=None))
            budget.tick()
    write_json(args.output,dict(created_utc=utc(),checkpoint=args.checkpoint,checkpoint_sha256=sha256(args.checkpoint),suite_sha256=sha256(suite),status='semantic_review_pending',results=result,mechanical=dict(examples=len(result),eos_rate=sum(r['eos'] for r in result)/len(result),replacement_count=sum(r['replacement_character'] for r in result),repeated_count=sum(r['repeated'] for r in result))))

if __name__=='__main__':main()
