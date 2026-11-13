"""Ask the archived 50M variants the same small, exploratory Chinese question set."""
from __future__ import annotations
import gc
import json
import os
from pathlib import Path
import time

from scripts.common import WORKSPACE, DATA, require_gpu, sha256, write_json
from sft.template import prompt

QUESTIONS = [
    ('intro', '你好，请介绍一下你自己。'),
    ('capital', '中国的首都是哪里？'),
    ('addition-small', '1+2等于多少？'),
    ('addition', '17+25等于多少？'),
    ('science', '请用一句话解释什么是蒸发。'),
    ('linux', 'Linux是什么？'),
    ('instruction', '请只输出“苹果”两个字，不要解释。'),
    ('reasoning', '小明比小红高，小红比小刚高。三个人中谁最高？'),
]


def main():
    import torch
    from inference.generate import load_model, complete
    require_gpu()
    torch.set_num_threads(4)
    torch.manual_seed(20261018)
    report = WORKSPACE/'reports'/time.strftime('50m-question-probe-%Y%m%d-%H%M%S')
    report.mkdir(parents=True)
    checkpoints = [
        ('Base', WORKSPACE/'runs/pretrain/checkpoints/last.pt'),
        ('SFT best', WORKSPACE/'runs/sft/checkpoints/best.pt'),
        ('SFT last', WORKSPACE/'runs/sft/checkpoints/last.pt'),
    ]
    protected = [p for _,p in checkpoints]+[DATA/'tokenizers/vocab-32000/tokenizer.json']
    before={str(p):sha256(p) for p in protected}
    result={
        'completed':False,'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,
        'settings':{'temperature':0.0,'top_k':0,'top_p':1.0,'repetition_penalty':1.0,
                    'max_new_tokens':96,'use_cache':True,'precision':'BF16 autocast','seed':20261018},
        'scope':'Exploratory probe, not an independent benchmark; training overlap not screened. Same native chat template for every model, including Base.',
        'protected_sha256_before':before,'rows':[],
    }
    try:
        for label,path in checkpoints:
            model,tokenizer=load_model(path,device='cuda:0')
            for key,question in QUESTIONS:
                text=prompt(question)
                started=time.monotonic()
                output,ids=complete(model,tokenizer,text,max_new_tokens=96,temperature=0.0,
                                    top_k=0,top_p=1.0,repetition_penalty=1.0,use_cache=True)
                torch.cuda.synchronize()
                row={'model':label,'checkpoint':str(path),'question_id':key,'question':question,
                     'prompt':text,'output':output,'generated_ids':ids,'tokens':len(ids),
                     'stop_reason':'eos' if ids and ids[-1]==model.config.eos_token_id else 'length_limit',
                     'seconds':round(time.monotonic()-started,3)}
                result['rows'].append(row)
                write_json(report/'outputs.json',result)
                print(json.dumps({k:v for k,v in row.items() if k not in ('generated_ids','prompt','checkpoint')},ensure_ascii=False),flush=True)
            del model,tokenizer
            gc.collect();torch.cuda.empty_cache()
        result['completed']=True
    finally:
        result['protected_sha256_after']={str(p):sha256(p) for p in protected}
        result['protected_files_unchanged']=result['protected_sha256_after']==before
        write_json(report/'outputs.json',result)
        lines=['# 50M八题实问：原始生成结果','',
               '相同原生聊天模板，greedy，BF16，每题最多96个新Token，不增加重复惩罚。Base不具备SFT对话训练，本次同模板结果仅作对照。题目未排查与训练集重合，不能当作独立能力验收。','']
        for key,question in QUESTIONS:
            lines.extend([f'## {question}',''])
            for row in [r for r in result['rows'] if r['question_id']==key]:
                lines.extend([f"### {row['model']}",'','```text',row['output'],'```','',
                              f"停止：{row['stop_reason']}；生成Token：{row['tokens']}。",''])
        (report/'RAW_ANSWERS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        print(f'Report: {report}',flush=True)
    if not result['protected_files_unchanged']:
        raise RuntimeError('Protected model or tokenizer changed')


if __name__=='__main__':
    main()
