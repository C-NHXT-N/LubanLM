"""Offline, bounded Chinese Q&A for two pinned Qwen models; no training or conversion."""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import time

from scripts.common import WORKSPACE, require_gpu, write_json
from scripts.probe_50m_questions import QUESTIONS


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--model',choices=['both','qwen3','qwen35'],default='both')
    args=parser.parse_args();report=args.report.resolve();report.mkdir(parents=True,exist_ok=True)
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForImageTextToText, GenerationConfig
    require_gpu();torch.set_num_threads(4);torch.manual_seed(20261018)
    all_models=[('qwen3','Qwen3-4B-Instruct-2507'),('qwen35','Qwen3.5-2B')]
    targets=[m for m in all_models if args.model=='both' or args.model==m[0]]
    system='你是一个中文助手。请简洁回答，尽量不超过三句话；用户指定输出格式时优先遵守。'
    result={'completed':False,'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,
            'transformers':transformers.__version__,'system_prompt':system,
            'settings':{'precision':'BF16','attention':'sdpa','sampling':'greedy','max_new_tokens':256,
                        'thinking':False,'seed':20261018,'local_files_only':True},
            'scope':'Exploratory questions, not an independent benchmark; native templates and simplified Chinese concise-answer instruction.',
            'models':[],'rows':[]}
    output_file=report/('question-outputs.json' if args.model=='both' else f'question-outputs-{args.model}.json')
    try:
        for key,name in targets:
            directory=WORKSPACE/'models/Qwen'/name
            manifest=json.loads((directory/'download_manifest.json').read_text(encoding='utf-8'))
            start=time.monotonic()
            print(f'Loading {name} on CUDA in BF16',flush=True)
            tokenizer=AutoTokenizer.from_pretrained(directory,local_files_only=True,trust_remote_code=False)
            loader=AutoModelForCausalLM if key=='qwen3' else AutoModelForImageTextToText
            torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            model,loading=loader.from_pretrained(directory,local_files_only=True,trust_remote_code=False,
                dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa',
                use_safetensors=True,output_loading_info=True)
            model.eval()
            missing=loading.get('missing_keys',[])
            if missing:raise RuntimeError(f'Uninitialized model parameters: {missing}')
            info={'name':name,'path':str(directory),'hf_revision':manifest['hf_revision'],
                  'loading_seconds':round(time.monotonic()-start,3),
                  'loading_info':{k:str(v) if not isinstance(v,(dict,list,str,int,float,type(None))) else v for k,v in loading.items()}}
            result['models'].append(info)
            eos=model.generation_config.eos_token_id or tokenizer.eos_token_id
            eos_ids=[eos] if isinstance(eos,int) else list(eos)
            pad=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_ids[0]
            generation=GenerationConfig(do_sample=False,max_new_tokens=256,use_cache=True,eos_token_id=eos,pad_token_id=pad)
            for qid,question in QUESTIONS:
                messages=[{'role':'system','content':system},{'role':'user','content':question}]
                template_args={'enable_thinking':False} if key=='qwen35' else {}
                text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,**template_args)
                inputs=tokenizer(text,add_special_tokens=False,return_tensors='pt').to('cuda:0')
                torch.cuda.synchronize();started=time.monotonic()
                with torch.inference_mode():
                    output=model.generate(**inputs,generation_config=generation)
                torch.cuda.synchronize()
                ids=output[0,inputs['input_ids'].shape[-1]:].tolist()
                answer=tokenizer.decode(ids,skip_special_tokens=True)
                row={'model':name,'question_id':qid,'question':question,'prompt_text':text,
                     'input_ids':inputs['input_ids'][0].tolist(),'output':answer,
                     'decoded_with_special_tokens':tokenizer.decode(ids,skip_special_tokens=False),
                     'generated_ids':ids,'tokens':len(ids),'seconds':round(time.monotonic()-started,3),
                     'stop_reason':'eos' if ids and ids[-1] in eos_ids else 'length_limit'}
                result['rows'].append(row);write_json(output_file,result)
                print(json.dumps({k:row[k] for k in ['model','question','output','tokens','stop_reason','seconds']},ensure_ascii=False),flush=True)
                del inputs,output
            info['peak_allocated_mib']=round(torch.cuda.max_memory_allocated()/2**20,1)
            del model,tokenizer;gc.collect();torch.cuda.empty_cache()
        result['completed']=True
    except BaseException as exc:
        result['error']=str(exc)
        raise
    finally:
        write_json(output_file,result)
        lines=['# Qwen模型简单问答：原始结果','',
               '使用原生聊天模板、简洁回答系统提示、非思考模式、greedy、BF16，每题最多256个新Token。没有微调或转换。这些题用于初步观察，未筛查训练重合，不构成独立能力评测。','']
        for qid,question in QUESTIONS:
            lines.extend([f'## {question}',''])
            for row in [r for r in result['rows'] if r['question_id']==qid]:
                lines.extend([f"### {row['model']}",'','```text',row['output'],'```','',
                              f"停止：{row['stop_reason']}；生成Token：{row['tokens']}。",''])
        md=report/('RAW_ANSWERS.md' if args.model=='both' else f'RAW_ANSWERS-{args.model}.md')
        md.write_text('\n'.join(lines)+'\n',encoding='utf-8')
        print(f'Question results: {output_file}',flush=True)


if __name__=='__main__':main()
