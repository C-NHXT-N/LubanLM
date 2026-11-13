"""Merge the selected adapter and verify next-token distributions on 20 development inputs."""
import argparse
import gc
import json
from pathlib import Path
import shutil
import torch
from scripts.common import write_json,sha256
from scripts.linux_assistant.common import config,DEFAULT_CONFIG,jsonl,verify_model
from scripts.linux_assistant.modeling import load,messages,render,generation_eos_ids


def distribution(model,tokenizer,c,rows):
    results=[];model.eval()
    for row in rows:
        prompt=render(tokenizer,messages(c,row),True)
        inputs=tokenizer(prompt,add_special_tokens=False,return_tensors='pt').to('cuda:0')
        with torch.inference_mode():result=model(**inputs,use_cache=False).logits[0,-1].float().cpu()
        results.append(result)
    return results


def main():
    from peft import PeftModel
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    a=p.parse_args();c=config(a.config);torch.set_num_threads(4)
    selected=json.loads((c['report']/'candidate-selection.json').read_text(encoding='utf8'))
    if not selected['passed']:raise ValueError('Development gate failed')
    adapter=Path(selected['adapter'])
    if sha256(adapter/'adapter_model.safetensors')!=selected['adapter_sha256']:raise ValueError('Selected adapter changed')
    verify_model(c['model'],c['revision'])
    out=c['export']/'merged'
    if out.exists():raise FileExistsError('Refuse to overwrite merged model; inspect prior merge evidence')
    model,tokenizer=load(c['model']);model=PeftModel.from_pretrained(model,adapter,is_trainable=False)
    # One variant of each of the twenty independent development scenarios.
    rows=[r for r in jsonl(c['data']/'dev.jsonl') if r['variant']==0]
    before=distribution(model,tokenizer,c,rows)
    merged=model.merge_and_unload(safe_merge=True)
    merged.generation_config.eos_token_id=generation_eos_ids(merged,tokenizer)
    merged.save_pretrained(out,safe_serialization=True,max_shard_size='2GB');tokenizer.save_pretrained(out)
    for name in ('chat_template.jinja','generation_config.json','preprocessor_config.json',
                 'video_preprocessor_config.json','vocab.json','merges.txt','LICENSE'):
        source=c['model']/name
        if source.exists() and not (out/name).exists():shutil.copy2(source,out/name)
    del model,merged;gc.collect();torch.cuda.empty_cache()
    reloaded,tokenizer=load(out);after=distribution(reloaded,tokenizer,c,rows)
    details=[]
    for row,left,right in zip(rows,before,after):
        p1=torch.softmax(left,dim=-1);p2=torch.softmax(right,dim=-1)
        js=.5*(torch.sum(p1*torch.log((p1+1e-12)/((p1+p2)*.5+1e-12)))+
                torch.sum(p2*torch.log((p2+1e-12)/((p1+p2)*.5+1e-12)))).item()
        details.append({'id':row['id'],'max_absolute_logit_difference':float((left-right).abs().max()),
                        'distribution_js_divergence':js,'top1_equal':left.argmax().item()==right.argmax().item()})
    # BF16 in-place merge is not bitwise equal: use distribution tolerance plus prediction agreement.
    passed=max(r['distribution_js_divergence'] for r in details)<=.001 and sum(r['top1_equal'] for r in details)>=19
    write_json(c['report']/'merge-check.json',{'passed':passed,'samples':details,
        'tolerance':{'max_js':.001,'minimum_top1_equal':19},
        'merged_files':{str(p.relative_to(out)):sha256(p) for p in out.rglob('*') if p.is_file()}})
    if not passed:raise ValueError('Merged/reloaded next-token distributions failed tolerance')
    from scripts.linux_assistant.evaluate import evaluate
    dev=jsonl(c['data']/'dev.jsonl')+jsonl(c['data']/'general-dev.jsonl')
    evaluate(c,reloaded,tokenizer,dev,c['report']/'merged-dev.jsonl',{'selected_adapter':selected['adapter_sha256'],'model':'merged'})
    # Calibration only uses training-side inputs, never development or final questions.
    training=jsonl(c['data']/'train.jsonl');calibration=[]
    from transformers import GenerationConfig
    eos=generation_eos_ids(reloaded,tokenizer)
    for i in range(128):
        if i<112:
            row=training[(i*37)%len(training)]
            msgs=[{'role':'system','content':c['system_prompt']},*row['messages'][:-1]]
        else:
            n=i+101
            msgs=[{'role':'system','content':c['system_prompt']},{'role':'user','content':f'只输出{n}加19的结果。'}]
        prompt=render(tokenizer,msgs,True);inputs=tokenizer(prompt,add_special_tokens=False,return_tensors='pt').to('cuda:0')
        with torch.inference_mode():ids=reloaded.generate(**inputs,do_sample=False,max_new_tokens=128,eos_token_id=eos,
                    pad_token_id=tokenizer.pad_token_id)[0,inputs['input_ids'].shape[-1]:].tolist()
        text=tokenizer.decode(ids,skip_special_tokens=True)
        if not text.strip():raise ValueError('Empty calibration answer')
        calibration.append({'input':prompt,'target':text})
    write_json(c['export']/'data_quant.json',calibration)
    write_json(c['report']/'calibration-manifest.json',{'samples':128,'source':'training-side prompts plus independent arithmetic',
        'sha256':sha256(c['export']/'data_quant.json'),'dev_test_used':False,'reuse_original_qparams':False})


if __name__=='__main__':main()
