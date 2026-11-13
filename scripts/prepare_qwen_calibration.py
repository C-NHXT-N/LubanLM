"""Generate 128 independent, bounded calibration answers per original model offline."""
import gc
import json
from pathlib import Path
from scripts.common import WORKSPACE, write_json, sha256, require_gpu
from scripts.probe_50m_questions import QUESTIONS

SYSTEM='你是一个中文助手。请简洁回答，尽量不超过三句话；用户指定输出格式时优先遵守。'
TOPICS=['光合作用','降雨','月相','四季','声音传播','热传导','磁铁','电流','电阻','太阳能',
        '风能','潮汐','地球自转','地球公转','食物链','细胞','细菌','酵母','种子发芽','根的作用',
        '空气成分','淡水','土壤','岩石','浮力','摩擦力','惯性','密度','质量','速度','距离','时间']
DAILY=['整理书桌','安排晨读','准备郊游','管理待办','记读书笔记','学习打字','给植物浇水','分类文件',
       '备份照片','准备演讲','制定学习计划','记录支出','清洁键盘','整理电缆','安排休息','制作清单',
       '整理衣柜','规划路线','保存文档','命名文件','比较方案','总结会议','写感谢信','准备材料',
       '阅读说明书','记录实验','检查计算','保存收据','制定预算','整理书架','选择关键词','复习单词']

def prompts():
    items=[f'用两句话向初中生解释{t}。' for t in TOPICS]
    items += [f'计算{13+i}加{31+2*i}，给出结果并简述计算过程。' for i in range(16)]
    items += [f'有{3+i}盒彩笔，每盒{4+i%5}支，一共有多少支？' for i in range(16)]
    items += [f'为“{t}”给出三条简短建议。' for t in DAILY]
    items += [f'只返回JSON对象，包含name和count两个字段，name为“物品{i+1}”，count为{i+2}。' for i in range(8)]
    items += [f'将句子翻译为英文：今天我计划{t}。' for t in DAILY[:8]]
    items += [f'请写一个Python函数，接收整数n，返回n乘以{i+2}的结果。' for i in range(8)]
    items += [f'甲有{i+4}本书，乙比甲多{i+1}本，乙有多少本？' for i in range(8)]
    assert len(items)==128 and len(set(items))==128
    assert not set(items)&{q for _,q in QUESTIONS}
    return items

def main():
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM,AutoModelForImageTextToText,GenerationConfig
    require_gpu();torch.set_num_threads(4);torch.manual_seed(20261018)
    report=WORKSPACE/'reports/rk3576-qwen-20261018';report.mkdir(parents=True,exist_ok=True)
    questions=prompts();write_json(report/'calibration-prompts.json',questions)
    for key,name in [('qwen35','Qwen3.5-2B'),('qwen3','Qwen3-4B-Instruct-2507')]:
        out=WORKSPACE/'exports/rk3576'/name;out.mkdir(parents=True,exist_ok=True)
        evidence=report/f'calibration-{key}.json'
        rows=json.loads(evidence.read_text(encoding='utf8')) if evidence.exists() else []
        if len(rows)==128:
            print(f'{name}: calibration already complete',flush=True);continue
        directory=WORKSPACE/'models/Qwen'/name
        tokenizer=AutoTokenizer.from_pretrained(directory,local_files_only=True)
        loader=AutoModelForImageTextToText if key=='qwen35' else AutoModelForCausalLM
        model=loader.from_pretrained(directory,local_files_only=True,dtype=torch.bfloat16,
                                     device_map={'':'cuda:0'},attn_implementation='sdpa').eval()
        eos=model.generation_config.eos_token_id or tokenizer.eos_token_id
        pad=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (eos if isinstance(eos,int) else eos[0])
        config=GenerationConfig(do_sample=False,max_new_tokens=96,eos_token_id=eos,pad_token_id=pad,use_cache=True)
        for i,question in enumerate(questions):
            if i<len(rows):
                if rows[i]['question']!=question:raise RuntimeError('Resume prompt mismatch')
                continue
            prompt=tokenizer.apply_chat_template([{'role':'system','content':SYSTEM},{'role':'user','content':question}],
                tokenize=False,add_generation_prompt=True,**({'enable_thinking':False} if key=='qwen35' else {}))
            inputs=tokenizer(prompt,add_special_tokens=False,return_tensors='pt').to('cuda:0')
            with torch.inference_mode():tokens=model.generate(**inputs,generation_config=config)[0,inputs['input_ids'].shape[-1]:].tolist()
            answer=tokenizer.decode(tokens,skip_special_tokens=True)
            if not answer.strip():raise RuntimeError('Empty calibration answer')
            rows.append({'question':question,'input':prompt,'target':answer,'tokens':len(tokens),
                         'length_limit':len(tokens)>=96})
            write_json(evidence,rows)
            if (i+1)%8==0:print(f'{name}: {i+1}/128',flush=True)
        write_json(out/'data_quant.json',[{'input':r['input'],'target':r['target']} for r in rows])
        write_json(out/'calibration-manifest.json',{'samples':128,'source':'original local BF16 model',
            'model':name,'system_prompt':SYSTEM,'max_new_tokens':96,'excluded_tests':[q for _,q in QUESTIONS],
            'sha256':sha256(out/'data_quant.json'),'length_limit_samples':sum(r['length_limit'] for r in rows)})
        del model,tokenizer;gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':main()
