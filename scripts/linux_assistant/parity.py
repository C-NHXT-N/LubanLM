"""Generate and verify twenty native tokenizer parity fixtures for the board."""
from __future__ import annotations
import argparse
import importlib.util
from pathlib import Path
from scripts.common import ROOT,write_json
from scripts.linux_assistant.common import config,DEFAULT_CONFIG


def frontend():
    spec=importlib.util.spec_from_file_location('board_frontend',ROOT/'deploy/linux_assistant/frontend.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def main():
    from transformers import AutoTokenizer
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    a=p.parse_args();c=config(a.config)
    native=AutoTokenizer.from_pretrained(c['model'],local_files_only=True)
    light=frontend().NativeTokenizer(c['model'])
    texts=['你好','查看Linux架构','a\nb','```sh\nuname -m\n```','路径含有 空格','tab\t字符','引号"与反斜杠\\',
           'emoji🙂测试','ASCII: 123!?','<|im_start|>用户原文','中文 English 混合','尾部换行\n',
           '空白  空白','JSON: {"a":1}','比较>=和<=','请只输出苹果','温度37°C','文件名模型.rkllm',
           '多轮历史测试','结束标记<|im_end|>原文']
    rows=[]
    for i,text in enumerate(texts):
        messages=[{'role':'system','content':c['system_prompt']}]
        if i%2:
            messages.extend([{'role':'user','content':'前一个问题'}, {'role':'assistant','content':'前一个回答。'}])
        messages.append({'role':'user','content':text})
        prompt=native.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        ids=native(prompt,add_special_tokens=False)['input_ids']
        other,other_ids=light.encode(messages)
        if prompt!=other or ids!=other_ids:raise ValueError(f'Parity failed for fixture {i}')
        rows.append({'id':f'parity-{i:02d}','messages':messages,'prompt':prompt,'input_ids':ids})
    write_json(c['report']/'tokenizer-parity.json',rows)
    write_json(c['report']/'tokenizer-parity-summary.json',{'pc_lightweight_parity':True,'samples':20,'board_parity':None})
    print('20/20 native/lightweight PC template and token-ID comparisons passed; board validation pending.')


if __name__=='__main__':main()
