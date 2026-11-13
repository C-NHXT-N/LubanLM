"""Package an accepted merged model for independent board operation, without credentials."""
import argparse
import json
from pathlib import Path
import shutil
from scripts.common import ROOT,WORKSPACE,sha256,write_json
from scripts.linux_assistant.common import config,DEFAULT_CONFIG,jsonl,write_jsonl


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    a=p.parse_args();c=config(a.config)
    merge=json.loads((c['report']/'merge-check.json').read_text(encoding='utf8'))
    conversion=json.loads((c['report']/'conversion-Qwen3.5-2B.json').read_text(encoding='utf8'))
    if not merge['passed'] or not conversion['completed']:raise ValueError('Merge/conversion gate not passed')
    wheel_dir=c['export']/'board-wheels'
    if not (wheel_dir/'requirements.lock.txt').exists():raise ValueError('Prepare offline board dependencies first')
    root=c['export']/'board-bundle'
    if root.exists():raise FileExistsError('Refusing to overwrite prior board bundle')
    root.mkdir(parents=True)
    source=Path(conversion['output'])
    if sha256(source)!=conversion['sha256']:raise ValueError('Converted file changed')
    shutil.copy2(source,root/'model.rkllm')
    for pth in (ROOT/'deploy/linux_assistant').iterdir():
        if pth.is_file() and pth.suffix in {'.py','.cpp','.sh','.txt','.md'}:shutil.copy2(pth,root/pth.name)
    for name in ['tokenizer.json','tokenizer_config.json','special_tokens_map.json','chat_template.jinja']:
        source=c['export']/'merged'/name
        if source.exists():
            (root/'tokenizer').mkdir(exist_ok=True);shutil.copy2(source,root/'tokenizer'/name)
    sdk=WORKSPACE/'.cache/rkllm-1.3.1/rkllm-runtime/Linux/librkllm_api'
    (root/'include').mkdir();shutil.copy2(sdk/'include/rkllm.h',root/'include/rkllm.h')
    (root/'lib').mkdir();shutil.copy2(sdk/'aarch64/librkllmrt.so',root/'lib/librkllmrt.so')
    (root/'wheels').mkdir()
    for wheel in wheel_dir.glob('*.whl'):shutil.copy2(wheel,root/'wheels'/wheel.name)
    shutil.copy2(wheel_dir/'requirements.lock.txt',root/'requirements.lock.txt')
    shutil.copy2(c['report']/'tokenizer-parity.json',root/'tokenizer-parity.json')
    write_jsonl(root/'dev-questions.jsonl',jsonl(c['data']/'dev.jsonl')+jsonl(c['data']/'general-dev.jsonl'))
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(c['export']/'merged',local_files_only=True)
    gen=json.loads((c['export']/'merged/generation_config.json').read_text(encoding='utf8'))
    configured=gen.get('eos_token_id',[]);configured=configured if isinstance(configured,list) else [configured]
    eos_ids=list(dict.fromkeys([tokenizer.eos_token_id,*configured]))
    write_json(root/'bundle-manifest.json',{'model':'Qwen3.5-2B Linux assistant','system_prompt':c['system_prompt'],
        'eos_token_ids':eos_ids,
        'runtime':'1.3.1','context_length':2048,'max_new_tokens':256,
        'files':{str(pth.relative_to(root)).replace('\\','/'):sha256(pth) for pth in root.rglob('*') if pth.is_file()},
        'final_tests_included':False,'board_validation':'pending'})
    print(root)


if __name__=='__main__':main()
