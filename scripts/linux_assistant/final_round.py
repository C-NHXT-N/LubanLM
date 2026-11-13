"""Lock a development-selected candidate, run the one final PC round, and assess reviewed results."""
import argparse
import gc
import json
from pathlib import Path
from scripts.common import sha256,write_json
from scripts.linux_assistant.common import config,DEFAULT_CONFIG,jsonl,write_jsonl,verify_data,verify_model,reviewed_scores,gate,fingerprint,verify_final_rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    p.add_argument('action',choices=['lock','pc','assess']);a=p.parse_args();c=config(a.config);data=verify_data(c)
    lockfile=c['report']/'candidate-lock.json'
    if a.action=='lock':
        selected=json.loads((c['report']/'candidate-selection.json').read_text(encoding='utf8'))
        merged=reviewed_scores(jsonl(c['report']/'merged-dev.reviewed.jsonl'))
        board=reviewed_scores(jsonl(c['report']/'board-dev.reviewed.jsonl'))
        for name in ('board-performance.summary.json','board-stability.summary.json','board-boundary.summary.json'):
            if not json.loads((c['report']/name).read_text(encoding='utf8')).get('passed'):raise ValueError('M4 incomplete: '+name)
        parity=json.loads((c['report']/'board-parity.json').read_text(encoding='utf8'))
        if parity.get('parity_passed') is not True or parity.get('samples')!=20:raise ValueError('Board tokenizer parity incomplete')
        baseline=json.loads((c['report']/'baseline-reviewed.json').read_text(encoding='utf8'))['models']['qwen35']['scores']
        if not gate(baseline['task'],merged['task'],baseline['general'],merged['general'],c['acceptance'],baseline['general_content'],merged['general_content']):raise ValueError('Merged development gate failed')
        if merged['task']-board['task']>c['acceptance']['quant_max_drop']+1e-9:raise ValueError('Quantization development quality gate failed')
        if merged['general']-board['general']>c['acceptance']['general_max_drop']+1e-9:raise ValueError('Quantization development general-retention gate failed')
        if merged['general_content']-board['general_content']>c['acceptance']['general_max_drop']+1e-9:raise ValueError('Quantization development content-retention gate failed')
        if lockfile.exists():raise FileExistsError('Candidate already locked')
        rows=jsonl(c['data']/'test.jsonl')+jsonl(c['data']/'general-test.jsonl')
        questions=c['report']/'final-questions.jsonl';write_jsonl(questions,rows)
        model=c['export']/'board-bundle/model.rkllm'
        merged_files={str(p.relative_to(c['export']/'merged')):sha256(p) for p in (c['export']/'merged').rglob('*') if p.is_file()}
        write_json(lockfile,{'selected':selected,'data':data,'merged_files':merged_files,
            'model_sha256':sha256(model),'questions_sha256':sha256(questions),
            'config_sha256':sha256(a.config),'final_round':1})
        return
    lock=json.loads(lockfile.read_text(encoding='utf8'))
    if lock['data']!=data or lock['config_sha256']!=sha256(a.config):raise ValueError('Final round config/data changed after lock')
    adapter=Path(lock['selected']['adapter'])
    if sha256(adapter/'adapter_model.safetensors')!=lock['selected']['adapter_sha256']:raise ValueError('Locked adapter changed')
    for name,digest in lock['merged_files'].items():
        if sha256(c['export']/'merged'/name)!=digest:raise ValueError('Locked merged model changed')
    if a.action=='pc':
        verify_model(c['model'],c['revision'])
        import torch
        from peft import PeftModel
        from scripts.linux_assistant.evaluate import evaluate
        from scripts.linux_assistant.modeling import load
        rows=jsonl(c['report']/'final-questions.jsonl')
        if sha256(c['report']/'final-questions.jsonl')!=lock['questions_sha256']:raise ValueError('Final questions changed')
        for name in ('original','adapter','merged'):
            output=c['report']/f'final-{name}.jsonl'
            # Complete outputs are retained; incomplete files resume the same frozen round only.
            if output.exists() and len(jsonl(output))==len(rows):
                identity=json.loads(output.with_suffix('.identity.json').read_text(encoding='utf8'))
                if identity.get('lock_sha256')!=sha256(lockfile) or identity.get('stage')!=name or identity.get('data_hash')!=fingerprint(rows):raise ValueError('Existing final output belongs to a different frozen round')
                continue
            model,tokenizer=load(c['export']/'merged' if name=='merged' else c['model'])
            if name=='adapter':model=PeftModel.from_pretrained(model,adapter,is_trainable=False)
            evaluate(c,model,tokenizer,rows,output,{'final_round':1,'lock_sha256':sha256(lockfile),'stage':name})
            del model,tokenizer;gc.collect();torch.cuda.empty_cache()
    else:
        scores={}
        questions=jsonl(c['report']/'final-questions.jsonl')
        if sha256(c['report']/'final-questions.jsonl')!=lock['questions_sha256']:raise ValueError('Final questions changed')
        for name in ('original','adapter','merged','board'):
            rows=jsonl(c['report']/f'final-{name}.reviewed.jsonl')
            if len(rows)!=len(questions):raise ValueError('Final results incomplete')
            raw=jsonl(c['report']/f'final-{name}.jsonl');verify_final_rows(questions,raw,rows)
            if name=='board':
                if any(r.get('model_sha256')!=lock['model_sha256'] or r.get('final_lock_sha256')!=sha256(lockfile) for r in raw):raise ValueError('Board final output does not match frozen model/round')
            else:
                identity=json.loads((c['report']/f'final-{name}.identity.json').read_text(encoding='utf8'))
                if identity.get('lock_sha256')!=sha256(lockfile) or identity.get('stage')!=name or identity.get('data_hash')!=fingerprint(questions):raise ValueError('PC final output does not match frozen round')
            scores[name]=reviewed_scores(rows)
        original=scores['original'];passed=all(gate(original['task'],scores[name]['task'],original['general'],scores[name]['general'],c['acceptance'],original['general_content'],scores[name]['general_content']) for name in ('adapter','merged'))
        passed=passed and scores['merged']['task']-scores['board']['task']<=c['acceptance']['quant_max_drop']+1e-9
        passed=passed and scores['merged']['general']-scores['board']['general']<=c['acceptance']['general_max_drop']+1e-9
        passed=passed and scores['merged']['general_content']-scores['board']['general_content']<=c['acceptance']['general_max_drop']+1e-9
        write_json(c['report']/'final-acceptance.json',{'passed':passed,'scores':scores,'round':1,
            'lock_sha256':sha256(lockfile),'if_failed':'Retain outputs. Further tuning requires a new final-test version.'})


if __name__=='__main__':main()
