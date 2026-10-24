"""Reconstruct exact native-stop prefixes from retained greedy token traces.

This is not a fresh timing measurement. A 20-scenario GPU replay is required
before reconstructed records can be admitted as a baseline/candidate evaluation.
"""
import argparse
import json
from pathlib import Path
from transformers import AutoTokenizer
from scripts.common import write_json,sha256
from scripts.linux_assistant.common import config,DEFAULT_CONFIG,jsonl,write_jsonl,screen,fingerprint


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',choices=['qwen35','qwen3'],default='qwen35');a=p.parse_args();c=config(a.config)
    if a.output.exists():raise FileExistsError('Do not overwrite reconstructed evidence')
    tokenizer=AutoTokenizer.from_pretrained(c['model'] if a.model=='qwen35' else c['comparison_model'],local_files_only=True)
    original=jsonl(a.source);rows=[]
    native=tokenizer.eos_token_id
    for row in original:
        ids=row['generated_ids'];position=next((i for i,t in enumerate(ids) if t==native),None)
        corrected=ids[:position+1] if position is not None else ids
        answer=tokenizer.decode(corrected,skip_special_tokens=True)
        result={**row,'generated_ids':corrected,'answer':answer,
                'stop_reason':'native_assistant_end' if position is not None else row['stop_reason'],
                'seconds':None,'source_full_generation_seconds':row['seconds'],
                'source_trace_sha256':sha256(a.source),'reconstruction':'exact prefix of recorded greedy token trajectory',
                'native_eos_token_id':native,'fresh_timing':False,'replay_verified':False,
                **screen(row,answer)}
        for field in ('reviewer','review_reason','content_correct'):result.pop(field,None)
        rows.append(result)
    write_jsonl(a.output,rows)
    write_json(a.output.with_suffix('.reconstruction.json'),{'source':str(a.source),'source_sha256':sha256(a.source),
        'rows':len(rows),'native_eos':native,'fresh_timing':False,'required_gpu_replay':20})


if __name__=='__main__':main()
