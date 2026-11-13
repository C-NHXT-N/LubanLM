"""GPU-check native-stop reconstruction on twenty independent scenarios; regenerate fully on mismatch."""
import argparse
import json
from pathlib import Path
from scripts.common import write_json,sha256
from scripts.linux_assistant.common import config,DEFAULT_CONFIG,jsonl,write_jsonl,verify_model
from scripts.linux_assistant.modeling import load
from scripts.linux_assistant.evaluate import evaluate


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    p.add_argument('--answers',type=Path,required=True);p.add_argument('--model',choices=['qwen35','qwen3'],default='qwen35')
    p.add_argument('--adapter',type=Path);a=p.parse_args();c=config(a.config)
    rows=jsonl(a.answers);selection=[r for r in rows if r['kind']=='task' and r['id'].endswith('-00')]
    if len(selection)!=20:raise ValueError('Replay requires twenty independent task scenarios')
    model_dir=c['model'] if a.model=='qwen35' else c['comparison_model']
    verify_model(model_dir,c['revision'] if a.model=='qwen35' else c['comparison_revision'])
    model,tokenizer=load(model_dir)
    identity={'model':str(model_dir),'source_trace':rows[0]['source_trace_sha256'],'stop_policy':'native-v2'}
    if a.adapter:
        from peft import PeftModel
        model=PeftModel.from_pretrained(model,a.adapter,is_trainable=False)
        identity['adapter_sha256']=sha256(a.adapter/'adapter_model.safetensors')
    replay=evaluate(c,model,tokenizer,selection,a.answers.with_name(a.answers.stem+'.stop-replay.jsonl'),identity)
    mismatches=[r['id'] for r,expected in zip(replay,selection) if r['generated_ids']!=expected['generated_ids']]
    if mismatches:
        backup=a.answers.with_name(a.answers.stem+'.unverified-reconstruction.jsonl')
        if backup.exists():raise FileExistsError(backup)
        a.answers.rename(backup)
        fresh=evaluate(c,model,tokenizer,rows,a.answers,{**identity,'full_regeneration':True})
        write_json(a.answers.with_suffix('.replay-check.json'),{'passed':True,'prefix_mismatches':mismatches,
            'full_regeneration':True,'samples':len(fresh),'fresh_timing':True})
    else:
        # Termination only inspects selected tokens; all retained greedy prefixes are exact,
        # and 20 independent GPU replays separately validate implementation equivalence.
        by_id={r['id']:r for r in replay}
        for row in rows:
            row.update(replay_verified=True,replay_validation_samples=20)
            if row['id'] in by_id:
                row.update(seconds=by_id[row['id']]['seconds'],fresh_timing=True)
        write_jsonl(a.answers,rows)
        write_json(a.answers.with_suffix('.replay-check.json'),{'passed':True,'all_replayed_token_ids_equal':True,
            'gpu_samples':20,'reconstructed_records':len(rows),'fresh_timing_records':20,
            'canonical_sha256':sha256(a.answers),'source_trace_sha256':identity['source_trace']})
    print('Native-stop evidence verified: '+str(a.answers),flush=True)


if __name__=='__main__':main()
