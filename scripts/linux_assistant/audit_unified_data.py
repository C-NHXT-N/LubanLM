"""CPU-only audit of the active dataset, target masks, immutable held-out files and provenance."""
from collections import Counter
import hashlib
import json
from transformers import AutoTokenizer
from scripts.common import WORKSPACE,write_json,sha256
from scripts.linux_assistant.common import jsonl,verify_data,fingerprint
from scripts.linux_assistant.modeling import encode_training,evaluation_messages


def main():
    raise RuntimeError('Legacy unified-data audit is retired. Use the new source-bound rebuild audit; old snapshots and acceptance data were removed.')
    root=WORKSPACE/'datasets/linux-assistant-unified'
    if (root/'curation/quality-contract.json').exists():
        from scripts.linux_assistant.quality_audit import audit
        return audit()
    report=WORKSPACE/'reports/linux-assistant-dataset'
    c={'data':root,'model':WORKSPACE/'models/Qwen/Qwen3.5-2B','max_length':1024,
       'system_prompt':json.loads((WORKSPACE/'LubanLM/configs/linux-assistant-qwen35-baseline.json').read_text(encoding='utf8'))['system_prompt']}
    manifest=verify_data(c)
    tokenizer=AutoTokenizer.from_pretrained(c['model'],local_files_only=True)
    train=jsonl(root/'train.jsonl');lengths=[];supervised=[]
    for row in train:
        encoded=encode_training(c,tokenizer,row)
        target=[x for x in encoded['labels'] if x!=-100]
        if tokenizer.decode(target,skip_special_tokens=True)!=row['reference']:
            raise ValueError('Supervision target mismatch: '+row['id'])
        lengths.append(len(encoded['input_ids']));supervised.append(len(target))
    dev=jsonl(root/'dev.jsonl')+jsonl(root/'general-dev.jsonl')
    for row in dev:
        context=evaluation_messages(c,row)
        expected=row['messages'][:-1] if row['messages'][-1]['role']=='assistant' else row['messages']
        if context[1:]!=expected or context[-1]['role']!='user':raise ValueError('Prompt leakage: '+row['id'])
    baseline_root=WORKSPACE/'reports/linux-assistant-baseline-qwen35-20261020'
    identity=json.loads((baseline_root/'identity.json').read_text(encoding='utf8'))
    if identity['rows']!=fingerprint(dev):raise ValueError('Frozen baseline question identity changed')
    frozen=json.loads((root/'curation/frozen-evaluation.json').read_text(encoding='utf8'))
    if not all(sha256(root/name)==digest for name,digest in frozen.items()):raise ValueError('Held-out file bytes changed')
    added=jsonl(root/'curation/train-additions.jsonl')
    if any(not r.get('quality_criteria',{}).get('required_facts') or r.get('scenario_origin')!='authored_synthetic_fixture' for r in added):
        raise ValueError('New examples lack semantic criteria or origin labels')
    native={'passed':True,'samples':len(train),'train_sha256':sha256(root/'train.jsonl'),'native_tokenizer':'Qwen3.5-2B',
            'max_length_limit':1024,'max_input_tokens':max(lengths),'min_supervised_tokens':min(supervised),
            'mean_input_tokens':sum(lengths)/len(lengths),'total_supervised_tokens':sum(supervised),
            'all_final_assistant_targets_match':True,'no_silent_truncation':True,
            'development_target_turns_excluded':len(dev),'frozen_evaluation_unchanged':True,
            'baseline_question_hash_identical':True,'new_examples_with_semantic_criteria':len(added),
            'multi_turn_training_records':sum(len(r['messages'])>2 for r in train),
            'independent_review_pending_scenarios':manifest['independent_review_pending_scenarios']}
    write_json(report/'native-token-audit.json',native)
    raw=jsonl(baseline_root/'answers.jsonl')
    profile={'samples':len(raw),'stop_reasons':dict(Counter(r['stop_reason'] for r in raw)),
        'task_stop_reasons':dict(Counter(r['stop_reason'] for r in raw if r['kind']=='task')),
        'category_necessary_condition_screen':{category:{'samples':sum(r['category']==category for r in raw),
            'passes':sum(r['category']==category and r['screen_pass'] for r in raw)} for category in sorted({r['category'] for r in raw})},
        'semantic_review':'pending','warning':'A regex pass can still contain factual errors or unsupported commands; do not label this profile as semantic model accuracy.',
        'training_data_policy':'New examples address error types, not rewritten development answers; known same-scenario training records were quarantined.'}
    write_json(report/'baseline-error-profile.json',profile)
    print(json.dumps(native,ensure_ascii=False))


if __name__=='__main__':main()
