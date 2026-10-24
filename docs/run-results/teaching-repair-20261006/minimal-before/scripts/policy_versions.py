"""Data-specific identities: changing supervision must not invalidate Base data."""
import ast
import hashlib
import json
from pathlib import Path
from scripts.common import ROOT, read_json, sha256

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def pretrain_identity(plan_path,root=ROOT):
    root=Path(root);plan=read_json(plan_path)
    rules={}
    for filename,names in {
        'scripts/corpus_quality.py':['clean_document'],
        'scripts/prepare_general_2b.py':['valid_text','key'],
        'eval/chat_suite.py':['normalize','repeated','excluded_questions'],
        'scripts/prepare_general_10b.py':['fingerprint','accept_doc'],
    }.items():
        tree=ast.parse((root/filename).read_text(encoding='utf-8-sig'))
        for node in ast.walk(tree):
            if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in names:
                rules[filename+':'+node.name]=ast.dump(node,include_attributes=False)
    # Include the actual row selection / encoding body, independent of scheduling.
    tree=ast.parse((root/'scripts/prepare_general_10b.py').read_text(encoding='utf-8-sig'))
    source=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='source')
    batch=next(n for n in ast.walk(source) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='batch')
    rules['source_batch']=ast.dump(batch,include_attributes=False)
    # Module constants used by the older validator / frozen question definitions.
    rules['legacy_validator']=sha256(root/'scripts/prepare_general_2b.py')
    rules['chat_suite']=sha256(root/'eval/chat_suite.py')
    frozen=root/'docs/run-results/general-10b-20261005/frozen-test.json'
    # Archived source tree uses the unchanged current frozen fixture.
    if not frozen.exists():frozen=ROOT/'docs/run-results/general-10b-20261005/frozen-test.json'
    settings={k:plan[k] for k in ('tokenizer_sha256','seed','wiki_train_tokens_target','validation_tokens_per_source_cap','sources','additional_tokens')}
    return dict(version='pretrain-rules-v3',rules_sha256=digest(rules),settings_sha256=digest(settings),frozen_sha256=sha256(frozen) if frozen.exists() else None)

def sft_identity(plan_path):
    plan=read_json(plan_path)
    from scripts.corpus_quality import rejected_groups
    files=['scripts/corpus_quality.py','scripts/prepare_sft_100k.py','scripts/sft_disk.py','scripts/balanced_sft.py','sft/template.py','eval/chat_suite.py']
    frozen=ROOT/'docs/run-results/general-10b-20261005/frozen-test.json'
    return dict(version='sft-rules-v3',settings_sha256=digest(dict(sft=plan['sft'],seed=plan['seed'],tokenizer=plan['tokenizer_sha256'])),code={f:sha256(ROOT/f) for f in files},blocked_sha256=digest(sorted(rejected_groups())),frozen_sha256=sha256(frozen) if frozen.exists() else None)

def data_identity(plan_path,kind):
    return pretrain_identity(plan_path) if kind=='pretrain' else sft_identity(plan_path)
