"""Pin every input used to select a controlled checkpoint."""
from __future__ import annotations
from pathlib import Path
import json

from scripts.common import WORKSPACE,sha256
from scripts.linux_assistant.common import fingerprint,jsonl

RULE_FILES=('select_controlled.py','selection_bindings.py','semantic_grading.py',
            'common.py','modeling.py','baseline.py','evaluate.py')


def _plain(value):
    if isinstance(value,Path):return str(value.resolve())
    if isinstance(value,dict):return {key:_plain(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):return [_plain(item) for item in value]
    return value


def selection_inputs(c,run_name):
    if run_name not in ('lr-1e-4','lr-5e-5'):
        raise ValueError('Unsupported controlled learning-rate run')
    base=WORKSPACE/c['baseline_report'];run=Path(c['runs'])/run_name
    saved=json.loads((run/'configuration.json').read_text('utf8'))
    identity=saved.get('identity',{})
    expected_lr=c['training']['learning_rate' if run_name=='lr-1e-4' else 'second_learning_rate']
    expected_config=fingerprint({key:value for key,value in c.items() if key not in
                                ('report','export','runs','data','model','comparison_model')})
    if (saved.get('resolved_configuration')!=_plain(c) or identity.get('lr')!=expected_lr or
            identity.get('revision')!=c['revision'] or identity.get('config')!=expected_config or
            saved.get('dataset_manifest_sha256')!=sha256(Path(c['data'])/'manifest.json')):
        raise ValueError('Saved training provenance differs from configured experiment: '+run_name)
    files={
        'training_configuration':run/'configuration.json',
        'dataset_manifest':Path(c['data'])/'manifest.json',
        'development':Path(c['data'])/'dev.jsonl',
        'general_development':Path(c['data'])/'general-dev.jsonl',
        'baseline_identity':base/'identity.json',
        'baseline_raw':base/'answers.jsonl',
        'baseline_review':base/'answers.reviewed.jsonl',
    }
    for step in c['training']['checkpoint_steps']:
        checkpoint=run/f'checkpoint-{step}'
        state=json.loads((checkpoint/'run-state.json').read_text('utf8'))
        if state.get('identity')!=identity or state.get('step')!=step:
            raise ValueError('Checkpoint training identity differs: '+str(step))
        files.update({
            f'checkpoint_{step}_training_state':checkpoint/'run-state.json',
            f'checkpoint_{step}_identity':run/f'dev-{step}.identity.json',
            f'checkpoint_{step}_raw':run/f'dev-{step}.jsonl',
            f'checkpoint_{step}_review':run/f'dev-{step}.reviewed.jsonl',
            f'checkpoint_{step}_weights':checkpoint/'adapter_model.safetensors',
            f'checkpoint_{step}_config':checkpoint/'adapter_config.json',
        })
    return {'run':run_name,'configuration_fingerprint':fingerprint(_plain(c)),
            'files':{key:{'path':str(path.resolve()),'sha256':sha256(path)} for key,path in files.items()},
            'rule_code_sha256':{name:sha256(Path(__file__).with_name(name)) for name in RULE_FILES}}


def validate_selection_inputs(c,decision,run_name):
    expected=selection_inputs(c,run_name)
    if decision.get('input_bindings')!=expected:
        raise ValueError('Selection input bindings are missing or stale: '+run_name)
    candidates=decision.get('candidates',[])
    if [row.get('step') for row in candidates]!=c['training']['checkpoint_steps']:
        raise ValueError('Selection must compare every configured checkpoint in order')
    rows=jsonl(Path(c['data'])/'dev.jsonl')+jsonl(Path(c['data'])/'general-dev.jsonl')
    if (decision.get('data_hash')!=fingerprint(rows) or decision.get('generation_budget')!=c['max_new_tokens'] or
            decision.get('protocol_id')!=c['selection_protocol_id']):
        raise ValueError('Selection decision protocol differs from current development set')
    return expected


def require_first_failure(c):
    path=Path(c['report'])/'first-run-decision.json'
    decision=json.loads(path.read_text('utf8'))
    validate_selection_inputs(c,decision,'lr-1e-4')
    if (decision.get('passed') is not False or decision.get('selected') is not None or
            any(row.get('passed') is not False for row in decision['candidates'])):
        raise ValueError('Second learning rate requires a complete failed first-run decision')
    return {'decision_sha256':sha256(path),'first_run_completed_without_qualified_candidate':True}
