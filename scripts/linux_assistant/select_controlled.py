"""Select a controlled LoRA checkpoint from one identical reviewed dev protocol."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random

from scripts.common import WORKSPACE, sha256, write_json
from scripts.linux_assistant.common import config, fingerprint, jsonl


AXES = ("content_correct", "completeness", "format_compliant")


def wilson(successes, total, z=1.959963984540054):
    if total <= 0: raise ValueError("Cannot score an empty evaluation slice")
    p=successes/total; d=1+z*z/total
    center=(p+z*z/(2*total))/d
    radius=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/d
    return [max(0.0,center-radius),min(1.0,center+radius)]


def paired_bootstrap(base, candidate, seed=20261019, iterations=10000):
    if len(base)!=len(candidate) or not base:raise ValueError("Paired results must be non-empty and equal length")
    rng=random.Random(seed);n=len(base);deltas=[]
    for _ in range(iterations):
        delta=sum(candidate[i]-base[i] for i in (rng.randrange(n) for _ in range(n)))/n
        deltas.append(delta)
    deltas.sort()
    return [deltas[int(.025*(iterations-1))],deltas[int(.975*(iterations-1))]]


def _axes(row):
    axes=dict(row.get("axes") or {})
    if "completeness" not in axes and type(axes.get("steps_complete")) is bool:
        axes["completeness"]=axes["steps_complete"]
    if any(type(axes.get(name)) is not bool for name in AXES):
        raise ValueError("Missing explicit calibrated axes: "+row.get("id","?"))
    accepted=all(axes[name] for name in AXES)
    if type(row.get("accepted")) is not bool or row["accepted"]!=accepted:
        raise ValueError("Accepted flag conflicts with calibrated axes: "+row.get("id","?"))
    if not row.get("review_reason") or not row.get("reviewer"):
        raise ValueError("Semantic review lacks reason/reviewer: "+row.get("id","?"))
    return axes


def _bind_review(questions, raw, reviewed, protocol_id):
    if len(raw)!=len(questions) or len(reviewed)!=len(questions):
        raise ValueError("Every model must cover the complete development set")
    bound=[]
    for q,r,d in zip(questions,raw,reviewed):
        if q["id"]!=r.get("id") or q["id"]!=d.get("id"):
            raise ValueError("Evaluation row order differs")
        for key in ("messages","reference","grading","answer","input_ids","generated_ids"):
            if key in q and key in ("messages","reference","grading") and r.get(key)!=q.get(key):
                raise ValueError("Raw evaluation question changed: "+q["id"]+" "+key)
            if d.get(key)!=r.get(key):
                raise ValueError("Semantic review changed raw output: "+q["id"]+" "+key)
        if d.get("scoring_protocol_id")!=protocol_id:
            raise ValueError("Scoring protocol differs: "+q["id"])
        axes=_axes(d)
        if 'grading' in r:
            from scripts.linux_assistant.semantic_grading import validate_model_review
            validate_model_review(r,d)
        if q.get("evaluation_capability")=="general" and type(d.get("final_conclusion_correct")) is not bool:
            raise ValueError("General row lacks final-conclusion decision: "+q["id"])
        bound.append({"question":q,"accepted":d["accepted"],"final":d.get("final_conclusion_correct"),"axes":axes})
    return bound


def _score(bound):
    result={}
    for name,predicate in {
        "task":lambda q:q.get("evaluation_capability")!="general",
        "command_generation":lambda q:q.get("evaluation_capability")=="command_generation",
        "log_troubleshooting":lambda q:q.get("evaluation_capability")=="log_troubleshooting",
        "general_full":lambda q:q.get("evaluation_capability")=="general",
    }.items():
        values=[int(row["accepted"]) for row in bound if predicate(row["question"])]
        result[name]={"passed":sum(values),"total":len(values),"rate":sum(values)/len(values),"wilson95":wilson(sum(values),len(values)),"values":values}
    final=[int(row["final"]) for row in bound if row["question"].get("evaluation_capability")=="general"]
    result["general_final"]={"passed":sum(final),"total":len(final),"rate":sum(final)/len(final),"wilson95":wilson(sum(final),len(final)),"values":final}
    return result


def _improved(base,candidate,thresholds):
    if base < thresholds["task_min"]:
        return candidate >= thresholds["task_min"] and candidate-base >= thresholds["task_gain"]-1e-12
    return (1-candidate) <= (1-base)*(1-thresholds["error_reduction"])+1e-12


def eligible(base,candidate,thresholds):
    task_names=("task","command_generation","log_troubleshooting")
    return (candidate["task"]["rate"] >= thresholds["task_min"] and
            candidate["command_generation"]["rate"] >= thresholds["command_min"] and
            candidate["log_troubleshooting"]["rate"] >= thresholds["troubleshooting_min"] and
            all(_improved(base[name]["rate"],candidate[name]["rate"],thresholds) for name in task_names) and
            base["general_full"]["rate"]-candidate["general_full"]["rate"] <= thresholds["general_max_drop"]+1e-12 and
            base["general_final"]["rate"]-candidate["general_final"]["rate"] <= thresholds["general_max_drop"]+1e-12)


def summarize(base,candidate,seed):
    public={}
    for name,value in candidate.items():
        public[name]={k:v for k,v in value.items() if k!="values"}
        public[name]["paired_delta_vs_base"]=value["rate"]-base[name]["rate"]
        metric_seed=seed+int(hashlib.sha256(name.encode()).hexdigest()[:8],16)
        public[name]["paired_bootstrap_delta95"]=paired_bootstrap(base[name]["values"],value["values"],metric_seed)
    return public


def _identity(path, data_hash, budget, system, c=None, baseline=False):
    identity=json.loads(Path(path).read_text(encoding="utf8"))
    observed_hash=identity.get("data_hash",identity.get("rows"))
    generation=identity.get("generation",{})
    observed_budget=generation.get("max_new_tokens",identity.get("max_new_tokens"))
    observed_system=identity.get("system",identity.get("system_prompt"))
    if observed_hash!=data_hash or observed_budget!=budget or observed_system!=system:
        raise ValueError("Evaluation identity differs: "+str(path))
    if generation and (generation.get("greedy") is not True or generation.get("thinking") is not False):
        raise ValueError("Evaluation decoding protocol differs: "+str(path))
    if not generation and (identity.get("greedy") is not True or identity.get("thinking") is not False):
        raise ValueError("Baseline decoding protocol differs: "+str(path))
    if c is not None:
        if (identity.get('context_length')!=c['context_length'] or
                identity.get('prompt_policy')!='context_through_last_user_exclude_target_assistant'):
            raise ValueError('Evaluation context or prompt policy differs: '+str(path))
        if baseline:
            if (Path(identity.get('model','')).resolve()!=Path(c['model']).resolve() or
                    identity.get('revision')!=c['revision']):
                raise ValueError('Baseline model or revision differs: '+str(path))
            expected_codes={name:sha256(Path(__file__).with_name(name)) for name in ('baseline.py','modeling.py')}
            if identity.get('code_sha256')!=expected_codes:
                raise ValueError('Baseline generation code differs: '+str(path))
        elif generation.get('stop_policy')!='native_assistant_end_or_text_end_v2':
            raise ValueError('Evaluation native stop policy differs: '+str(path))
    return identity


def select(c,run_name):
    from scripts.linux_assistant.selection_bindings import selection_inputs,require_first_failure
    if run_name=='lr-5e-5':require_first_failure(c)
    bindings=selection_inputs(c,run_name)
    questions=jsonl(c["data"]/"dev.jsonl")+jsonl(c["data"]/"general-dev.jsonl")
    if len(questions)!=c["dataset_contract"]["evaluation"]["total"]:
        raise ValueError("Selection data is not the complete controlled development set")
    data_hash=fingerprint(questions);protocol=c.get("selection_protocol_id","controlled-eval-axes-v1")
    base_dir=WORKSPACE/c["baseline_report"]
    _identity(base_dir/"identity.json",data_hash,c["max_new_tokens"],c["system_prompt"],c,baseline=True)
    base_raw=jsonl(base_dir/'answers.jsonl')
    base=_bind_review(questions,base_raw,jsonl(base_dir/"answers.reviewed.jsonl"),protocol)
    base_scores=_score(base)
    run=Path(c["runs"])/run_name
    steps=c["training"]["checkpoint_steps"]
    candidates=[]
    for step in steps:
        raw_path=run/f"dev-{step}.jsonl";reviewed_path=run/f"dev-{step}.reviewed.jsonl"
        identity=_identity(raw_path.with_suffix(".identity.json"),data_hash,c["max_new_tokens"],c["system_prompt"],c)
        adapter=run/f'checkpoint-{step}'
        weights_sha=sha256(adapter/'adapter_model.safetensors')
        if (identity.get('step')!=step or identity.get('adapter')!=weights_sha or
                identity.get('adapter_config_sha256')!=sha256(adapter/'adapter_config.json')):
            raise ValueError('Development generation does not match its checkpoint: '+str(step))
        raw=jsonl(raw_path)
        bound=_bind_review(questions,raw,jsonl(reviewed_path),protocol)
        for original,candidate in zip(base_raw,raw):
            if any(original.get(key)!=candidate.get(key) for key in ('prompt','input_ids','input_messages')):
                raise ValueError('Baseline/checkpoint actual input differs: '+candidate['id'])
        scores=_score(bound);passed=eligible(base_scores,scores,c["acceptance"])
        candidates.append({"step":step,"adapter":str(run/f"checkpoint-{step}"),"passed":passed,
            "raw_sha256":sha256(raw_path),"review_sha256":sha256(reviewed_path),
            "identity_sha256":sha256(raw_path.with_suffix('.identity.json')),
            "adapter_config_sha256":sha256(adapter/'adapter_config.json'),
            "adapter_sha256":weights_sha,
            "scores":summarize(base_scores,scores,c["seed"])})
    qualified=[row for row in candidates if row["passed"]]
    def rank(row):
        score=row["scores"]
        task_focus=(score["command_generation"]["rate"]+score["log_troubleshooting"]["rate"])/2
        return task_focus,score["task"]["rate"],score["general_full"]["rate"],-row["step"]
    selected=max(qualified,key=rank) if qualified else None
    baseline_public={name:{k:v for k,v in metric.items() if k!="values"} for name,metric in base_scores.items()}
    if selection_inputs(c,run_name)!=bindings:
        raise ValueError('Selection inputs changed during scoring')
    result={"passed":selected is not None,"protocol_id":protocol,"data_hash":data_hash,
        'input_bindings':bindings,
        "generation_budget":c["max_new_tokens"],"baseline":baseline_public,"candidates":candidates,
        "selected":selected,"next_action":"lock_candidate" if selected else ("run_lr_5e-5" if run_name=="lr-1e-4" else "stop_no_qualified_candidate")}
    destination=Path(c["report"])/("first-run-decision.json" if run_name=="lr-1e-4" else "second-run-decision.json")
    write_json(destination,result)
    if selected:write_json(Path(c["report"])/"candidate-selection.json",selected)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--config",type=Path,required=True)
    p.add_argument("--run",choices=["lr-1e-4","lr-5e-5"],default="lr-1e-4")
    a=p.parse_args();result=select(config(a.config),a.run);print(json.dumps(result,ensure_ascii=False))


if __name__=="__main__":main()
