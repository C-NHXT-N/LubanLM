"""Recompute the controlled 1000/250/250 data release contract.

This gate deliberately derives counts from final rows and encoded targets.  It
does not trust summaries copied into the manifest and it never grants semantic
approval; independent review remains a separate, hash-bound requirement.
"""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path

from scripts.common import sha256
from scripts.linux_assistant.common import jsonl,fingerprint,review_digest


def validate_cross_review(data,rows):
    """Require pinned, independent decisions for >=10% in every nonempty stratum."""
    manifest=json.loads((data/'manifest.json').read_text('utf8'))
    name=manifest.get('cross_review_file')
    if not isinstance(name,str) or Path(name).name!=name:
        raise ValueError('Controlled release requires a pinned cross-review ledger')
    path=data/name
    if not path.is_file() or sha256(path)!=manifest.get('cross_review_sha256'):
        raise ValueError('Independent cross-review ledger is missing or changed')
    primary=jsonl(data/manifest.get('content_review_file','content-review.jsonl'))
    primary_by_id={r['id']:r for r in primary}
    by_id={r['id']:r for r in rows}
    decisions=jsonl(path)
    decision_ids=[r.get('id') for r in decisions]
    if len(set(decision_ids))!=len(decision_ids) or not set(decision_ids)<=set(by_id):
        raise ValueError('Cross-review contains duplicate or unknown record IDs')
    covered=Counter();population=Counter()
    def stratum(row):return row.get('split'),row.get('category'),row.get('dataset_source')
    for row in rows:population[stratum(row)]+=1
    for decision in decisions:
        row=by_id[decision['id']];reviewer=decision.get('reviewer_id')
        if (decision.get('decision')!='accept' or not decision.get('reason') or
                not reviewer or reviewer in (row.get('created_by'),primary_by_id.get(row['id'],{}).get('reviewer_id')) or
                review_digest(decision)!=fingerprint(row)):
            raise ValueError('Independent cross-review is rejected or stale: '+row['id'])
        covered[stratum(row)]+=1
    gaps={str(key):{'actual':covered[key],'required':math.ceil(n*.1)}
          for key,n in population.items() if covered[key]<math.ceil(n*.1)}
    if gaps:raise ValueError('Stratified cross-review coverage is incomplete: '+str(gaps))
    return {'reviewed_records':len(decisions),'population':len(rows),
            'fraction':len(decisions)/len(rows) if rows else 0,'ledger_sha256':sha256(path)}


CATEGORY_ALIASES = {
    "environment": "environment_check",
    "environment_check": "environment_check",
    "环境检查": "environment_check",
    "linux": "linux_operation",
    "linux_operation": "linux_operation",
    "Linux操作": "linux_operation",
    "deployment": "deployment",
    "部署流程": "deployment",
    "troubleshooting": "troubleshooting",
    "故障排查": "troubleshooting",
    "interaction": "interaction",
    "交互规范": "interaction",
    "general": "general",
    "通用能力": "general",
}


def _category(row):
    value = row.get("category", row.get("category_label"))
    if value not in CATEGORY_ALIASES:
        raise ValueError(f"Unknown controlled-release category in {row.get('id')}: {value!r}")
    return CATEGORY_ALIASES[value]


def _tags(row):
    tags = row.get("skill_tags")
    if not isinstance(tags, list) or not tags or any(not isinstance(x, str) or not x for x in tags):
        raise ValueError("Missing/non-string skill_tags: " + str(row.get("id")))
    return set(tags)


def _metadata(rows):
    for row in rows:
        _tags(row)
        if not isinstance(row.get("evidence_refs"), list) or not row["evidence_refs"]:
            raise ValueError("Missing evidence_refs: " + row["id"])
        if row.get("context_sufficient") is not True:
            raise ValueError("Context is not sufficient: " + row["id"])
        support = row.get("rubric_support")
        if not isinstance(support, (dict, list)) or not support:
            raise ValueError("Missing rubric-to-prompt support mapping: " + row["id"])
        for name in ('skill_objective','input_dependencies','counterfactual_change','dataset_source'):
            if not row.get(name):
                raise ValueError('Missing skill preparation metadata '+name+': '+row['id'])
        if row.get("stage_review_status") != "accepted":
            raise ValueError("Independent review is incomplete: " + row["id"])


def _load_split(data: Path, *names):
    return [row for name in names if (data / name).is_file() for row in jsonl(data / name)]


def _check_eval(rows, split, expected):
    if len(rows) != expected["total"]:
        raise ValueError(f"{split} must contain {expected['total']} records; found {len(rows)}")
    capabilities = Counter(row.get("evaluation_capability") for row in rows)
    wanted = expected["capabilities"]
    if capabilities != Counter(wanted):
        raise ValueError(f"{split} capability counts differ: {dict(capabilities)}")
    general = [row for row in rows if row.get("evaluation_capability") == "general"]
    subtype_names = set(expected["general_subtypes"])
    subtype_counts = Counter()
    for row in general:
        subtype = row.get("evaluation_subskill")
        found = _tags(row) & subtype_names
        if subtype not in subtype_names or found != {subtype}:
            raise ValueError(f"{split} general row needs exactly one subtype: {row['id']}")
        subtype_counts.update([subtype])
    if subtype_counts != Counter(expected["general_subtypes"]):
        raise ValueError(f"{split} general subtype counts differ: {dict(subtype_counts)}")
    return {"records": len(rows), "capabilities": dict(capabilities),
            "general_subtypes": dict(subtype_counts)}


def validate_release_contract(c, training_rows, encoded_rows):
    """Fail closed unless the final, reviewed release meets every data target."""
    contract = c.get("dataset_contract")
    if not contract:
        raise ValueError("Controlled training requires an explicit dataset_contract")
    if len(training_rows) != len(encoded_rows):
        raise ValueError("Encoded training coverage differs from final training rows")
    _metadata(training_rows)

    wanted = contract["training_scenarios"]
    scenarios = Counter(_category(row) for row in training_rows)
    if len(training_rows) != contract["training_total"] or scenarios != Counter(wanted):
        raise ValueError(f"Training scenario contract differs: total={len(training_rows)} counts={dict(scenarios)}")
    general_quotas=contract.get('training_general_subtypes')
    if general_quotas:
        observed=Counter()
        for row in training_rows:
            if _category(row)=='general':
                subtypes=_tags(row)&set(general_quotas)
                if len(subtypes)!=1:
                    raise ValueError('Training general row needs exactly one subskill: '+row['id'])
                observed.update(subtypes)
        if observed!=Counter(general_quotas):
            raise ValueError('Training general subskill quotas differ: '+str(dict(observed)))

    multiturn = sum(len(row.get("messages", [])) > 2 for row in training_rows)
    if multiturn < contract["multiturn_min"]:
        raise ValueError(f"Genuine multi-turn coverage is short: {multiturn} < {contract['multiturn_min']}")

    tagged = [(row, _tags(row), encoded) for row, encoded in zip(training_rows, encoded_rows)]
    documents = [item for item in tagged if "document_explanation" in item[1]]
    if len(documents) > contract["document_records_max"]:
        raise ValueError(f"Document explanation records exceed cap: {len(documents)}")
    target_tokens = lambda encoded: sum(value != -100 for value in encoded["labels"])
    supervised_total = sum(target_tokens(encoded) for _, _, encoded in tagged)
    supervised_documents = sum(target_tokens(encoded) for _, _, encoded in documents)
    fraction = supervised_documents / supervised_total if supervised_total else 1.0
    if fraction > contract["document_supervised_token_fraction_max"] + 1e-12:
        raise ValueError(f"Document supervised-token share exceeds cap: {fraction:.6f}")
    command_or_log = sum(bool(tags & {"command_generation", "log_troubleshooting"}) for _, tags, _ in tagged)
    if command_or_log < contract["command_and_troubleshooting_min"]:
        raise ValueError("Command/troubleshooting skill coverage is short: " + str(command_or_log))

    data = Path(c["data"])
    dev = _load_split(data, "dev.jsonl", "general-dev.jsonl")
    test = _load_split(data, "test.jsonl", "general-test.jsonl")
    _metadata(dev + test)
    if c.get('experiment_scope')=='controlled_1000_skill_rebuild':
        atom_splits={}
        for split,records in (('train',training_rows),('dev',dev),('test',test)):
            for row in records:
                atom=row.get('source_atom_id')
                if not isinstance(atom,str) or not atom:
                    raise ValueError('Missing source_atom_id: '+row['id'])
                if atom in atom_splits and atom_splits[atom]!=split:
                    raise ValueError('Original source entry crosses data splits: '+atom)
                atom_splits[atom]=split
    eval_expected = contract["evaluation"]
    cross_review=validate_cross_review(data,training_rows+dev+test) if c.get('experiment_scope')=='controlled_1000_skill_rebuild' else None
    return {
        "passed": True,
        "training_records": len(training_rows),
        "training_scenarios": dict(scenarios),
        "training_multiturn_records": multiturn,
        "training_unique_cases": len({row.get("case_id", row.get("group")) for row in training_rows}),
        "document_explanation_records": len(documents),
        "document_supervised_tokens": supervised_documents,
        "all_supervised_tokens": supervised_total,
        "document_supervised_token_fraction": fraction,
        "command_or_log_troubleshooting_records": command_or_log,
        "development": _check_eval(dev, "development", eval_expected),
        "sealed_test": _check_eval(test, "sealed_test", eval_expected),
        "independent_cross_review":cross_review,
    }
