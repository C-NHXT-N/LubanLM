"""只读审计已有50M训练与数据；结果写入工作区，不启动训练、不读取最终题目。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def rows(path):
    with Path(path).open(encoding='utf-8-sig') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, default=Path('D:/LubanLM-Data/chat-v2'))
    p.add_argument('--runs-root', type=Path, default=Path('D:/LubanLM-Runs/chat-v2'))
    p.add_argument('--output', type=Path, default=ROOT/'docs/run-results/chat-diagnosis-20261005/audit.json')
    args = p.parse_args()
    from tokenizers import Tokenizer
    import torch
    from sft.template import prompt
    from training.batches import SFTPairs, TASKS, TASK_WEIGHTS

    directory = args.data_root/'processed/sft-v7'
    tokenizer_path = args.data_root/'tokenizers/vocab-32000/tokenizer.json'
    tok = Tokenizer.from_file(str(tokenizer_path))
    cfg = read(ROOT/'configs/config_50m.json')
    train, valid = rows(directory/'train.jsonl'), rows(directory/'validation.jsonl')
    findings = {}
    for split, data in [('train', train), ('validation', valid)]:
        groups = defaultdict(list)
        for r in data:
            groups[r['task']].append(r)
        findings[split] = {
            task: dict(examples=len(rs), unique_answers=len({r['assistant'] for r in rs}),
                       unique_families=len({r['family_group'] for r in rs}),
                       authored_topics=sorted({r.get('family', '') for r in rs}) if task != 'public' else None,
                       supervised_tokens=sum(r['assistant_tokens'] for r in rs),
                       source_counts=dict(Counter(r['source'] for r in rs)))
            for task, rs in groups.items()
        }
    errors = Counter()
    examples = []
    for split, data in [('train', train), ('validation', valid)]:
        for r in data:
            prefix = tok.encode(prompt(r['user']), add_special_tokens=False).ids
            ids, labels = r['input_ids'], r['labels']
            checks = {
                'length_mismatch': len(ids) != len(labels),
                'prefix_mismatch': ids[:len(prefix)] != prefix,
                'prompt_supervised': labels[:len(prefix)] != [-100]*len(prefix),
                'answer_label_mismatch': labels[len(prefix):] != ids[len(prefix):],
                'missing_supervised_eos': ids[-1] != cfg['eos_token_id'] or labels[-1] != cfg['eos_token_id'],
                'roundtrip_mismatch': tok.decode(ids[len(prefix):-1], skip_special_tokens=False) != r['assistant'],
                'invalid_token_id': any(i < 0 or i >= cfg['vocab_size'] for i in ids),
                'empty_answer': len(ids) <= len(prefix)+1,
                'replacement_character': '\ufffd' in r['user']+r['assistant'],
                'embedded_control_token': any(i in (cfg['pad_token_id'], cfg['bos_token_id'], cfg['eos_token_id']) for i in ids[len(prefix):-1]),
            }
            for key, failed in checks.items():
                if failed:
                    errors[key] += 1
                    if len(examples) < 10:
                        examples.append(dict(split=split, question_sha256=r['question_sha256'], problem=key))
    prefix_checked = len(train)+len(valid)
    token_file_alignment = {}
    for split, data in [('train', train), ('validation', valid)]:
        encoded = rows(directory/f'{split}-tokens.jsonl')
        token_file_alignment[split] = len(data) == len(encoded) and all(
            all(r[k] == e[k] for k in ('input_ids','labels','task','question_sha256'))
            for r, e in zip(data, encoded))
    train_answers = {r['assistant'] for r in train}
    overlap = {
        'exact_question_overlap': len({r['user'] for r in train} & {r['user'] for r in valid}),
        'family_overlap': len({r['family_group'] for r in train} & {r['family_group'] for r in valid}),
        'validation_answer_seen_in_train_by_task': {
            task: dict(total=sum(r['task'] == task for r in valid),
                       same_answer=sum(r['task'] == task and r['assistant'] in train_answers for r in valid))
            for task in TASKS},
        'interpretation': 'Shared short answers alone do not prove leakage. Exact ancestor exposure and template-heavy coverage make this unsuitable as an independent capability test.',
    }
    ancestor_path = args.data_root/'processed/sft-v5/train-unique.jsonl'
    if ancestor_path.exists():
        old = {r['user'] for r in rows(ancestor_path)}
        overlap['validation_questions_in_v5_ancestor_training'] = dict(Counter(r['task'] for r in valid if r['user'] in old))

    # 使用真实采样器和正式训练种子，量化2000步究竟接触了多少独立样本。
    sampler = SFTPairs(directory/'train-tokens.jsonl', cfg['pad_token_id'], torch.Generator().manual_seed(2026), batching='flat', sampling='stratified')
    first_2000 = sampler.order[:2000*8]
    exposures = Counter(first_2000)
    sampled = {}
    for task in TASKS:
        indices = [i for i, r in enumerate(train) if r['task'] == task]
        seen = [i for i in indices if exposures[i]]
        sampled[task] = dict(available=len(indices), draws=sum(exposures[i] for i in indices),
                            unique_seen=len(seen), coverage=len(seen)/len(indices),
                            max_repeats=max((exposures[i] for i in indices), default=0))
    sampling = dict(actual_stage_steps=2000, epoch_batches=len(sampler.order)//8,
                    batch_composition=dict(public=2, greeting=1, identity=1, daily=1, concept=1, instruction=1, arithmetic=1),
                    objective_weights=dict(zip(TASKS, TASK_WEIGHTS)), first_2000_steps=sampled,
                    interpretation='Stratified epochs are not complete passes over each source. Repeats are exposures, not independent examples.')
    pretrain = read(args.data_root/'processed/pretrain-train/manifest.json')
    stages = {}
    nonfinite = {}
    for name in ('50m-base-v2','50m-base-correction-v6','50m-chat-v2','50m-chat-v4','50m-chat-v5',
                 '50m-chat-v6-token','50m-chat-v6-balanced','50m-chat-v6-full','50m-chat-v7-clean',
                 '50m-v6-base-probe-400m','50m-v6-base-probe-extended'):
        d = args.runs_root/'checkpoints'/name
        summary = read(d/'summary.json')
        settings = read(d/'run_config.json')['settings']
        stage = {k: summary.get(k) for k in ('step','planned_steps','supervised_tokens','cumulative_pretrain_tokens',
                 'cumulative_sft_tokens','initial_validation_loss','final_validation_loss','stop_reason','elapsed_seconds')}
        stage['settings'] = {k: settings.get(k) for k in ('init','continue_from','lr','warmup_steps','epochs','sft_sampling','sft_objective','validation_every_steps','early_stop_patience','length')}
        vf = d/'validation_by_task.jsonl'
        if vf.exists():
            stage['validation_by_task'] = rows(vf)
        metrics_file = d/'metrics.jsonl'
        metrics = rows(metrics_file) if metrics_file.exists() else []
        nonfinite[name] = {
            'logged_updates': len(metrics),
            'nonfinite_logged_loss_lr_or_grad_norm': sum(any(not math.isfinite(r[k]) for k in ('loss','lr','grad_norm') if k in r) for r in metrics),
            'max_logged_grad_norm': max((r['grad_norm'] for r in metrics), default=None),
            'limits': 'Only logged updates are inspected; this does not prove every unlogged intermediate value was correct.',
        }
        stages[name] = stage
    h, v, layers = cfg['hidden_size'], cfg['vocab_size'], cfg['num_hidden_layers']
    per_layer = 2*h*h+2*h*cfg['num_key_value_heads']*cfg['head_dim']+3*h*cfg['intermediate_size']+2*h+2*cfg['head_dim']
    total_parameters = v*h+layers*per_layer+h
    result = dict(
        created_utc=datetime.now(timezone.utc).isoformat(), training_started=False,
        config=cfg, parameters=total_parameters, embedding_parameters=v*h,
        embedding_fraction=v*h/total_parameters,
        corpus=dict(encoded_tokens=pretrain['tokens'], documents=pretrain['documents'], packing=pretrain['packing'],
                    cumulative_pretrain_tokens=stages['50m-base-correction-v6']['cumulative_pretrain_tokens'],
                    cumulative_exposures_are_not_unique_data=True),
        data=findings, structural_audit=dict(checked_pairs=prefix_checked, errors=dict(errors), examples=examples,
                                           token_files_align_with_semantic_files=token_file_alignment),
        overlap=overlap, sampling=sampling, stages=stages, numeric_log_audit=nonfinite,
        development_review=read(args.runs_root/'reports/50m-chat-v7-clean-dev.review.json'),
        base_comparison=read(args.runs_root/'reports/base-correction-v6-comparison.json'),
        original_budget_minutes=sum(r['seconds'] for r in read(args.runs_root/'reports/gpu_budget.json'))/60,
        provenance={str(f):sha(f) for f in (tokenizer_path, directory/'train.jsonl', directory/'validation.jsonl',
                   directory/'train-tokens.jsonl', directory/'validation-tokens.jsonl', args.runs_root/'reports/base-correction-v6-comparison.json')},
    )
    write(args.output, result)
    print(json.dumps({k:result[k] for k in ('parameters','embedding_fraction','corpus','data','structural_audit','overlap','sampling')}, ensure_ascii=False, indent=2))
    print('Saved:', args.output)


if __name__ == '__main__':
    main()
