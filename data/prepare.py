"""下载有界 Wikipedia 与 OASST1；保留来源、去重和不泄漏的划分。"""
from __future__ import annotations
from scripts.common import DATA, RUNS, write_json, sha256, utc
import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
import yaml
from datasets import load_dataset
from huggingface_hub import HfApi

def save_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    temporary.replace(path)

def load_jsonl(path):
    with Path(path).open(encoding='utf-8-sig') as stream:
        return [json.loads(line) for line in stream if line.strip()]

def clean(text):
    text = unicodedata.normalize('NFC', text).replace('\r\n', '\n').replace('\r', '\n')
    text = ''.join(c for c in text if c in '\n\t' or unicodedata.category(c) != 'Cc')
    return re.sub(r'\n{3,}', '\n\n', text).strip()

def wikipedia(limit):
    repo = 'wikimedia/wikipedia'
    revision = HfApi().dataset_info(repo).sha
    target = DATA / 'raw/wikipedia-zh.jsonl'
    if not target.exists():
        ds = load_dataset(repo, '20231101.zh', revision=revision, split='train', streaming=True)
        save_jsonl(target, (dict(row) for _, row in zip(range(limit), ds)))
        write_json(target.with_suffix('.source.json'), {'repo': repo, 'revision': revision, 'config': '20231101.zh', 'license': 'CC-BY-SA-3.0 / GFDL', 'limit': limit, 'sha256': sha256(target)})
    source = json.loads(target.with_suffix('.source.json').read_text(encoding='utf-8'))
    seen, kept, rejected = set(), [], Counter()
    for row in load_jsonl(target):
        text = clean(row['text'])
        if not 80 <= len(text) <= 100000:
            rejected['length'] += 1
            continue
        if any(token in text for token in ('<bos>', '<eos>', '<pad>')):
            rejected['special_spelling'] += 1
            continue
        digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
        if digest in seen:
            rejected['duplicate'] += 1
            continue
        seen.add(digest)
        # 内容哈希固定划分；同一篇重复正文永远进入同一侧。
        split = 'validation' if int(digest[:8], 16) % 100 < 2 else 'train'
        kept.append({'text': text, 'category': '中文', 'id': row['id'], 'url': row['url'], 'title': row['title'], 'sha256': digest, 'split': split})
    for split in ('train', 'validation'):
        save_jsonl(DATA / f'processed/wiki-{split}.jsonl', (r for r in kept if r['split'] == split))
    train = [r for r in kept if r['split'] == 'train']
    # BPE 只看训练文档；诊断集包含独立中文文档与明确标注的英文/代码算例。
    # BPE 合并统计使用有界训练片段，避免 Rust trainer 的临时表耗尽系统 RAM。
    # 预训练正文不截断；每隔一篇取前 512 字符，最多八百万字符。
    tokenizer_records, remaining = [], 8_000_000
    for row in train[::2]:
        piece = row['text'][:min(512, remaining)]
        if piece:
            tokenizer_records.append({'text': piece, 'category': '中文'})
            remaining -= len(piece)
        if remaining == 0:
            break
    save_jsonl(DATA / 'processed/tokenizer-train.jsonl', tokenizer_records)
    diagnostics = [r for r in kept if r['split'] == 'validation'][:100]
    diagnostics += [{'category': '英文', 'text': 'Linux is an operating system. A language model predicts the next token.'}, {'category': '代码', 'text': 'int main() { return 0; }\nfor i in range(3):\n    print(i)'}]
    save_jsonl(DATA / 'processed/tokenizer-eval.jsonl', diagnostics)
    lengths = sorted(len(r['text']) for r in kept)
    stats = {'raw_documents': len(load_jsonl(target)), 'kept_documents': len(kept), 'split_documents': dict(Counter(r['split'] for r in kept)), 'rejected': dict(rejected), 'characters': sum(lengths), 'length_min': lengths[0], 'length_median': lengths[len(lengths)//2], 'length_max': lengths[-1], 'language': 'zh', 'dedup': 'NFC cleaned exact SHA256; no near dedup', 'split': 'content hash modulo 100, 2% validation', 'tokenizer_training': {'records': len(tokenizer_records), 'characters': 8_000_000-remaining, 'selection': 'every second training document, first <=512 chars; total cap 8M; full pretraining documents retained'}}
    return source, stats

def oasst(limit):
    repo = 'OpenAssistant/oasst1'
    revision = HfApi().dataset_info(repo).sha
    target = DATA / 'raw/oasst-zh.jsonl'
    if not target.exists():
        ds = load_dataset(repo, revision=revision)
        save_jsonl(target, (dict(row, split=split) for split in ('train', 'validation') for row in ds[split] if row['lang'] == 'zh'))
        write_json(target.with_suffix('.source.json'), {'repo': repo, 'revision': revision, 'license': 'Apache-2.0', 'sha256': sha256(target)})
    rows = load_jsonl(target)
    lookup = {r['message_id']: r for r in rows}
    groups = {}
    for row in rows:
        parent = lookup.get(row['parent_id'])
        if row['role'] != 'assistant' or parent is None or parent['role'] != 'prompter':
            continue
        if row.get('deleted') or parent.get('deleted') or row.get('synthetic') or parent.get('synthetic'):
            continue
        if row.get('review_result') is not True or parent.get('review_result') is not True:
            continue
        if row['split'] != parent['split']:
            continue
        rank = row.get('rank')
        key = parent['message_id']
        if key not in groups or (rank if rank is not None else 999) < groups[key][0]:
            groups[key] = (rank if rank is not None else 999, {'user': clean(parent['text']), 'assistant': clean(row['text']), 'tree_id': row['message_tree_id'], 'split': row['split'], 'message_id': row['message_id']})
    pairs = [v[1] for _, v in sorted(groups.items()) if v[1]['user'] and v[1]['assistant'] and not any(t in v[1]['user']+v[1]['assistant'] for t in ('<bos>', '<eos>', '<pad>'))]
    for split in ('train', 'validation'):
        selected = [r for r in pairs if r['split'] == split][:limit]
        if not selected:
            raise RuntimeError(f'No usable OASST zh pairs for {split}')
        save_jsonl(DATA / f'processed/sft-{split}.jsonl', selected)
    return json.loads(target.with_suffix('.source.json').read_text(encoding='utf-8')), {'chinese_messages': len(rows), 'pairs_before_cap': dict(Counter(r['split'] for r in pairs)), 'pair_limit_per_split': limit, 'selection': 'human reviewed zh parent-child pair, best sibling rank, original split'}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--documents', type=int, default=30000)
    parser.add_argument('--sft-pairs', type=int, default=2000)
    args = parser.parse_args()
    wiki_source, wiki_stats = wikipedia(args.documents)
    print('Wikipedia prepared:', wiki_stats, flush=True)
    sft_source, sft_stats = oasst(args.sft_pairs)
    manifest = {'created_utc': utc(), 'version': 'LubanCorpus-v1-teaching', 'pretraining': wiki_source, 'sft': sft_source, 'raw_data_redistribution': False}
    (DATA / 'processed/sources.yaml').write_text(yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False), encoding='utf-8')
    write_json(RUNS / 'reports/data_stats.json', {'wiki': wiki_stats, 'sft': sft_stats, 'sources': manifest})
    print('SFT prepared:', sft_stats, flush=True)

if __name__ == '__main__':
    main()
