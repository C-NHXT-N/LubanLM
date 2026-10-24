"""编码清洗语料、分片、SFT 掩码，并据真实词表生成运行配置。"""
from __future__ import annotations
from scripts.common import DATA, ROOT, RUNS, read_json, write_json, sha256
from data.prepare import load_jsonl, save_jsonl
from sft.template import encode_pair, CHAT_TEMPLATE
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer

def main():
    directory = DATA / 'tokenizers/vocab-32000'
    tokenizer = Tokenizer.from_file(str(directory / 'tokenizer.json'))
    vocab = tokenizer.get_vocab_size()
    dtype = np.dtype('uint16' if vocab <= 65536 else 'uint32')
    eos = tokenizer.token_to_id('<eos>')
    statistics = {}
    for split in ('train', 'validation'):
        rows = load_jsonl(DATA / f'processed/wiki-{split}.jsonl')
        out = DATA / f'processed/pretrain-{split}'
        out.mkdir(exist_ok=True)
        chunk, shards, total, lengths = [], [], 0, []
        def flush():
            nonlocal chunk
            if not chunk:
                return
            path = out / f'{len(shards):04d}.bin'
            np.asarray(chunk, dtype=dtype).tofile(path)
            shards.append({'path': str(path), 'tokens': len(chunk), 'sha256': sha256(path)})
            chunk = []
        for row in rows:
            ids = tokenizer.encode(row['text'], add_special_tokens=False).ids + [eos]
            lengths.append(len(ids))
            chunk.extend(ids)
            total += len(ids)
            if len(chunk) >= 2_000_000:
                flush()
        flush()
        write_json(out / 'manifest.json', {'dtype': dtype.name, 'shards': shards, 'tokens': total, 'documents': len(rows), 'tokenizer_sha256': sha256(directory / 'tokenizer.json'), 'packing': 'cross-document context allowed; EOS between documents; no per-document block mask'})
        statistics[split] = {'tokens': total, 'documents': len(rows), 'mean_document_tokens': total / len(rows), 'shards': len(shards)}
    for split in ('train', 'validation'):
        rows = load_jsonl(DATA / f'processed/sft-{split}.jsonl')
        result = []
        for row in rows:
            pair = encode_pair(tokenizer, row['user'], row['assistant'])
            if pair is not None:
                result.append(dict(pair, tree_id=row['tree_id'], message_id=row['message_id']))
        if not result:
            raise RuntimeError(f'No SFT {split} records within context limit')
        save_jsonl(DATA / f'processed/sft-{split}-tokens.jsonl', result)
        statistics[f'sft_{split}'] = {'before_length_filter': len(rows), 'examples': len(result), 'tokens': sum(len(r['input_ids']) for r in result), 'supervised_tokens': sum(r['assistant_tokens'] for r in result), 'max_length': 512}
    config = read_json(directory / 'tokenizer_config.json')
    config['chat_template'] = CHAT_TEMPLATE
    write_json(directory / 'tokenizer_config.json', config)
    for size in ('50m', '0.5b'):
        conf = read_json(ROOT / f'C2/program/2.4/config_{size}.json')
        conf.update(vocab_size=vocab, status='runtime-teaching-v1', pad_token_id=tokenizer.token_to_id('<pad>'), bos_token_id=tokenizer.token_to_id('<bos>'), eos_token_id=eos)
        write_json(ROOT / f'configs/config_{size}.json', conf)
    write_json(RUNS / 'reports/tokenization_stats.json', statistics)
    print(statistics)

if __name__ == '__main__':
    main()
