"""固定公开版本、校验下载哈希，构建新中文混合语料。仅用CPU；不修改旧数据。"""
from __future__ import annotations
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
os.environ.setdefault('RAYON_NUM_THREADS', '4')
from scripts.common import ROOT, read_json, write_json, sha256, utc
from data.prepare import clean
from eval.chat_suite import excluded_questions, normalize
from tokenizers import Tokenizer
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
import sqlite3
import threading
import time
import urllib.request
import numpy as np
import pyarrow.parquet as pq

CJK = re.compile(r'[\u4e00-\u9fff]')
WHITESPACE = re.compile(r'\s')


def key(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def valid_text(text):
    text = clean(text)
    if not 100 <= len(text) <= 20000:
        return None, 'length'
    if any(x in text for x in ('\ufffd', '<bos>', '<eos>', '<pad>', '<think>', '</think>')):
        return None, 'invalid_character_or_control'
    if any(ord(c) < 32 and c not in '\n\t' for c in text):
        return None, 'control_character'
    if re.search(r'(.{4,80}?)\1\1', text):
        return None, 'consecutive_repeat'
    visible_length = len(WHITESPACE.sub('', text))
    if len(CJK.findall(text)) / visible_length < .55:
        return None, 'chinese_ratio'
    # 图片列表、HTML、URL堆积和代码块不是本次语言训练重点。
    if text.count('http') > 4 or text.count('![') > 2 or text.count('```') > 2:
        return None, 'markup_or_code_density'
    lines = [s.strip() for s in text.splitlines() if s.strip()]
    if len(lines) >= 5 and len(set(lines)) / len(lines) < .65:
        return None, 'duplicate_lines'
    return text, None


class Shards:
    def __init__(self, folder, tokenizer_hash):
        self.folder = folder
        folder.mkdir(parents=True, exist_ok=True)
        self.tokenizer_hash = tokenizer_hash
        self.shards, self.parts = [], []
        self.buffer_tokens = self.tokens = self.documents = 0

    def add(self, ids):
        if not ids or min(ids) < 0 or max(ids) >= 32000:
            raise ValueError('invalid token IDs')
        self.parts.append(np.asarray(ids, dtype=np.uint16))
        self.buffer_tokens += len(ids)
        self.tokens += len(ids)
        self.documents += 1
        if self.buffer_tokens >= 2_000_000:
            self.flush()

    def flush(self):
        if not self.parts:
            return
        path = self.folder / f'{len(self.shards):04d}.bin'
        np.concatenate(self.parts).tofile(path)
        self.shards.append(dict(path=str(path), tokens=self.buffer_tokens, sha256=sha256(path)))
        self.parts, self.buffer_tokens = [], 0

    def finish(self):
        self.flush()
        result = dict(dtype='uint16', shards=self.shards, tokens=self.tokens,
                      documents=self.documents, tokenizer_sha256=self.tokenizer_hash,
                      packing='EOS between documents; cross-document attention allowed')
        write_json(self.folder / 'manifest.json', result)
        return result


class Preparation:
    def __init__(self, plan):
        self.plan = plan
        self.data, self.runs = Path(plan['data_root']), Path(plan['runs_root'])
        self.report = self.runs / 'reports/preparation_progress.json'
        self.start, self.downloaded = time.monotonic(), 0
        self.download_lock = threading.Lock()
        self.booked_bytes = 0
        self.active_file = None
        self.timings = Counter()
        self.blocked = excluded_questions()
        probes = read_json(ROOT / 'docs/run-results/chat-diagnosis-20261005/probe-definition.json')
        def visit(value):
            if isinstance(value, dict):
                if 'question' in value:
                    self.blocked.add(normalize(value['question']))
                for v in value.values(): visit(v)
            elif isinstance(value, list):
                for v in value: visit(v)
        visit(probes)
        for suite_path in (ROOT/'eval/suites').glob('*.json'):
            visit(read_json(suite_path))
        self.blocked = sorted(q for q in self.blocked if len(q) >= 12)
        self.data.mkdir(parents=True, exist_ok=True)
        self.runs.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.data / 'dedup.sqlite')
        self.db.execute('CREATE TABLE IF NOT EXISTS documents (hash TEXT PRIMARY KEY, split TEXT)')
        self.db.execute('CREATE TABLE IF NOT EXISTS leads (hash TEXT PRIMARY KEY, split TEXT)')
        self.sources = []

    def guard(self):
        if time.monotonic() - self.start > self.plan['preparation_hours_cap'] * 3600:
            raise RuntimeError('CPU preparation time cap reached; no GPU training started')
        if shutil.disk_usage(self.data).free < self.plan['disk_free_bytes_floor']:
            raise RuntimeError('disk reserve reached; no GPU training started')

    def progress(self, **values):
        if self.active_file and 'file' not in values:
            values['file'] = self.active_file
        write_json(self.report, dict(updated_utc=utc(), downloaded_bytes=self.downloaded,
                                   booked_download_bytes=self.booked_bytes,
                                   timings_seconds=dict(self.timings),
                                   elapsed_seconds=round(time.monotonic()-self.start), **values))
        print(json.dumps(values, ensure_ascii=False), flush=True)

    def download(self, source, row):
        self.guard()
        repo, rev, name = source['repo'], source['revision'], row['rfilename']
        expected = row['lfs']['sha256']
        path = self.data / 'raw' / repo.replace('/', '--') / rev / name
        size = row['size']
        with self.download_lock:
            if self.booked_bytes + size > self.plan['raw_download_bytes_cap']:
                raise RuntimeError('raw download size cap reached')
            self.booked_bytes += size
        path.parent.mkdir(parents=True, exist_ok=True)
        sample = ROOT / '.cache/general-chat-v8/samples' / (repo.replace('/', '--')+'--'+name.replace('/', '--'))
        if not path.exists() and sample.exists() and sample.stat().st_size == size and sha256(sample) == expected:
            shutil.copyfile(sample, path)
        if not path.exists():
            part = path.with_suffix('.part')
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(f'https://huggingface.co/datasets/{repo}/resolve/{rev}/{name}', timeout=120) as src, part.open('wb') as dst:
                        copied, last = 0, time.monotonic()
                        while block := src.read(1 << 20):
                            dst.write(block); copied += len(block)
                            if copied > size:
                                raise RuntimeError('download exceeds pinned file size')
                            if time.monotonic()-last > 30:
                                self.guard()
                                last = time.monotonic()
                    if part.stat().st_size != size or sha256(part) != expected:
                        raise RuntimeError('download hash/size mismatch')
                    part.replace(path)
                    break
                except Exception:
                    if attempt == 2: raise
                    time.sleep(2)
        if path.stat().st_size != size or sha256(path) != expected:
            raise RuntimeError('cached public file hash/size mismatch')
        with self.download_lock:
            self.downloaded += size
        return path

    def prefetched_files(self, source, files):
        """线程只下载和验哈希；主线程按原固定顺序清洗、去重、分割、写入。"""
        workers = self.plan.get('download_workers', 1)
        window = self.plan.get('download_prefetch_files', workers)
        byte_limit = self.plan.get('download_prefetch_bytes', 1500000000)
        if not 1 <= workers <= 8 or not 1 <= window <= 8:
            raise ValueError('download concurrency must be bounded to 1..8')
        queued, next_index = deque(), 0
        executor = ThreadPoolExecutor(max_workers=workers)
        def fill():
            nonlocal next_index
            pending_bytes = sum(row['size'] for row, _ in queued)
            while next_index < len(files) and len(queued) < window:
                row = files[next_index]
                if queued and pending_bytes + row['size'] > byte_limit:
                    break
                queued.append((row, executor.submit(self.download, source, row)))
                pending_bytes += row['size']; next_index += 1
        try:
            fill()
            while queued:
                row, future = queued.popleft()
                self.active_file = row['rfilename']
                self.progress(stage='waiting_for_download', source=source['name'],
                              file=self.active_file, expected_bytes=row['size'],
                              download_workers=workers)
                waited = time.monotonic()
                # 后台线程不写共享progress，避免原子临时文件竞争。
                while not future.done():
                    self.guard()
                    self.progress(stage='waiting_for_download', source=source['name'],
                                  file=self.active_file, expected_bytes=row['size'],
                                  waiting_seconds=round(time.monotonic()-waited),
                                  download_workers=workers)
                    time.sleep(1)
                path = future.result()
                self.timings['download_wait'] += round(time.monotonic()-waited, 3)
                # 一出队便补下一文件，使网络与当前文件的CPU编码重叠。
                fill()
                yield row, path
        finally:
            for _, future in queued:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)

    def wiki(self):
        legacy = Path(self.plan['legacy_data'])
        tok = legacy / 'tokenizers/vocab-32000'
        if sha256(tok / 'tokenizer.json') != self.plan['tokenizer_sha256']:
            raise ValueError('legacy tokenizer hash changed')
        shutil.copytree(tok, self.data / 'tokenizers/vocab-32000', dirs_exist_ok=True)
        # 旧Wiki所有文档的哈希禁止作为新域验证；旧Wiki自身验证保留。
        for split in ('train', 'validation'):
            path = legacy / f'processed/wiki-{split}.jsonl'
            with path.open(encoding='utf-8') as stream:
                for i, line in enumerate(stream):
                    row = json.loads(line)
                    self.db.execute('INSERT OR IGNORE INTO documents VALUES (?,?)',
                                    (row['sha256'], 'legacy-'+split))
                    if i % 20000 == 0: self.db.commit(); self.guard()
        self.db.commit()
        manifest = read_json(legacy / 'processed/pretrain-train/manifest.json')
        shards = list(manifest['shards'])
        random.Random(self.plan['seed']).shuffle(shards)
        selected, total = [], 0
        for row in shards:
            selected.append(row); total += row['tokens']
            if total >= self.plan['wiki_train_tokens_target']: break
        train = dict(manifest, shards=selected, tokens=total, documents=None,
                     selection='fixed-seed shuffled complete immutable legacy token shards')
        return train, read_json(legacy / 'processed/pretrain-validation/manifest.json')

    def source(self, source):
        metadata = read_json(ROOT / source['metadata'])
        if metadata['sha'] != source['revision']:
            raise ValueError('metadata revision mismatch')
        files = [r for r in metadata['siblings'] if r['rfilename'].startswith(source['prefix']) and r['rfilename'].endswith('.parquet')]
        random.Random(f"{self.plan['seed']}:{source['name']}").shuffle(files)
        writers = {s: Shards(self.data / f"processed/{source['name']}/{s}", self.plan['tokenizer_sha256']) for s in ('train', 'validation')}
        tokenizer = Tokenizer.from_file(str(self.data / 'tokenizers/vocab-32000/tokenizer.json'))
        counts, downloads, previews = Counter(), [], {'train': [], 'validation': []}
        rng = random.Random(source['name'])
        preview_seen = Counter()
        raw_index = self.data / f"processed/{source['name']}/document-index.jsonl"
        stream = self.prefetched_files(source, files)
        with raw_index.open('w', encoding='utf-8') as index:
            for file, path in stream:
                if writers['train'].tokens >= source['train_tokens_target']: break
                file_start = time.monotonic()
                downloads.append(dict(file=file['rfilename'], bytes=file['size'], sha256=file['lfs']['sha256']))
                for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
                    pending = []
                    for row in batch.to_pylist():
                        counts['raw_documents'] += 1
                        if row.get('score', 0) < source['score_min']:
                            counts['score'] += 1; continue
                        if source.get('formats') and (row.get('data_format') not in source['formats'] or row.get('source') in source.get('exclude_seed_sources', [])):
                            counts['style'] += 1; continue
                        text, reason = valid_text(row['text'])
                        if reason: counts[reason] += 1; continue
                        # Cosmopedia中短篇、通俗解释优先，保留完整文档而非硬截断。
                        if source.get('formats') and len(text) > 6000:
                            counts['synthetic_long'] += 1; continue
                        normalized = normalize(text)
                        if any(q in normalized for q in self.blocked):
                            counts['known_evaluation_question'] += 1; continue
                        digest = key(text)
                        # 同开头的派生版本合并为一组；这是保守文本分组，不宣称语义去重。
                        lead = key(normalized[:160])
                        split = 'validation' if int(lead[:8], 16) % 100 < 2 else 'train'
                        if self.db.execute('SELECT 1 FROM documents WHERE hash=?', (digest,)).fetchone():
                            counts['exact_duplicate_or_legacy'] += 1; continue
                        previous = self.db.execute('SELECT split FROM leads WHERE hash=?', (lead,)).fetchone()
                        if previous:
                            counts['same_lead_duplicate'] += 1; continue
                        self.db.execute('INSERT INTO documents VALUES (?,?)', (digest, split))
                        self.db.execute('INSERT INTO leads VALUES (?,?)', (lead, split))
                        if split == 'validation' and writers[split].tokens >= self.plan['validation_tokens_per_source_cap']:
                            counts['heldout_cap_excluded'] += 1; continue
                        pending.append((text, digest, lead, split, row))
                    encodings = tokenizer.encode_batch([r[0] for r in pending], add_special_tokens=False)
                    for entry, encoding in zip(pending, encodings):
                        text, digest, lead, split, row = entry
                        ids = encoding.ids + [2]
                        if len(ids) < 128:
                            counts['too_few_tokens'] += 1; continue
                        if split == 'train' and writers[split].tokens >= source['train_tokens_target']: continue
                        writers[split].add(ids)
                        counts[f'{split}_documents'] += 1
                        index.write(json.dumps(dict(hash=digest, lead_hash=lead, split=split,
                            source=source['name'], upstream_source=row.get('source'),
                            file=file['rfilename'], tokens=len(ids), score=row['score']), ensure_ascii=False)+'\n')
                        preview_seen[split] += 1
                        sample = dict(text=text, sha256=digest, score=row['score'],
                                      upstream_source=row.get('source'), data_format=row.get('data_format'))
                        if len(previews[split]) < 32: previews[split].append(sample)
                        else:
                            position = rng.randrange(preview_seen[split])
                            if position < 32: previews[split][position] = sample
                    self.db.commit()
                    self.guard()
                    if counts['raw_documents'] % 4096 < 256:
                        self.progress(stage='encoding', source=source['name'], counts=dict(counts),
                                      train_tokens=writers['train'].tokens,
                                      validation_tokens=writers['validation'].tokens)
                    if writers['train'].tokens >= source['train_tokens_target']: break
                self.timings['clean_dedup_encode_write'] += round(time.monotonic()-file_start, 3)
                self.progress(stage='file_encoded', source=source['name'], file=file['rfilename'],
                              counts=dict(counts), train_tokens=writers['train'].tokens,
                              validation_tokens=writers['validation'].tokens)
                if writers['train'].tokens >= source['train_tokens_target']: break
        stream.close()
        result = dict(source=source, downloads=downloads, filtering_counts=dict(counts),
                      timings_seconds=dict(self.timings),
                      download_workers=self.plan.get('download_workers', 1),
                      train=writers['train'].finish(), validation=writers['validation'].finish())
        if result['train']['tokens'] < source['train_tokens_target'] or result['validation']['tokens'] < self.plan.get('source_minimum_validation_tokens', 100000):
            raise RuntimeError('source target/heldout coverage not met')
        write_json(self.runs / f"reports/{source['name']}.json", result)
        write_json(self.runs / f"reports/{source['name']}-samples.json", previews)
        self.sources.append(result)
        return result

    def run(self):
        complete = self.runs / 'reports/preparation_complete.json'
        if complete.exists():
            result = read_json(complete)
            if result['plan_sha256'] != self.plan['_sha256']:
                raise ValueError('completed preparation plan changed')
            return result
        if (self.data / 'processed/pretrain-train/manifest.json').exists() or self.db.execute("SELECT 1 FROM documents WHERE split IN ('train','validation') LIMIT 1").fetchone():
            raise RuntimeError('incomplete preparation exists: inspect before retry; do not silently overwrite')
        wiki_train, wiki_valid = self.wiki()
        sources = [self.source(source) for source in self.plan['sources']]
        new_tokens = sum(s['train']['tokens'] for s in sources)
        if new_tokens < self.plan['minimum_new_train_tokens']:
            raise RuntimeError('insufficient genuinely new corpus')
        manifests = {}
        for split, wiki in [('train', wiki_train), ('validation', wiki_valid)]:
            pieces = [dict(name='legacy-wiki', manifest=wiki)] + [dict(name=s['source']['name'], manifest=s[split]) for s in sources]
            combined = dict(dtype='uint16', tokenizer_sha256=self.plan['tokenizer_sha256'],
                            shards=[r for p in pieces for r in p['manifest']['shards']],
                            tokens=sum(p['manifest']['tokens'] for p in pieces),
                            documents=None, source_tokens={p['name']:p['manifest']['tokens'] for p in pieces},
                            packing='EOS between documents; cross-document attention allowed',
                            validation_scope='fixed new heldout documents plus unchanged legacy Wiki validation; text grouping is not a semantic leakage guarantee')
            path = self.data / f'processed/pretrain-{split}/manifest.json'
            write_json(path, combined)
            manifests[split] = dict(path=str(path), sha256=sha256(path), tokens=combined['tokens'],
                                    source_tokens=combined['source_tokens'])
        result = dict(status='prepared_not_chat_accepted', created_utc=utc(),
                      plan_sha256=self.plan['_sha256'], manifests=manifests,
                      new_train_tokens=new_tokens, raw_download_bytes=self.downloaded,
                      corpus_tokens_are_distinct_from_training_exposures=True,
                      filtered_datasets_do_not_guarantee_factual_correctness=True)
        write_json(complete, result)
        self.progress(stage='prepared', result=result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = read_json(args.plan)
    plan['_sha256'] = sha256(args.plan)
    preparation = Preparation(plan)
    try:
        preparation.run()
    finally:
        preparation.db.close()


if __name__ == '__main__':
    main()
