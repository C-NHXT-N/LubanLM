"""100k isolated storage fixtures; no model, optimizer or training is run."""
import json
from pathlib import Path
import tempfile
import time
import psutil
import torch
from scripts.common import ROOT,write_json,utc
from training.batches import SFTPairs,PrefetchedSFT

def main():
    p=psutil.Process();started=time.monotonic();before=p.memory_info().rss
    with tempfile.TemporaryDirectory(prefix='reader-fixture-',dir=ROOT/'.cache') as temp:
        path=Path(temp)/'fixtures.jsonl'
        row=dict(input_ids=[1]+[3]*510+[2],labels=[-100]*256+[3]*255+[2],encoding_version='prefix-stable-v1',task='daily',messages=[dict(role='user',content='仅用于存储压力测试'),dict(role='assistant',content='不会用于训练')])
        payload=json.dumps(row,ensure_ascii=False)+'\n'
        with path.open('w',encoding='utf-8') as f:
            for _ in range(100000):f.write(payload)
        baseline=p.memory_info().rss
        data=SFTPairs(path,0,torch.Generator().manual_seed(1),batching='bucketed')
        after_load=p.memory_info().rss;peak=after_load;prefetch=PrefetchedSFT(data,16)
        try:
            for _ in range(100):
                batch=prefetch.batch(16,data.generator)
                if batch[0].shape!=(16,511):raise RuntimeError('Unexpected batch shape')
                peak=max(peak,p.memory_info().rss)
        finally:prefetch.close();data.rows.close()
        result=dict(created_utc=utc(),fixtures_only=True,never_used_for_training=True,examples=100000,tokens_per_example=512,batches=100,loader_rss_increase=after_load-baseline,observed_peak_rss=peak,windows_peak_working_set=p.memory_info().peak_wset,process_baseline=before,seconds=time.monotonic()-started,scope='CPU loader plus prefetch only; real model/optimizer/validation memory still checked by SFT benchmark')
        if result['loader_rss_increase']>256<<20:raise RuntimeError('Mapped loader exceeds 256 MiB incremental RSS')
    write_json(ROOT/'docs/run-results/teaching-repair-20261006/sft-reader-100k.json',result)
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
