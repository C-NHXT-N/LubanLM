"""700k artificial storage fixtures, isolated and deleted; never training data."""
import hashlib
import json
from pathlib import Path
import tempfile
import time
import psutil
from scripts.common import ROOT,write_json,utc
from scripts.sft_disk import connect,finalize

def main():
    started=time.monotonic();process=psutil.Process();peak=process.memory_info().rss
    with tempfile.TemporaryDirectory(prefix='sft-scale-fixture-',dir=ROOT/'.cache') as temp:
        folder=Path(temp);db=connect(folder/'test.sqlite');out=folder/'outputs';out.mkdir()
        task_names=['daily']*6+['concept']*2+['instruction']*2
        payload=json.dumps(dict(input_ids=[3]*512,labels=[-100]*256+[3]*256,encoding_version='prefix-stable-v1',task='daily',source='fixture',messages=[dict(role='user',content='仅用于存储压力测试'),dict(role='assistant',content='仅用于测试')]),ensure_ascii=False)
        for i in range(700000):
            identity=str(i);task=task_names[i%10];source='fixture-A' if i%3 else 'fixture-B'
            # Unique columns represent already-filtered candidates. The payload
            # arrays have realistic 512-token size; no expensive all-pool objects.
            db.execute('INSERT INTO candidates VALUES (?,?,?,?,?,?,?,?,NULL,0)',(identity,identity,identity,identity,task,source,hashlib.sha256(identity.encode()).hexdigest(),payload))
            if (i+1)%10000==0:
                db.commit();peak=max(peak,process.memory_info().rss)
                if peak>=3<<30:raise RuntimeError('RSS bound failed')
            if (i+1)%100000==0:print(json.dumps(dict(inserted=i+1,rss_bytes=process.memory_info().rss)),flush=True)
        plan=dict(seed=1,tokenizer_sha256='test-fixture-only',sft=dict(train_minimum=100000,validation_examples=5000,test_examples=1000,daily_minimum=50000,concept_maximum=30000,source_fraction_caps={'fixture-A':.8}))
        finalize(db,plan,{'test_fixture':True},out,folder,{});peak=max(peak,process.memory_info().rss)
        rows=db.execute('SELECT COUNT(*) FROM candidates').fetchone()[0]
        selected=dict(db.execute('SELECT split,COUNT(*) FROM candidates WHERE chosen=1 GROUP BY split'))
        db.close()
        result=dict(created_utc=utc(),fixtures_only=True,never_used_for_training=True,candidates=rows,selected=selected,peak_observed_rss_bytes=peak,windows_peak_working_set_bytes=process.memory_info().peak_wset,rss_cap_bytes=3<<30,seconds=time.monotonic()-started)
        if result['windows_peak_working_set_bytes']>=3<<30:raise RuntimeError('Peak working set failed')
        write_json(ROOT/'docs/run-results/teaching-repair-20261006/sft-700k-memory-test.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
