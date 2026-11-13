from scripts.common import ROOT,read_json,write_json,utc
from scripts.direct_mirror import PhysicalMirror
from pathlib import Path
import argparse
import shutil
import pyarrow.parquet as pq
import json

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--plan',default='configs/50m-general-pretrain-10b.json')
    args=parser.parse_args(); plan=read_json(args.plan)
    data,runs=Path(plan['data_root']),Path(plan['runs_root'])
    mirror=PhysicalMirror(runs/'reports/network.jsonl')
    report=[]
    def guard():
        if shutil.disk_usage(data).free<plan['disk_free_bytes_floor']: raise RuntimeError('Disk reserve reached')
    data.mkdir(parents=True,exist_ok=True)
    for source in plan['sft']['raw_sources']:
        metadata=read_json(ROOT/source['metadata'])
        if metadata['sha']!=source['revision']: raise ValueError('SFT metadata revision mismatch')
        for filename in source['files']:
            row=next(r for r in metadata['siblings'] if r['rfilename']==filename)
            path=data/'sft-raw'/source['repo'].replace('/','--')/filename
            mirror.download(f"{mirror.endpoint}/datasets/{source['repo']}/resolve/{source['revision']}/{filename}",path,row['size'],row['lfs']['sha256'],guard)
            report.append(dict(repo=source['repo'],revision=source['revision'],file=filename,path=str(path),sha256=row['lfs']['sha256'],bytes=row['size']))
            if path.suffix=='.parquet':
                table=pq.ParquetFile(path)
                sample=next(table.iter_batches(batch_size=1)).to_pylist()[0]
                print(json.dumps(dict(file=filename,rows=table.metadata.num_rows,columns=list(sample),sample=str(sample)[:1300]),ensure_ascii=False),flush=True)
            else:
                with path.open(encoding='utf-8') as stream:
                    first=stream.read(1);stream.seek(0)
                    sample=json.load(stream)[0] if first=='[' else json.loads(next(stream))
                print(json.dumps(dict(file=filename,sample=sample),ensure_ascii=False)[:1600],flush=True)
            write_json(runs/'reports/sft_downloads.json',dict(timestamp=utc(),files=report))
    write_json(runs/'reports/sft_download_complete.json',dict(timestamp=utc(),files=report,physical_interface=mirror.adapter))

if __name__=='__main__': main()
