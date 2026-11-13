"""Bundle bounded test code and native token prompts, excluding weights and credentials."""
import hashlib
import io
import json
import tarfile
from scripts.common import WORKSPACE,ROOT,write_json

def main():
    report=WORKSPACE/'reports/rk3576-qwen-20261018'
    files=[(ROOT/'scripts/rk3576_probe.cpp','src/rk3576_probe.cpp'),
           (ROOT/'scripts/run_rk3576_probe.sh','run-board.sh'),
           (WORKSPACE/'.cache/rkllm-1.3.1/rkllm-runtime/Linux/librkllm_api/include/rkllm.h','include/rkllm.h')]
    for model in ['Qwen3.5-2B','Qwen3-4B-Instruct-2507']:
        for name in ['native-tokens.tsv','questions.json']:
            files.append((WORKSPACE/'exports/rk3576'/model/name,f'models/{model}/{name}'))
    manifest=[]
    with tarfile.open(report/'board-inputs.tar.gz','w:gz') as archive:
        for source,name in files:
            data=source.read_bytes();item=tarfile.TarInfo(name);item.size=len(data)
            item.mode=0o755 if name.endswith('.sh') else 0o644
            archive.addfile(item,io.BytesIO(data))
            manifest.append({'path':name,'sha256':hashlib.sha256(data).hexdigest()})
        data=(''.join(r['sha256']+'  '+r['path']+'\n' for r in manifest)).encode()
        item=tarfile.TarInfo('bundle.sha256');item.size=len(data);archive.addfile(item,io.BytesIO(data))
    write_json(report/'board-bundle-manifest.json',manifest)

if __name__=='__main__':main()
