"""Fetch the pinned official SDK over the physical NIC; inspectable local assets."""
import json
from pathlib import Path
import hashlib
from scripts.common import WORKSPACE, write_json
from scripts.direct_mirror import PhysicalMirror

COMMIT='f7390530443bf84f0394255a449d7cbe81e69d1c'
def main():
    root=WORKSPACE/'.cache/rkllm-1.3.1';root.mkdir(parents=True,exist_ok=True)
    report=WORKSPACE/'reports/rk3576-qwen-20261018';report.mkdir(parents=True,exist_ok=True)
    m=PhysicalMirror(report/'asset-network.jsonl',interface='WLAN',allowed_hosts={
        'dns.alidns.com','api.github.com','raw.githubusercontent.com','mirrors.tuna.tsinghua.edu.cn'})
    tree=json.loads(m.bytes(f'https://api.github.com/repos/airockchip/rknn-llm/git/trees/{COMMIT}?recursive=1'))
    write_json(report/'sdk-tree.json',tree)
    rows=[]
    for f in tree['tree']:
        p=f['path']
        needed=(p.startswith('examples/rkllm_api_demo/deploy/') or
                p.startswith('rkllm-runtime/Linux/librkllm_api/') or
                p in ['rkllm-toolkit/packages/requirements.txt',
                      'rkllm-toolkit/packages/rkllm_toolkit-1.3.1-cp310-cp310-linux_x86_64.whl',
                      'examples/rkllm_api_demo/export/export_rkllm.py',
                      'examples/rkllm_api_demo/export/generate_data_quant.py'])
        if not needed or f['type']!='blob':continue
        out=root/p;out.parent.mkdir(parents=True,exist_ok=True)
        data=out.read_bytes() if out.exists() else m.bytes(f'https://raw.githubusercontent.com/airockchip/rknn-llm/{COMMIT}/{p}',maximum=200<<20)
        if hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()!=f['sha']:raise RuntimeError(f'Git blob mismatch: {p}')
        if not out.exists():out.write_bytes(data)
        rows.append({'path':p,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
        print('Verified',p,len(data),flush=True)
    write_json(report/'sdk-manifest.json',{'commit':COMMIT,'files':rows})

if __name__=='__main__':main()
