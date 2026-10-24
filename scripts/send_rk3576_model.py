"""Send a completed, verified conversion only to the authorized cat board via Taildrop."""
import argparse
import json
from pathlib import Path
import subprocess
from scripts.common import WORKSPACE,sha256,write_json

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=['Qwen3.5-2B','Qwen3-4B-Instruct-2507'],required=True)
    args=p.parse_args();report=WORKSPACE/'reports/rk3576-qwen-20261018'
    result=json.loads((report/f'conversion-{args.model}.json').read_text(encoding='utf8'))
    if not result['completed']:raise RuntimeError('Conversion is not complete')
    folder=WORKSPACE/'exports/rk3576'/args.model
    model=folder/Path(result['output']).name
    if not model.is_file() or sha256(model)!=result['sha256']:raise RuntimeError('Converted output checksum mismatch')
    manifest=folder/f'{args.model}.sha256'
    manifest.write_text(result['sha256']+'  '+model.name+'\n',encoding='utf8')
    subprocess.run(['C:/Program Files/Tailscale/tailscale.exe','file','cp','--verbose',str(model),str(manifest),'cat:'],check=True)
    write_json(report/f'transfer-{args.model}.json',{'sent':True,'peer':'cat','peer_ip':'100.64.207.34',
        'bytes':model.stat().st_size,'sha256':result['sha256'],'remote_verified':False})
    print('Sent; board must verify the SHA256 before running.',flush=True)

if __name__=='__main__':main()
