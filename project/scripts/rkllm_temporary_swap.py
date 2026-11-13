"""Temporary task-owned WSL swap for SDK FlatBuffer export; no WSL reboot/config edit."""
import argparse
import json
import os
from pathlib import Path
import subprocess

FOLDER=Path('/home/aiyong/.cache/lubanlm-rkllm')
SWAP=FOLDER/'export-20261018.swap'
MARKER=FOLDER/'export-20261018.owner.json'
REPORT=Path('/mnt/c/Lab/LubanLM-lab/reports/rk3576-qwen-20261018/swap-status.json')
BYTES=32*1024**3
def active():return str(SWAP) in subprocess.check_output(['swapon','--noheadings','--raw','--show=NAME'],text=True).splitlines()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cleanup',action='store_true');a=p.parse_args()
    if os.geteuid()!=0:raise RuntimeError('Run only as WSL root for swap management')
    FOLDER.mkdir(parents=True,exist_ok=True)
    if SWAP.exists() and not MARKER.exists():raise RuntimeError('Refusing to modify unrecognized existing swap file')
    if a.cleanup:
        if not MARKER.exists():raise RuntimeError('Task ownership marker missing')
        if active():subprocess.run(['swapoff',str(SWAP)],check=True)
        if SWAP.exists():SWAP.unlink()
        MARKER.unlink()
        result={'active':False,'removed':True,'path':str(SWAP),'bytes':BYTES}
    else:
        if not SWAP.exists():
            SWAP.touch(exist_ok=False);SWAP.chmod(0o600)
            MARKER.write_text(json.dumps({'task':'rk3576-qwen-20261018','path':str(SWAP)}))
            subprocess.run(['fallocate','-l',str(BYTES),str(SWAP)],check=True)
            subprocess.run(['mkswap',str(SWAP)],check=True)
        if not active():subprocess.run(['swapon',str(SWAP)],check=True)
        result={'active':True,'removed':False,'path':str(SWAP),'bytes':BYTES}
    REPORT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result));subprocess.run(['free','-h'],check=True)

if __name__=='__main__':main()
