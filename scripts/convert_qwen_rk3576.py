"""Convert one verified local Qwen snapshot with official RKLLM Toolkit 1.3.1."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
import traceback

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while b:=f.read(8<<20):h.update(b)
    return h.hexdigest()

def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')

def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--model',choices=['Qwen3.5-2B','Qwen3-4B-Instruct-2507'],required=True)
    a.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    a.add_argument('--dtype',choices=['bfloat16','float16','float32'],default='bfloat16')
    a.add_argument('--reuse-qparams',action='store_true',help='Reuse a checksum-verified GRQ cache belonging to this snapshot and calibration')
    a.add_argument('--model-dir',type=Path,help='Explicit merged Hugging Face model directory')
    a.add_argument('--calibration',type=Path,help='Independent 128-row calibration JSON')
    a.add_argument('--output-dir',type=Path,help='New experiment output directory')
    a.add_argument('--report-dir',type=Path,help='New experiment report directory')
    args=a.parse_args()
    base=Path(__file__).resolve().parents[2]
    custom=[args.model_dir,args.calibration,args.output_dir,args.report_dir]
    if any(custom) and not all(custom):raise ValueError('Custom conversion requires all four explicit paths')
    directory=(args.model_dir or base/'models/Qwen'/args.model).resolve()
    out=(args.output_dir or base/'exports/rk3576'/args.model).resolve()
    if out==directory or out.is_relative_to(directory):raise ValueError('Output cannot be inside source model')
    out.mkdir(parents=True,exist_ok=True);os.chdir(out)
    report=(args.report_dir or base/'reports/rk3576-qwen-20261018').resolve()
    report.mkdir(parents=True,exist_ok=True)
    result={'completed':False,'model':args.model,'stage':'initialization','device':args.device,'dtype':args.dtype,
            'toolkit':importlib.metadata.version('rkllm-toolkit'),'quantization':'W4A16','algorithm':'grq',
            'platform':'RK3576','num_npu_core':2,'max_context':2048,'optimization_level':1}
    destination=out/f'{args.model}_w4a16_rk3576_ctx2048.rkllm'
    if destination.exists():raise FileExistsError('Refusing to overwrite existing converted model')
    temp=out/f'{args.model}_w4a16_rk3576_ctx2048.partial.rkllm'
    if temp.exists():raise FileExistsError('Inspect existing partial output before retry')
    if result['toolkit']!='1.3.1':raise RuntimeError('Requires pinned Toolkit 1.3.1')
    calibration=(args.calibration or out/'data_quant.json').resolve()
    if len(json.loads(calibration.read_text(encoding='utf8')))!=128:raise RuntimeError('Expected128 calibration samples')
    result['calibration_sha256']=digest(calibration)
    if args.model_dir:
        files={str(p.relative_to(directory)):digest(p) for p in sorted(directory.rglob('*'))
               if p.is_file() and p.suffix in {'.json','.safetensors','.jinja'}}
        if not any(n.endswith('.safetensors') for n in files):raise ValueError('Missing merged weights')
        result['source_files']=files
        result['source']=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
        if args.reuse_qparams:raise ValueError('Fresh merged-model conversion must not reuse original GRQ cache')
    else:
        result['source']=json.loads((directory/'download_manifest.json').read_text(encoding='utf8'))['hf_revision']
    qparams=None
    if args.reuse_qparams:
        cache=out/'grq.qparams'
        metadata=json.loads((out/'quantization-cache.json').read_text(encoding='utf8'))
        if (metadata['model']!=args.model or metadata['source_revision']!=result['source'] or
            metadata['calibration_sha256']!=result['calibration_sha256'] or metadata['toolkit']!='1.3.1' or
            metadata['sha256']!=digest(cache)):
            raise RuntimeError('GRQ cache provenance or checksum mismatch')
        qparams=str(cache);result['reused_qparams_sha256']=metadata['sha256']
    os.environ.update(HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    started=time.monotonic();llm=None
    try:
        import torch
        torch.set_num_threads(4)
        result.update(torch=torch.__version__,effective_dtype='float32' if args.device=='cpu' else args.dtype)
        if args.device=='cuda':result['gpu']=torch.cuda.get_device_name(0)
        if args.device=='cuda':
            from torch26_scheduler_compat import install
            result['scheduler_compatibility']=install()
        from rkllm.api import RKLLM
        llm=RKLLM()
        result['stage']='load';print('STAGE load',flush=True)
        ret=llm.load_huggingface(model=str(directory),model_lora=None,device=args.device,dtype=args.dtype,custom_config=None,load_weight=True)
        if ret!=0:raise RuntimeError(f'load_huggingface returned {ret}')
        result['stage']='build';print('STAGE build W4A16 GRQ RK3576 cores=2 ctx=2048',flush=True)
        ret=llm.build(do_quantization=True,optimization_level=1,quantized_dtype='W4A16',quantized_algorithm='grq',
            target_platform='RK3576',num_npu_core=2,extra_qparams=qparams,dataset=str(calibration),hybrid_rate=0,max_context=2048)
        if ret!=0:raise RuntimeError(f'build returned {ret}')
        result['stage']='export';print('STAGE export',flush=True)
        ret=llm.export_rkllm(str(temp))
        if ret!=0 or not temp.is_file() or temp.stat().st_size==0:raise RuntimeError(f'export failed: {ret}')
        temp.rename(destination)
        result.update(completed=True,stage='completed',output=str(destination),bytes=destination.stat().st_size,sha256=digest(destination))
    except BaseException as exc:
        result['error']=str(exc);result['traceback']=traceback.format_exc();raise
    finally:
        result['seconds']=round(time.monotonic()-started,3)
        save(report/f'conversion-{args.model}.json',result)
        print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
