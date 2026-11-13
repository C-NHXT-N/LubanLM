"""Bounded CUDA and read-only native-checkpoint checks; never starts training."""
from __future__ import annotations
import argparse
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

from scripts.common import ROOT, WORKSPACE, DATA, sha256, require_gpu, write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,required=True)
    args=parser.parse_args()
    result={'passed':False,'python':sys.version,'executable':sys.executable,'models':[]}
    baseline=json.loads(args.baseline.read_text(encoding='utf-8'))
    try:
        import torch
        import numpy
        import tokenizers
        import transformers
        import datasets
        import safetensors
        import yaml
        from torch.nn import functional as F
        require_gpu()
        torch.set_num_threads(4)
        torch.manual_seed(20261018)
        result.update(torch=torch.__version__,cuda_build=torch.version.cuda,gpu=torch.cuda.get_device_name(0),
            compute_capability=torch.cuda.get_device_capability(0),arch_list=torch.cuda.get_arch_list(),
            packages={n:importlib.metadata.version(n) for n in ['numpy','tokenizers','transformers','datasets','pyarrow','safetensors','PyYAML','lm-eval']})
        a=torch.randn(64,64,device='cuda')
        product=a@a.T
        torch.cuda.synchronize()
        assert torch.isfinite(product).all().item()
        if not torch.cuda.is_bf16_supported(): raise RuntimeError('BF16 unavailable on selected GPU')
        with torch.autocast('cuda',dtype=torch.bfloat16):
            bf=a@a.T
        assert bf.dtype==torch.bfloat16 and torch.isfinite(bf).all().item()
        q=torch.randn(1,2,16,32,device='cuda',dtype=torch.bfloat16)
        sdpa=F.scaled_dot_product_attention(q,q,q,is_causal=True)
        assert torch.isfinite(sdpa).all().item()
        parameter=torch.nn.Parameter(torch.ones(16,device='cuda'))
        optimizer=torch.optim.AdamW([parameter],lr=0.01,fused=True)
        parameter.square().mean().backward()
        optimizer.step()
        torch.cuda.synchronize()
        assert torch.isfinite(parameter).all().item() and not torch.equal(parameter.detach(),torch.ones_like(parameter))
        result['gpu_checks']={'cuda_matmul':True,'bf16':True,'sdpa':True,'fused_adamw_isolated_tensor':True}
        print('CUDA, BF16, SDPA and isolated fused AdamW checks passed',flush=True)
        del optimizer,parameter,a,product,bf,q,sdpa
        from inference.generate import load_model, complete
        checkpoints=[('base',WORKSPACE/'runs/pretrain/checkpoints/last.pt','中国的首都是'),
                     ('sft-best',WORKSPACE/'runs/sft/checkpoints/best.pt','请用一句话解释什么是蒸发。'),
                     ('sft-last',WORKSPACE/'runs/sft/checkpoints/last.pt','请用一句话解释什么是蒸发。')]
        from sft.template import prompt
        for name,path,question in checkpoints:
            start=time.monotonic()
            model, tokenizer=load_model(path,device='cuda:0')
            text=question if name=='base' else prompt(question)
            generated,ids=complete(model,tokenizer,text,max_new_tokens=24,temperature=0.0)
            torch.cuda.synchronize()
            row={'name':name,'checkpoint':str(path),'sha256':sha256(path),'prompt':text,'output':generated,
                 'token_ids':ids,'max_new_tokens':24,'stop_reason':'eos' if ids and ids[-1]==model.config.eos_token_id else 'length_limit',
                 'seconds':round(time.monotonic()-start,3),'parameters':sum(p.numel() for p in model.parameters())}
            result['models'].append(row)
            print(json.dumps(row,ensure_ascii=False),flush=True)
            del model,tokenizer
            gc.collect();torch.cuda.empty_cache()
        actual={path:sha256(Path(path)) for path in baseline}
        assert actual==baseline, 'Protected checkpoint/tokenizer hash changed'
        result.update(passed=True,protected_files_unchanged=True,quality_evaluation=False,training_started=False)
    except BaseException as exc:
        result['error']=str(exc)
        raise
    finally:
        # Even an interrupted GPU check must report any change to protected files.
        result['protected_files_after']={path:sha256(Path(path)) for path in baseline}
        result['protected_files_unchanged']=result['protected_files_after']==baseline
        write_json(args.output,result)


if __name__=='__main__':
    main()
