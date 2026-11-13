"""Pinned model loading and final-assistant supervision, without truncation."""
from __future__ import annotations
import json
import os
from pathlib import Path
from scripts.linux_assistant.common import jsonl, verify_data, fingerprint, training_rows

os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('HF_DATASETS_OFFLINE','1')


def load(path, train=False):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForImageTextToText
    cfg=json.loads((Path(path)/'config.json').read_text(encoding='utf8'))
    loader=AutoModelForImageTextToText if cfg.get('model_type')=='qwen3_5' else AutoModelForCausalLM
    tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False)
    model,info=loader.from_pretrained(path,local_files_only=True,trust_remote_code=False,
        dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa',
        use_safetensors=True,output_loading_info=True)
    if any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
        raise RuntimeError(f'Model loading mismatch: {info}')
    if train: model.train()
    else: model.eval()
    return model,tokenizer


def messages(c,row):
    from scripts.linux_assistant.dataset_quality import conversation_issues
    problems=conversation_issues(row['messages'])
    if problems:
        raise ValueError('Invalid conversation sequence: '+row['id']+' '+','.join(problems))
    return [{'role':'system','content':c['system_prompt']},*row['messages']]


def evaluation_messages(c,row):
    """Use context through the last user turn; never expose the target answer."""
    from scripts.linux_assistant.dataset_quality import conversation_issues
    problems=conversation_issues(row['messages'])
    if problems:
        raise ValueError('Invalid conversation sequence: '+row['id']+' '+','.join(problems))
    context=list(row['messages'])
    if context and context[-1]['role']=='assistant':
        context=context[:-1]
    if not context or context[-1]['role']!='user':
        raise ValueError('Evaluation requires a final user question: '+row['id'])
    return [{'role':'system','content':c['system_prompt']},*context]


def render(tokenizer,msgs,generation):
    return tokenizer.apply_chat_template(msgs,tokenize=False,add_generation_prompt=generation,enable_thinking=False)


def generation_eos_ids(model,tokenizer):
    """A chat turn must stop at native im_end, even if text_config only names endoftext."""
    values=[]
    for item in (tokenizer.eos_token_id,tokenizer.get_vocab().get('<|im_end|>'),
                 model.generation_config.eos_token_id):
        if item is None:continue
        for token in item if isinstance(item,list) else [item]:
            if token not in values:values.append(token)
    if not values:raise ValueError('No native generation stop token')
    return values


def encode_training(c,tokenizer,row):
    msgs=messages(c,row)
    if msgs[-1]['role']!='assistant' or not msgs[-1]['content'].strip():
        raise ValueError(f'Empty/missing final assistant: {row["id"]}')
    full=render(tokenizer,msgs,False)
    prefix=render(tokenizer,evaluation_messages(c,row),True)
    if not full.startswith(prefix):
        raise ValueError(f'Training/generation prefix mismatch: {row["id"]}')
    encoded=tokenizer(full,add_special_tokens=False,return_offsets_mapping=True)
    ids=encoded['input_ids']; offsets=encoded['offset_mapping']
    if len(ids)>c['max_length']:
        raise ValueError(f'Overlong sample must be revised, never truncated: {row["id"]}: {len(ids)}')
    boundary=len(prefix)
    # A token crossing the character boundary makes safe masking ambiguous: fail loudly.
    if any(start < boundary < end for start,end in offsets):
        raise ValueError(f'Answer boundary crosses a token: {row["id"]}')
    begin=next((i for i,(start,end) in enumerate(offsets) if start>=boundary and end>start),None)
    eos=tokenizer.convert_tokens_to_ids('<|im_end|>')
    ending=[i for i in range(begin or 0,len(ids)) if ids[i]==eos]
    if begin is None or not ending: raise ValueError('No final answer or terminal EOS')
    end=ending[-1]+1
    labels=[-100]*len(ids)
    labels[begin:end]=ids[begin:end]
    return {'input_ids':ids,'attention_mask':[1]*len(ids),'labels':labels}


def training_data(c,tokenizer):
    verify_data(c)
    rows=training_rows(c)
    encoded=[encode_training(c,tokenizer,r) for r in rows]
    return rows,encoded


class Padding:
    def __init__(self,pad): self.pad=pad
    def __call__(self,rows):
        import torch
        longest=max(len(r['input_ids']) for r in rows)
        return {key:torch.tensor([r[key]+[fill]*(longest-len(r[key])) for r in rows],dtype=torch.long)
                for key,fill in [('input_ids',self.pad),('attention_mask',0),('labels',-100)]}


def targets(model):
    import torch
    found=[name for name,module in model.named_modules()
           if name.startswith('model.language_model.layers.') and isinstance(module,torch.nn.Linear)]
    if not found or any('visual' in n or 'lm_head' in n for n in found):
        raise ValueError('No safe language-only LoRA target list')
    for suffix in ('in_proj_qkv','out_proj','q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'):
        if not any(n.endswith('.'+suffix) for n in found): raise ValueError('Missing hybrid target '+suffix)
    return found
