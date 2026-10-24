from __future__ import annotations
from scripts.common import DATA, read_json, sha256
from pathlib import Path
import torch
from tokenizers import Tokenizer
from model import LubanLM, LubanConfig
from sft.template import prompt

def load_model(checkpoint, device='cuda:0'):
    checkpoint = Path(checkpoint)
    directory = DATA / 'tokenizers/vocab-32000'
    if checkpoint.is_dir():
        from safetensors.torch import load_file
        manifest = read_json(checkpoint / 'lubanlm_conversion_manifest.json')
        weights = load_file(str(checkpoint / 'model.safetensors'))
        if manifest['source_config']['tie_word_embeddings']:
            if 'lm_head.weight' not in weights:
                weights['lm_head.weight'] = weights['model.embed_tokens.weight']
            if 'model.embed_tokens.weight' not in weights:
                weights['model.embed_tokens.weight'] = weights['lm_head.weight']
        state = {'config': manifest['source_config'], 'model': weights}
        directory = checkpoint
    else:
        from training.train import load_checkpoint
        state = load_checkpoint(checkpoint)
        if state['tokenizer_sha256'] != sha256(directory / 'tokenizer.json'):
            raise ValueError('checkpoint/tokenizer hash mismatch')
    model = LubanLM(LubanConfig.from_dict(state['config'])).to(device).eval()
    model.load_state_dict(state['model'], strict=True)
    tokenizer = Tokenizer.from_file(str(directory / 'tokenizer.json'))
    return model, tokenizer

def sample(logits, history, temperature=0., top_k=0, top_p=1., repetition_penalty=1., generator=None):
    if temperature < 0 or top_k < 0 or not 0 < top_p <= 1 or repetition_penalty <= 0:
        raise ValueError('invalid sampling parameters')
    logits = logits.float().clone()
    if history and repetition_penalty != 1:
        ids = torch.tensor(sorted(set(history)), device=logits.device)
        values = logits[ids]
        logits[ids] = torch.where(values < 0, values * repetition_penalty, values / repetition_penalty)
    if temperature == 0:
        return int(logits.argmax())
    logits /= temperature
    if top_k:
        threshold = torch.topk(logits, min(top_k, logits.numel())).values[-1]
        logits[logits < threshold] = -float('inf')
    if top_p < 1:
        sorted_logits, indices = torch.sort(logits, descending=True)
        probabilities = sorted_logits.softmax(-1)
        remove = probabilities.cumsum(-1) > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        logits[indices[remove]] = -float('inf')
    probabilities = logits.softmax(-1)
    return int(torch.multinomial(probabilities, 1, generator=generator))

@torch.inference_mode()
def generate_ids(model, tokenizer, text, max_new_tokens=64, use_cache=True, temperature=0., top_k=0, top_p=1., repetition_penalty=1., amp=True):
    """生成器逐步返回完整的新 ID 列表，让 UI 自行处理 UTF-8 部分字节。"""
    device = next(model.parameters()).device
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    if not ids or len(ids) >= model.config.context_length:
        raise ValueError('prompt must be nonempty and leave context space for generation')
    generated, past = [], None
    dtype = torch.bfloat16 if device.type == 'cuda' and torch.cuda.is_bf16_supported() else torch.float32
    for _ in range(min(max_new_tokens, model.config.context_length-len(ids))):
        current = ids + generated if past is None else [generated[-1]]
        x = torch.tensor([current], dtype=torch.long, device=device)
        with torch.autocast(device.type, dtype=dtype, enabled=amp and device.type == 'cuda'):
            result = model(x, past_key_values=past, use_cache=use_cache)
        if use_cache:
            logits, past = result.logits, result.past_key_values
        else:
            logits = result
        # 模型不应生成 padding 或重新开始；EOS 可以自然结束回答。
        scores = logits[0, -1].float().clone()
        scores[model.config.pad_token_id] = -float('inf')
        scores[model.config.bos_token_id] = -float('inf')
        token = sample(scores, ids+generated, temperature, top_k, top_p, repetition_penalty)
        generated.append(token)
        yield generated.copy()
        if token == model.config.eos_token_id:
            break

def complete(model, tokenizer, text, **kwargs):
    ids = []
    for ids in generate_ids(model, tokenizer, text, **kwargs):
        pass
    return tokenizer.decode(ids, skip_special_tokens=True), ids
