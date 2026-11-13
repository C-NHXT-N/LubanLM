"""同一个模板服务训练、PC 推理及后续板端协议。"""
SYSTEM = '你是LubanLM，一个中文助手。'
CHAT_TEMPLATE = "{{ bos_token }}系统：{{ messages[0]['content'] if messages[0]['role'] == 'system' else '你是LubanLM，一个中文助手。' }}\n{% for message in messages %}{% if message['role'] == 'user' %}用户：{{ message['content'] }}\n{% elif message['role'] == 'assistant' %}助手：{{ message['content'] }}{{ eos_token }}\n{% endif %}{% endfor %}{% if add_generation_prompt %}助手：{% endif %}"

def prompt(user, system=SYSTEM):
    return f'<bos>系统：{system}\n用户：{user}\n助手：'

LEGACY_ENCODING = 'legacy-joint-v1'
PREFIX_ENCODING = 'prefix-stable-v1'

def encode_messages(tokenizer,messages,max_length=512,max_answer_tokens=256,system=SYSTEM):
    """Keep complete history; separately encode generation prefixes and answers."""
    if not messages or len(messages)%2: return None
    ids=[];labels=[]
    for index,message in enumerate(messages):
        role='user' if index%2==0 else 'assistant'
        if message['role']!=role or not message['content'].strip(): return None
        if role=='user':
            prefix=prompt(message['content'], system) if index==0 else '\n用户：'+message['content']+'\n助手：'
            part=tokenizer.encode(prefix,add_special_tokens=False).ids
            ids+=part;labels+=[-100]*len(part)
        else:
            answer=tokenizer.encode(message['content'],add_special_tokens=False).ids
            if len(answer)>max_answer_tokens or any(x in answer for x in (0,1,2)): return None
            ids+=answer+[2];labels+=answer+[2]
    if len(ids)>max_length: return None
    return dict(input_ids=ids,labels=labels,assistant_tokens=sum(x!=-100 for x in labels),encoding_version=PREFIX_ENCODING)

def encode_pair(tokenizer, user, assistant, max_length=512, *, encoding=LEGACY_ENCODING):
    """旧版本保留复现；新版本以推理时的提示 IDs 为不变前缀。"""
    prefix = prompt(user)
    if not assistant.strip():
        return None
    if encoding == PREFIX_ENCODING:
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False).ids
        answer_ids = tokenizer.encode(assistant, add_special_tokens=False).ids
        eos = tokenizer.token_to_id('<eos>')
        if eos is None or eos in answer_ids or tokenizer.token_to_id('<bos>') in answer_ids:
            raise ValueError('answer contains a reserved control token')
        ids = prefix_ids + answer_ids + [eos]
        if len(ids) > max_length:
            return None
        labels = [-100] * len(prefix_ids) + answer_ids + [eos]
        return {'input_ids': ids, 'labels': labels, 'assistant_tokens': len(answer_ids)+1,
                'encoding_version': PREFIX_ENCODING, 'prompt_tokens': len(prefix_ids)}
    if encoding != LEGACY_ENCODING:
        raise ValueError(f'unknown SFT encoding: {encoding}')
    text = prefix + assistant + '<eos>'
    encoded = tokenizer.encode(text, add_special_tokens=False)
    if len(encoded.ids) > max_length or not assistant.strip():
        return None  # 不截断问题/回答，也不制造没有 EOS 的训练样本。
    boundary = len(prefix)
    labels = [token if end > boundary and start >= boundary else -100 for token, (start, end) in zip(encoded.ids, encoded.offsets)]
    # 特殊 EOS 的 offsets 在此 tokenizer 中指向其可见拼写；仍明确保证 EOS 被监督。
    labels[-1] = tokenizer.token_to_id('<eos>')
    if labels[-1] != encoded.ids[-1]:
        raise ValueError('chat template EOS mismatch')
    return {'input_ids': encoded.ids, 'labels': labels, 'assistant_tokens': sum(x != -100 for x in labels)}
