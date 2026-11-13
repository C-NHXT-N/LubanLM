"""按 v3 开发错误补充通用基础课程；独立保存数据，不读取验收回答。"""
from scripts.common import DATA, RUNS, read_json, write_json, sha256, utc
from data.prepare import load_jsonl, save_jsonl, clean
from data.prepare_chat_v2 import FACTS
from data.prepare_sft_v3 import extra_instruction_pairs
from eval.chat_suite import normalize, excluded_questions
from sft.template import encode_pair
from tokenizers import Tokenizer
from collections import Counter
import hashlib
import math
import re


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def arithmetic_key(op, a, b):
    if op in ('+', '*'):
        a, b = sorted((a, b))
    return f'{op}:{a}:{b}'


def arithmetic_rows():
    # 系统枚举范围与模板，答案由整数运算产生，不按开发/验收答案挑样本。
    problems = []
    for a in range(21):
        for b in range(21):
            problems += [('+', a, b, a+b), ('-', a, b, a-b)]
    for a in range(13):
        for b in range(13):
            problems.append(('*', a, b, a*b))
    for divisor in range(1, 13):
        for quotient in range(21):
            problems.append(('/', divisor*quotient, divisor, quotient))
    words = {'+': '加', '-': '减', '*': '乘', '/': '除以'}
    for op, a, b, value in problems:
        symbolic = f'{a}{op}{b}'
        verbal = f'{a}{words[op]}{b}'
        questions = [f'算一下：{symbolic}。', f'{symbolic}的数值结果为多少？',
                     f'求{symbolic}，只写数值。', f'{verbal}是多少呢？',
                     f'算术练习，{verbal}应得多少？', f'请给出{verbal}的答案。']
        for question in questions:
            yield dict(user=question, assistant=f'{value}。',
                       source='authored-arithmetic-v4',
                       arithmetic_group=arithmetic_key(op, a, b),
                       operation=op, operands=[a, b], result=value)


def basic_rows():
    # 通用模板加不同负载：抄写、输出长度、枚举数量、日常建议、助手身份。
    for topic, answer in FACTS:
        for pattern in ['请把{}解释给初学者听。', '{}的基本含义是什么呢？',
                        '我希望了解{}的用途，请简短回答。', '给新同学讲讲{}。']:
            yield dict(user=pattern.format(topic), assistant=answer, source='authored-facts-v4')
    yield from extra_instruction_pairs()
    nouns = {'水果': ['苹果', '香蕉', '梨', '橙子'],
             '蔬菜': ['白菜', '胡萝卜', '黄瓜', '西红柿'],
             '颜色': ['红色', '蓝色', '绿色', '黄色'],
             '交通工具': ['公交车', '地铁', '火车', '自行车']}
    for kind, items in nouns.items():
        for n, number in enumerate(['一', '二', '三', '四'], 1):
            for pattern in ['只列{}种{}，不要多列。', '需要{}个{}名称，请列出。',
                            '数量要求为{}：请举出{}。', '给出{}种{}即可。']:
                yield dict(user=pattern.format(number, kind),
                           assistant='、'.join(items[:n])+'。', source='authored-instruction-v4')
    # 有不同词长，不只训练固定的两个字。
    for text in ['行', '对', '是', '收到', '明白', '欢迎', '同意', '没问题',
                 '已收到', '请稍候', '祝你顺利', '今天真好']:
        for pattern in ['按照要求仅输出“{}”。', '请直接写出{}，不要增加别的文字。',
                        '回答限定为{}，不作解释。']:
            yield dict(user=pattern.format(text), assistant=text, source='authored-instruction-v4')
    subjects = ['小鸟', '小河', '同学', '家人', '小狗', '老师', '朋友', '妈妈']
    actions = ['正在读书', '在院子里休息', '看着窗外', '走过小桥', '来到公园', '坐在树下']
    for subject in subjects:
        for action in actions:
            text = subject+action+'。'
            for pattern in ['把引号内的内容原样写出来：“{}”', '只抄写以下文字一次：{}']:
                yield dict(user=pattern.format(text), assistant=text, source='authored-instruction-v4')
    scenarios = [
        ('今天任务很多，怎样安排比较好', '先列出要做的事，挑最重要的一件开始，一次做一件。'),
        ('准备第二天一早离开家，需要提前做什么', '提前收拾要带的物品，设好闹钟，并尽量早点睡。'),
        ('书本和文具堆在桌子上，怎么收拾', '先清走不用的物品，再将书本和文具分别放好。'),
        ('雨天去外面办事，有什么准备建议', '带好雨具，穿合适的鞋，注意湿滑的路面。'),
        ('想开始一个阅读习惯，有什么办法', '每天安排一小段阅读时间，从感兴趣的书开始。'),
        ('别人替我做了事，如何礼貌回应', '可以真诚地说谢谢，并告诉对方这次帮助对你很有用。'),
        ('离家时常漏带物品，怎么办', '准备物品清单，出门时逐项检查钥匙、手机等必需品。'),
        ('看书容易被手机打断，怎么改善', '把手机放远，先专心读一小段，再安排短暂休息。'),
        ('去户外走走，怎样做好准备', '看看天气，穿舒适的鞋，选择安全的路线。'),
        ('有点疲惫，怎样调整', '先暂停手头的事，喝些水并休息一会儿。'),
        ('朋友过生日，怎样写一句祝愿', '祝你生日快乐，愿你平安顺利，每一天都开心。'),
        ('睡前怎样放松', '可以放下手机，读几页轻松的书，然后按时休息。'),
    ]
    prefixes = ['', '请问，', '朋友，', '给我一个小建议：', '我想了解一下，']
    for question, answer in scenarios:
        for prefix in prefixes:
            yield dict(user=prefix+question+'？', assistant=answer, source='authored-daily-v4')
    chats = [
        ('早晨见到你，问候一下', '早上好！祝你今天顺利。'),
        ('到了晚上，跟你打招呼', '晚上好！很高兴和你聊天。'),
        ('见到新朋友真高兴', '我也很高兴认识你！'),
        ('感谢你刚才帮了我', '不客气，很高兴能帮到你。'),
        ('现在得去工作，回头见', '再见！祝你工作顺利。'),
        ('第一次跟你说话，来打声招呼', '你好！很高兴认识你。'),
        ('愿意跟我聊一会儿么', '当然愿意。你想聊什么呢？'),
        ('这位助手的名字是什么', '我的名字是LubanLM。'),
        ('请介绍这位助手的身份和用途', '我是LubanLM，一个中文语言模型，可以用文字交流和解释简单知识。'),
        ('你是活生生的人么', '我是一个语言模型，不是真人。'),
        ('你有手脚和身体么', '我没有身体，通过文字和你交流。'),
        ('日常交流你能提供什么帮助', '我可以和你聊天，解释基础知识，并提供简单的日常建议。'),
        ('能够讲讲基础知识么', '可以。你想了解什么基础知识呢？'),
        ('用中文交流可以么', '可以，我可以用中文和你交流。'),
    ]
    for question, answer in chats:
        for prefix in prefixes:
            yield dict(user=prefix+question+'。', assistant=answer, source='authored-chat-v4')


def public_ok(row):
    q, a = row['user'], row['assistant']
    # 这是可复查的启发式筛选，不声称公共样本已逐题验证事实正确。
    conflicts = ('OpenAssistant', 'ChatGPT', 'GPT-', '人工智能', '语言模型',
                 '无法', '不能回答', '没有能力', '没有个人', '没有情感')
    return (len(q) <= 100 and len(a) <= 100 and q.strip() != a.strip()
            and not any(word.lower() in (q+a).lower() for word in conflicts)
            and not re.search(r'(.{3,30}?)\1\1', a))


def main():
    target = DATA/'processed/sft-v4'
    if target.exists():
        raise ValueError('v4 already exists; retain immutable versioned data')
    source = DATA/'processed/sft-v3'
    tok = Tokenizer.from_file(str(DATA/'tokenizers/vocab-32000/tokenizer.json'))
    blocked = excluded_questions()  # 只用于剔除同文问题，不读取任何验收回答。
    valid_raw = load_jsonl(source/'validation.jsonl')
    valid_keys = {normalize(r['user']) for r in valid_raw}
    seen = set(valid_keys)
    rejected = Counter()
    train, validation = [], list(valid_raw)
    forced_arithmetic_holdout = set()
    for r in valid_raw:
        if r['source'] == 'authored-arithmetic':
            m = re.search(r'计算(\d+)([+\-乘])(\d+)', r['user'])
            if not m:
                raise ValueError('cannot identify inherited arithmetic validation problem')
            a, op, b = m.groups()
            forced_arithmetic_holdout.add(arithmetic_key('*' if op == '乘' else op, int(a), int(b)))

    def add(r, split='train'):
        r = dict(r, user=clean(r['user']), assistant=clean(r['assistant']))
        key = normalize(r['user'])
        if key in blocked:
            rejected['suite_exact_question'] += 1
            return
        if not key or key in seen:
            rejected['duplicate_or_validation_question'] += 1
            return
        if encode_pair(tok, r['user'], r['assistant'], 256) is None:
            rejected['length'] += 1
            return
        seen.add(key)
        r.update(split=split, question_sha256=digest(key))
        (train if split == 'train' else validation).append(r)

    for r in arithmetic_rows():
        group = r['arithmetic_group']
        held = group in forced_arithmetic_holdout or int(digest(group)[:8], 16) % 100 < 5
        add(r, 'validation' if held else 'train')
    for r in basic_rows():
        add(r, 'validation' if int(digest(normalize(r['user']))[:8], 16) % 100 < 2 else 'train')
    old_train = load_jsonl(source/'train-unique.jsonl')
    for r in old_train:
        if r['source'].startswith('authored-') and r['source'] != 'authored-arithmetic':
            add(r)
    public = sorted([r for r in old_train if not r['source'].startswith('authored-') and public_ok(r)],
                    key=lambda r: digest(normalize(r['user'])))
    arithmetic = [r for r in train if r['source'] == 'authored-arithmetic-v4']
    basics = [r for r in train if r['source'] != 'authored-arithmetic-v4']
    public_limit = math.ceil(len(arithmetic)/2)
    for r in public[:public_limit]:
        add(r)
    public_count = len(train)-len(arithmetic)-len(basics)
    basic_multiplier = max(1, round(len(arithmetic)/len(basics)))
    weighted = train+basics*(basic_multiplier-1)
    train_groups = {r['arithmetic_group'] for r in train if 'arithmetic_group' in r}
    val_groups = {r['arithmetic_group'] for r in validation if 'arithmetic_group' in r} | forced_arithmetic_holdout
    assert not train_groups & val_groups
    assert not {normalize(r['user']) for r in train} & {normalize(r['user']) for r in validation}
    assert not {normalize(r['user']) for r in train+validation} & blocked
    target.mkdir(parents=True)
    save_jsonl(target/'train-unique.jsonl', train)
    save_jsonl(target/'validation.jsonl', validation)
    for split, rows in [('train', weighted), ('validation', validation)]:
        save_jsonl(target/f'{split}-tokens.jsonl', [dict(encode_pair(tok, r['user'], r['assistant'], 512),
                    source=r['source'], question_sha256=r.get('question_sha256', digest(normalize(r['user'])))) for r in rows])
    manifest = dict(created_utc=utc(),parent_manifest_sha256=sha256(source/'manifest.json'),
                    unique_train=len(train), weighted_train=len(weighted), validation=len(validation),
                    inherited_v3_validation=len(valid_raw), added_validation=len(validation)-len(valid_raw),
                    basics_multiplier=basic_multiplier, sources_unique_train=dict(Counter(r['source'] for r in train)),
                    weighted_sources=dict(Counter(r['source'] for r in weighted)), public_selected=public_count,
                    public_eligible=len(public), rejected=dict(rejected),
                    arithmetic_operation_counts=dict(Counter(r['operation'] for r in arithmetic)),
                    arithmetic_group_split_disjoint=True, question_split_disjoint=True,
                    suite_exact_questions_excluded=True, tokenizer_sha256=sha256(DATA/'tokenizers/vocab-32000/tokenizer.json'),
                    train_sha256=sha256(target/'train-tokens.jsonl'), validation_sha256=sha256(target/'validation-tokens.jsonl'),
                    policy='General curriculum from development error types; no final-answer tuning; v2/v3 and fixed probe immutable; public filtering is heuristic, not semantic certification')
    write_json(target/'manifest.json', manifest)
    write_json(RUNS/'reports/sft-v4.json', manifest)
    print(manifest)


if __name__ == '__main__':
    main()
