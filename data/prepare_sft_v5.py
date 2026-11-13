"""保留v4算术分组，扩大基础表达和公共问答覆盖；不覆盖既有数据。"""
from scripts.common import DATA,RUNS,write_json,sha256,utc
from data.prepare import load_jsonl,save_jsonl,clean
from data.prepare_sft_v4 import public_ok,digest
from eval.chat_suite import normalize,excluded_questions
from sft.template import encode_pair
from tokenizers import Tokenizer
from collections import Counter
import re


def new_basics():
    # 同一任务配多种问法与负载，避免只对一个固定句子作出反应。
    identities = {
        'name': (['你的名称是什么', '请说一下助手的名字', '这位助手如何称呼',
                  '你有一个什么样的名字', '告诉我用什么名字叫你', '我想知道你名叫什么',
                  '我可以用哪个名字称呼这位助手', '介绍自己的名称', '别人通常怎样称呼你',
                  '如何称呼正在和我交流的助手'], '我叫LubanLM，你可以这样称呼我。'),
        'intro': (['做一个简短的自我介绍', '讲讲这位助手的身份', '告诉我你是什么助手',
                   '让新朋友认识一下你', '简单说明你自己是什么', '向同学介绍这位助手',
                   '请介绍你作为助手的角色', '自我介绍用一句话就好', '简单介绍自己是谁',
                   '你是什么类型的聊天助手'], '我是LubanLM，一个用中文进行文字交流的语言模型。'),
        'help': (['你可以提供哪些简单帮助', '说明助手能协助哪些事', '平时怎样使用这个助手',
                  '请列举你能够帮忙的事情', '能给用户什么基础帮助', '讲讲你的主要用途',
                  '聊天之外你可以协助什么', '你能解释基础知识么', '日常问题可以请你帮忙么',
                  '中文交流时你能怎样帮忙'], '我可以用中文聊天，解释简单知识，并给出日常建议。'),
        'human': (['你属于人类么', '你是不是一个真人助手', '你是计算机里的软件么',
                   '和我交流的是真人还是语言模型', '助手是不是活生生的人', '你作为语言模型是什么',
                   '你有自己的手和脚么', '你可以像人那样行走么', '你有实体的身体么',
                   '这个聊天助手有肉身么'], '我是一个语言模型，不是真人，也没有身体。'),
        'chinese': (['这是一位中文语言助手么', '你会用中文进行交流么', '我可以和你说中文么',
                     '请说明是否支持中文聊天', '想用中文与你交流可以么', '你属于中文聊天助手么',
                     '能给我中文的回答么', '这个助手用中文交流么', '能用中文解释基础知识么',
                     '可以使用中文讨论日常事情么'], '可以，我是一个中文助手，可以用文字和你交流。'),
    }
    prefixes=['','请问，','朋友，','助手，','我想了解一下，','打扰一下，']
    for family,(questions,answer) in identities.items():
        for q in questions:
            for prefix in prefixes:
                yield dict(user=prefix+q+'？',assistant=answer,source='authored-identity-v5',family=family)
    people=['我','同学','老师','爸爸','妈妈','朋友','小明','小红']
    actions=['喜欢读故事','喜欢画画','正在做笔记','准备去公园','今天去看望家人','想学会做饭']
    copying=['只把引号里的文字写一遍：“{}”','照抄下列句子，保持原文不变：{}',
             '请复制这段文字一次，不添加解释：{}','把下面的文字原样作为回答：{}']
    for person in people:
        for action in actions:
            sentence=person+action+'。'
            for pattern in copying:
                yield dict(user=pattern.format(sentence),assistant=sentence,source='authored-copy-v5')
    keywords={'春天':'春天来了，花朵慢慢开放。','夏天':'夏天来了，树叶长得很茂盛。',
              '秋天':'秋天到了，树叶渐渐变黄。','冬天':'冬天很冷，出门记得穿暖。',
              '阳光':'阳光照进了明亮的房间。','雨水':'雨水轻轻落在屋顶上。',
              '书本':'书本整齐地放在书架上。','公园':'公园里有许多绿色的树。',
              '学习':'认真学习可以获得新的知识。','朋友':'朋友之间可以互相帮助。',
              '小鸟':'小鸟在树枝上唱歌。','电脑':'我用电脑整理文字。',
              '花朵':'花朵在清晨慢慢开放。','月亮':'月亮静静地挂在夜空里。',
              '风':'风轻轻吹动了窗帘。','教室':'教室里摆着整齐的桌椅。'}
    for word,answer in keywords.items():
        for pattern in ['写一个含有“{}”的句子。','用词语{}造一个简短的句子。',
                        '请给出一句话，必须使用{}这个词。','造句时要包含{}，只写一句。',
                        '以{}为关键词写一句短话。','请写出包含{}的简短句子。']:
            yield dict(user=pattern.format(word),assistant=answer,source='authored-keyword-v5')
    for animal in ['小狗','小猫','小兔子','小鸟','小乌龟']:
        for prefix in prefixes:
            for pattern in ['想给{}取名字，给一个中文的建议。','请替{}想一个中文名字。',
                            '养了一只{}，可以取什么中文名字？']:
                yield dict(user=prefix+pattern.format(animal),assistant='可以叫它“小白”。',source='authored-naming-v5')
    topics=[
        ('冰块受到热量','冰块吸收热量后会融化成水。'),
        ('提前安排第二天早起','可以先收拾物品、设置闹钟，然后早点睡觉。'),
        ('开始收拾杂乱的桌子','先清走不用的东西，再把书本、文具分别放好。'),
        ('下雨时外出带的物品','带上伞或雨衣，并注意湿滑路面。'),
        ('每天坚持阅读的方法','留出固定的阅读时间，先从感兴趣的书读起。'),
        ('与新朋友开始聊天','先友善地打个招呼，再聊一个双方都感兴趣的话题。'),
    ]
    for topic,answer in topics:
        for prefix in prefixes:
            for pattern in ['关于{}，请给我一个简短说明。','想了解{}，怎样理解比较好？',
                            '谈谈{}，一句话就行。','给初学者讲讲{}。']:
                yield dict(user=prefix+pattern.format(topic),assistant=answer,source='authored-basic-v5')


def main():
    target=DATA/'processed/sft-v5'
    if target.exists():raise ValueError('v5 exists; preserve immutable data')
    previous=DATA/'processed/sft-v4'
    tok=Tokenizer.from_file(str(DATA/'tokenizers/vocab-32000/tokenizer.json'))
    valid=load_jsonl(previous/'validation.jsonl')
    seen={normalize(r['user']) for r in valid}
    blocked=excluded_questions()
    train=[];rejected=Counter()
    def add(r,split='train'):
        r=dict(r,user=clean(r['user']),assistant=clean(r['assistant']))
        key=normalize(r['user'])
        if key in seen or key in blocked:
            rejected['duplicate_validation_or_suite']+=1;return
        if encode_pair(tok,r['user'],r['assistant'],256) is None:
            rejected['length']+=1;return
        seen.add(key);r.update(split=split,question_sha256=digest(key))
        (train if split=='train' else valid).append(r)
    for r in new_basics():
        add(r,'validation' if int(digest(normalize(r['user']))[:8],16)%100<2 else 'train')
    for r in load_jsonl(previous/'train-unique.jsonl'):
        if r['source'].startswith('authored-'):add(r)
    # 扩大覆盖而不重用验证问题；同一OASST树划分继承v3。
    for r in load_jsonl(DATA/'processed/sft-v3/train-unique.jsonl'):
        if r['source'].startswith('authored-'):continue
        if not public_ok(r) or len(normalize(r['assistant']))<4:
            rejected['public_filter']+=1;continue
        # 数值计算只用已核算的自建课程，避免合成数学监督冲突。
        if re.search(r'\d\s*[+*/×÷=]|计算|乘以|除以|平方根',r['user']):
            rejected['public_calculation']+=1;continue
        add(r)
    arithmetic=[r for r in train if r['source']=='authored-arithmetic-v4']
    basics=[r for r in train if r['source'].startswith('authored-') and r['source']!='authored-arithmetic-v4']
    public=[r for r in train if not r['source'].startswith('authored-')]
    # 用整数倍近似平衡基础课程和公共问答；实际比例以清单为准，重复仅训练侧。
    arithmetic_multiplier=max(1,round(len(public)/2/len(arithmetic)))
    basic_multiplier=max(1,round(len(public)/len(basics)))
    weighted=train+arithmetic*(arithmetic_multiplier-1)+basics*(basic_multiplier-1)
    tg={r['arithmetic_group'] for r in train if 'arithmetic_group' in r}
    vg={r['arithmetic_group'] for r in valid if 'arithmetic_group' in r}
    assert not tg&vg
    assert not {normalize(r['user']) for r in train}&{normalize(r['user']) for r in valid}
    assert not {normalize(r['user']) for r in train+valid}&blocked
    target.mkdir(parents=True)
    save_jsonl(target/'train-unique.jsonl',train);save_jsonl(target/'validation.jsonl',valid)
    for split,rows in [('train',weighted),('validation',valid)]:
        save_jsonl(target/f'{split}-tokens.jsonl',[dict(encode_pair(tok,r['user'],r['assistant'],512),source=r['source'],question_sha256=r['question_sha256']) for r in rows])
    result=dict(created_utc=utc(),parent_manifest_sha256=sha256(previous/'manifest.json'),unique_train=len(train),
                weighted_train=len(weighted),validation=len(valid),inherited_v4_validation=1239,
                arithmetic_multiplier=arithmetic_multiplier,basic_multiplier=basic_multiplier,
                sources_unique_train=dict(Counter(r['source'] for r in train)),weighted_sources=dict(Counter(r['source'] for r in weighted)),
                rejected=dict(rejected),question_split_disjoint=True,arithmetic_group_split_disjoint=True,
                suite_exact_questions_excluded=True,train_sha256=sha256(target/'train-tokens.jsonl'),
                validation_sha256=sha256(target/'validation-tokens.jsonl'),
                policy='Broader public coverage and general identity/copy/keyword/naming curriculum; inherited arithmetic split immutable; no final-answer tuning; public filtering is heuristic')
    write_json(target/'manifest.json',result);write_json(RUNS/'reports/sft-v5.json',result)
    print(result)


if __name__=='__main__':main()
