"""固定题目与评分约定。评分需要逐题阅读原始输出，不用关键词冒充语义验收。"""
from __future__ import annotations
from scripts.common import ROOT, RUNS, write_json, read_json, sha256, utc, GPUBudget, require_gpu
import argparse
import re
import unicodedata
from pathlib import Path

CATEGORIES = ['greeting', 'identity', 'daily', 'concept', 'instruction', 'arithmetic']
DEV = [
    ['早上好！', '晚上好呀。', '很高兴见到你。', '谢谢你的帮助。', '我先去忙了，再见。', '今天心情不错，想和你打个招呼。', '你可以陪我说句话吗？', '初次见面，请多关照。', '下午好，朋友。', '我来找你聊聊天。'],
    ['介绍一下你自己。', '你叫什么名字呢？', '你能帮我做些什么？', '你是人还是程序？', '你有自己的身体吗？', '我该怎么称呼你？', '你是一个中文助手吗？', '你能够帮我理解简单概念吗？', '你可以回答日常问题吗？', '说说你的身份。'],
    ['我有点累，应该怎么休息？', '明天要早起，今晚怎么安排？', '桌面很乱，我该从哪里开始整理？', '下雨了，出门要带什么？', '我想每天读书，你有什么小建议？', '朋友帮了我，我应该怎么表达感谢？', '出门前怎么检查有没有忘带东西？', '学习时总分心，给我一个简单建议。', '我准备散步，有什么要注意的？', '我想给朋友写句生日祝福。'],
    ['用一句话解释什么是电脑。', '键盘通常用来做什么？', '鼠标有什么作用？', '浏览器是用来干什么的？', '简单说说什么是互联网。', '什么叫文件夹？', '为什么要给手机充电？', '雨是怎么形成的，简单讲一下。', '植物为什么需要阳光？', '冰加热后会怎样？'],
    ['请只回答一个字：好。', '列出两种常见水果。', '把“我喜欢学习”原样写一遍。', '请用“春天”写一句短句。', '给小狗起一个中文名字。', '说出红色和蓝色这两种颜色。', '给我一句鼓励学习的话。', '用一句话向老师问好。', '请回答“谢谢”，不要解释。', '说出一种常见交通工具。'],
    ['2+3等于几？', '7减4是多少？', '3乘2的结果是什么？', '8除以2等于多少？', '5加4是多少？', '9减3得到几？', '4乘3等于多少？', '12除以3的结果是什么？', '6加1等于几？', '10减8是多少？'],
]
TEST = [
    ['嗨，见到你很开心。', '今天第一次和你聊天，向你问个好。', '祝你有个愉快的上午。', '辛苦啦，多谢你。', '我要出门了，回头聊。', '哈喽，来聊一句吧。', '刚吃完午饭，向你打个招呼。', '收到你的帮助了，感谢！', '准备睡觉了，晚安。', '很高兴认识这位新朋友。'],
    ['请告诉我你的名称。', '你属于什么类型的助手？', '用一句话说明你的身份。', '你是真人吗？', '你能像人一样走路吗？', '我不知道该如何称呼你，请告诉我。', '你的主要作用是什么？', '请简单介绍你可以提供的帮助。', '你会用中文和我交流吗？', '我能请你解释一些基础知识吗？'],
    ['忙了一天想放松，给我一个建议。', '我明早六点起床，睡前应做什么准备？', '书桌上东西太多，如何开始收拾？', '外面正在下雨，我出门应拿上什么？', '我想养成阅读习惯，建议怎么开始？', '同学借给我一支笔，我应该说什么？', '准备离家时，怎样确认必需品带齐了？', '做作业时注意力不集中，怎么调整？', '第一次去公园走走，给我一个提醒。', '同学生日到了，请写一句祝福。'],
    ['电脑是一种什么工具？', '输入文字通常会使用电脑的哪个设备？', '操作电脑时，鼠标能帮助我们做什么？', '如果要访问网页，应该使用哪类软件？', '互联网可以帮助人们做什么？', '电脑里为什么要使用文件夹？', '手机电池快没电了，要怎么处理？', '天空中的雨水大致来自哪里？', '阳光对植物生长有什么帮助？', '把冰放在温暖处，通常会发生什么？'],
    ['只写“可以”两个字作为回答。', '请说出两种蔬菜的名称。', '把这句话照抄一遍：今天很开心。', '写一句含有“秋天”的短句。', '帮一只小猫取个中文名字。', '请列出绿色和黄色。', '给正在读书的同学一句鼓励。', '用一句短句向同学打招呼。', '直接输出“不客气”三个字。', '举一个公共交通工具的例子。'],
    ['4+5的答案是多少？', '8-3等于几？', '2×4的结果是多少？', '9÷3得几？', '3+6算出来是多少？', '10-4的答案是什么？', '5×2等于多少？', '12÷4等于几？', '7+2的计算结果？', '9-7是多少？'],
]
CORE = [('你好','greeting'), ('你是谁','identity'), ('Linux是什么','concept'), ('操作系统是什么','concept'), ('1+2是多少','arithmetic')]

def normalize(text):
    text=unicodedata.normalize('NFKC', text).replace('×','*').replace('÷','/').replace('−','-')
    # 运算符是语义：1+2 与 1-2 不能合并为同一个问题。
    return ''.join(c.lower() for c in text if c.isalnum() or c in '+-*/=')

def suite(split):
    groups = DEV if split == 'dev' else TEST
    return [dict(id=f'{split}-{category}-{i:02d}', category=category, question=q) for category, questions in zip(CATEGORIES, groups) for i,q in enumerate(questions)]

def excluded_questions():
    return {normalize(r['question']) for split in ('dev','test') for r in suite(split)} | {normalize(q) for q,_ in CORE}

def load_suite(path,split):
    definition=read_json(path)
    rows=definition['dev' if split=='sampling' else split]
    if len(rows)!=(5 if split=='core' else 60) or len({r['id'] for r in rows})!=len(rows):
        raise ValueError('custom frozen suite has incomplete/duplicate IDs')
    if split!='core' and any(sum(r['category']==c for r in rows)!=10 for c in CATEGORIES):
        raise ValueError('custom frozen suite must contain ten questions per category')
    if any(not r['question'].strip() or r['category'] not in CATEGORIES for r in rows):
        raise ValueError('invalid custom suite row')
    return rows

def repeated(text):
    text = normalize(text)
    # 至少 3 次连续重复、片段长至少 3 个字符；最终仍需阅读核查。
    return bool(re.search(r'(.{3,40}?)\1\1', text))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--split', choices=['dev','test','core','sampling'], default='dev')
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--minutes', type=float, default=25)
    p.add_argument('--suite-file',type=Path)
    args = p.parse_args()
    if args.output.exists():
        p.error('output exists; preserve previous raw answers')
    from inference.generate import load_model, complete
    from sft.template import prompt
    import torch
    rows = [dict(id=f'core-{i}',category=c,question=q) for i,(q,c) in enumerate(CORE)] if args.split == 'core' else suite('dev' if args.split == 'sampling' else args.split)
    if args.suite_file:rows=load_suite(args.suite_file,args.split)
    if args.split == 'sampling':
        rows = rows[:4] + rows[10:14] + rows[20:24] + rows[30:34] + rows[40:42] + rows[50:52]
    results = []
    with GPUBudget('chat-evaluation-'+args.split,args.minutes) as budget:
        model, tokenizer = load_model(args.checkpoint, require_gpu())
        for row in rows:
            for seed in ([2026,2027,2028] if args.split == 'sampling' else [2026]):
                if budget.remaining() < 15:
                    raise RuntimeError('evaluation budget exhausted; do not accept partial suite')
                torch.manual_seed(seed)
                answer, ids = complete(model, tokenizer, prompt(row['question']), max_new_tokens=96, temperature=.7 if args.split == 'sampling' else 0., top_k=40 if args.split == 'sampling' else 0, top_p=.9 if args.split == 'sampling' else 1., repetition_penalty=1.1 if args.split == 'sampling' else 1.)
                results.append(dict(row, seed=seed, answer=answer, generated_ids=ids, eos=bool(ids and ids[-1] == model.config.eos_token_id), repeated_flag=repeated(answer), grading={'relevant_readable': None,'correct': None,'looping': None,'reason': ''}))
                print(row['id'], seed, answer.replace('\n',' '), flush=True)
                budget.tick()
    write_json(args.output, {'created_utc':utc(),'checkpoint':str(args.checkpoint),'checkpoint_sha256':sha256(args.checkpoint),'split':args.split,'suite_sha256':sha256(Path(__file__)),
                            'suite_file':str(args.suite_file) if args.suite_file else None,
                            'suite_definition_sha256':sha256(args.suite_file) if args.suite_file else None,
                            'status':'awaiting_review','results':results})

if __name__ == '__main__':
    main()
