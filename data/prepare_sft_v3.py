"""独立保存修正去重与短问答平衡数据，不覆盖 v2 或固定 probe。"""
from scripts.common import DATA, RUNS, write_json, sha256, utc
from data.prepare import load_jsonl,save_jsonl,clean
from data.prepare_chat_v2 import authored_pairs
from eval.chat_suite import normalize,excluded_questions
from sft.template import encode_pair
from tokenizers import Tokenizer
import hashlib
import re
from collections import Counter

def extra_instruction_pairs():
    # 新的通用短指令训练文本；不直接使用开发题/验收题的问题与期望句子。
    sentences=['清晨的空气很清新','小河静静流过树林','山间的风很清凉','我把书放回了书架','窗外的小鸟正在唱歌','我们一起保持教室整洁','周末我打算去看望家人','一杯温水放在桌子上','阅读让我学到新的知识','做完事情记得收好工具']
    rows=[]
    for text in sentences:
        for q in [f'请照原文输出以下句子：{text}',f'把下面这句话抄一次，不增加内容：{text}']:
            rows.append(dict(user=q,assistant=text,source='authored-instruction-v3'))
    for word in ['明白','收到','欢迎','早安','晚安','加油','平安','顺利','快乐','完成']:
        for q in [f'只输出这两个字：{word}',f'不要添加解释，直接回复：{word}']:
            rows.append(dict(user=q,assistant=word,source='authored-instruction-v3'))
    facts=[('两种水果','苹果、香蕉。'),('三种水果','苹果、香蕉、梨。'),('两种蔬菜','白菜、胡萝卜。'),('三个颜色名称','红色、绿色、蓝色。'),('一个宠物名字','小白。'),('两种交通工具','公交车、地铁。'),('一句学习鼓励','慢慢来，坚持练习，你会有所进步。'),('一句生日祝愿','祝你生日快乐，每一天都开心。'),('一句向老师问候的话','老师您好，祝您今天愉快。'),('一句向朋友问候的话','朋友你好，很高兴和你见面。')]
    for kind,a in facts:
        for template in ['举出{}，回答简短。','请给出{}。','我想要{}，请直接写出来。','写下{}就可以。']:
            rows.append(dict(user=template.format(kind),assistant=a,source='authored-instruction-v3'))
    daily=[('雨天外出','带上雨伞或雨衣，注意路面湿滑。'),('休息恢复精力','可以暂停手头的事情，喝点水，适当休息。'),('提前准备早起','提前整理好要带的东西，设好闹钟，并早点休息。'),('整理书桌','先清理不需要的东西，再把书本和工具分类放好。'),('养成阅读习惯','每天留出一点时间，从喜欢的书开始读。'),('向帮助你的人致谢','真诚地说声谢谢，并说明对方的帮助对你很有用。'),('出门检查物品','列一个简短清单，核对钥匙、手机和需要的物品。'),('集中注意力学习','先把手机等干扰物放远，一次专心完成一个小任务。'),('户外散步','穿舒适的鞋，留意天气，并注意交通安全。'),('准备睡觉','放下电子设备，放松心情，按时休息。')]
    for kind,a in daily:
        for template in ['关于{}，给我一条实用建议。','怎样{}比较合适？','请简单谈谈{}。','我想了解{}，你有什么建议？']:
            rows.append(dict(user=template.format(kind),assistant=a,source='authored-daily-v3'))
    return rows

def main():
    target=DATA/'processed/sft-v3'
    manifest=target/'manifest.json'
    if manifest.exists():
        raise ValueError('v3 exists; preserve versioned data')
    target.mkdir(parents=True,exist_ok=True)
    tok=Tokenizer.from_file(str(DATA/'tokenizers/vocab-32000/tokenizer.json'))
    blocked=excluded_questions()
    seen=set()
    rows={'train':[],'validation':[]}
    raw={'train':[],'validation':[]}
    rejected=Counter()
    def add(r,forced=None):
        q,a=clean(r['user']),clean(r['assistant'])
        canonical=normalize(q)
        if canonical in blocked:
            rejected['suite_exact_question']+=1;return
        if not canonical or canonical in seen:
            rejected['duplicate']+=1;return
        pair=encode_pair(tok,q,a,512)
        if pair is None:
            rejected['length']+=1;return
        key=hashlib.sha256(canonical.encode()).hexdigest()
        split=forced or ('validation' if int(key[:8],16)%100<2 else 'train')
        seen.add(canonical)
        rows[split].append(dict(pair,source=r['source'],tree_id=r.get('tree_id',key),message_id=r.get('message_id',key),question_sha256=key))
        raw[split].append(dict(r,user=q,assistant=a,split=split,question_sha256=key))
    # 来源 OASST 在 v2 中已经做原始树隔离；优先验证侧，保持这一划分。
    public=[]
    for split in ('validation','train'):
        for r in load_jsonl(DATA/f'processed/sft-{split}.jsonl'):
            if r['source']=='oasst1':
                add(r,split)
            elif r['source']=='belle-synthetic':
                public.append(r)
    for r in authored_pairs()+extra_instruction_pairs():
        add(r)
    for r in public:
        q,a=r['user'],r['assistant']
        if len(q)>180 or len(a)>160 or a.startswith(('抱歉','很抱歉','对不起','很遗憾','我无法','我不能')) or q.strip()==a.strip() or re.search(r'(.{3,30}?)\1\1',a):
            rejected['constructive_short_filter']+=1;continue
        if re.search(r'上面|上述|这篇|这个数据|音频|视频|今天.*天气|最新.*新闻',q):
            rejected['context_or_live_request']+=1;continue
        add(r)
    unique_train=rows['train']
    authored=[r for r in unique_train if r['source'].startswith('authored-')]
    # 让基础短问答约占 20%；重复仅在训练侧，不重复制备验证题。
    multiplier=max(1,round(.2*(len(unique_train)-len(authored))/(.8*len(authored))))
    weighted=unique_train+authored*(multiplier-1)
    save_jsonl(target/'train-tokens.jsonl',weighted)
    save_jsonl(target/'validation-tokens.jsonl',rows['validation'])
    save_jsonl(target/'train-unique.jsonl',raw['train'])
    save_jsonl(target/'validation.jsonl',raw['validation'])
    train_keys={r['question_sha256'] for r in rows['train']}
    val_keys={r['question_sha256'] for r in rows['validation']}
    assert not train_keys & val_keys
    assert normalize('请计算1+2')!=normalize('请计算1-2')
    assert not {normalize(r['user']) for side in raw.values() for r in side}&blocked
    arithmetic=[r for r in raw['train'] if r['source']=='authored-arithmetic']
    assert any('-' in r['user'] for r in arithmetic)
    result=dict(created_utc=utc(),normalization='NFKC; preserve arithmetic operators; ×/÷ mapped to */',unique_train=len(unique_train),weighted_train=len(weighted),validation=len(rows['validation']),authored_multiplier=multiplier,authored_weighted_fraction=len(authored)*multiplier/len(weighted),arithmetic_train_operations={'addition':sum('+' in r['user'] for r in arithmetic),'subtraction':sum('-' in r['user'] for r in arithmetic),'multiplication':sum('乘' in r['user'] for r in arithmetic)},sources_unique_train=dict(Counter(r['source'] for r in rows['train'])),rejected=dict(rejected),question_split_disjoint=True,suite_exact_questions_excluded=True,train_sha256=sha256(target/'train-tokens.jsonl'),validation_sha256=sha256(target/'validation-tokens.jsonl'),policy='v2 and fixed probe unchanged; v3 for later full SFT only')
    write_json(manifest,result)
    write_json(RUNS/'reports/sft-v3.json',result)
    print(result)

if __name__=='__main__':
    main()
