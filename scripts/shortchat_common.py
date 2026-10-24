"""Frozen short Chinese chat policies; no GPU or model imports."""
from collections import Counter
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from scripts.common import ROOT, read_json, sha256, write_json

CATEGORIES = {'daily_single':50000,'daily_multi':25000,'instruction':10000,
              'boundary':5000,'concept':5000,'basic':5000}
TASKS = {'daily_single':'daily','daily_multi':'daily','instruction':'instruction',
         'boundary':'identity','concept':'concept','basic':'arithmetic'}
SPLITS = {'train':100000,'validation':5000,'dev':200,'test':1000}
SYSTEM = '你是LubanLM，一个中文助手。'

def digest(value):
    return hashlib.sha256((value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,sort_keys=True)).encode()).hexdigest()

def norm(text):
    return re.sub(r'\s+','',unicodedata.normalize('NFKC',text)).lower()

def skeleton(text):
    return re.sub(r'\d+(?:\.\d+)?','#',norm(text))

def partition(family):
    n=int(digest(family)[:8],16)%10000
    return 'train' if n<8800 else 'validation' if n<9600 else 'dev' if n<9670 else 'test'

def repetition(text):
    from eval.chat_suite import repeated
    if repeated(text): return True
    parts=[norm(re.sub(r'^\s*(?:[-*•]|\d+[.、．)]|[一二三四五六七八九十]+[、．.])\s*','',s))
           for s in re.split(r'[\n。！？!?]+',text)]
    return any(n>=3 for s,n in Counter(s for s in parts if len(s)>=6).items())

def classify(messages, filename=''):
    # Classify the outer request, never quoted text inside its input. A complete
    # conversation keeps the intent of its initial exchange.
    q=messages[0]['content']; first=q
    if filename in ('rewrite.parquet','editing.parquet','format-constrain.parquet','summary.parquet'):return 'instruction'
    outer=re.split(r'(?:原文(?:本)?|文本内容|给定文本|输入内容|材料|待处理文本)\s*[:：]',q,maxsplit=1)[0]
    if re.search(r'改(?:写|成|为|进|正)|调整.{0,12}(?:文字|文本|句子|段落|内容|表达|文风)|(?:文字|文本|句子|段落|内容).{0,12}调整|语法.{0,8}(?:改|校)|重写|简化|口语化|修辞|提高清晰度|连贯性|编辑|添加.{0,8}结尾|(?:写|撰写|生成|编写|创作).{0,18}(?:文章|故事|诗|文案|邮件|求职|自我介绍|介绍自己|短文|信|段落|简历)|翻译|润色|总结|概括|提取|分类|照抄|排序|校对|检查.{0,18}(?:错误|拼写|语法)|阅读.{0,18}(?:文章|段落|文本)|根据.{0,18}(?:输入|描述).{0,18}(?:生成|识别)|询问.{0,8}(?:姓名|人名|年龄)',outer):return 'instruction'
    if re.search(r'(?:身份|姓名|公司名称|职位名称)\s*[:：]',q) and re.search(r'介绍|简历|段落|陌生人',outer):return 'instruction'
    if direct_boundary(outer):return 'boundary'
    if re.search(r'计算|算一下|等于多少|\d\s*[+×*/÷−-]\s*\d',q):return 'basic'
    if re.search(r'提取|分类|照抄|只(?:回答|输出|给)|排序|改写|修改|润色|翻译|总结|概括|写.{0,5}(?:文章|故事|诗|文案|邮件)|标点|校对',q):return 'instruction'
    if re.search(r'什么是|什么叫|原理|区别|是什么|解释|定义|学科|艺术史|历史|科学|物理|化学',q):return 'concept'
    if '为什么' in q and not re.search(r'我|心情|朋友|休息|学习|生活|工作|难过|累|烦|紧张',q):return 'concept'
    everyday = filename in ('everyday.parquet','advice-seeking.parquet','planning.parquet')
    cues = re.search(r'心情|朋友|休息|周末|学习|整理|书包|桌面|生日|道歉|感谢|聊天|难过|高兴|安慰|沟通|生活|工作|怎么办|建议|无聊|紧张|累|烦|放松|你好|最近|今天',first)
    if not (everyday or cues):return None
    return 'daily_multi' if len(messages)>=4 else 'daily_single'


def direct_boundary(question):
    q=question.strip()
    if not q:return False
    if re.fullmatch(r'(?:你好|您好|早上好|早安|晚安|谢谢|再见)[！!。 ，,]*',q):return True
    # Quoted dialogue, invitations and someone else's self-introduction are not
    # questions about this assistant's capabilities.
    if re.search(r'写作|故事|邮件|求职|结尾|原文|文本|引用|扮演|假设你|假装你|[「“\"]|(?:姓名|身份|公司|职位)\s*[:：]',q):return False
    # Attached source material must never supply the subject of the request.
    q=q.splitlines()[0]
    q=re.sub(r'^(?:你好[，,！! ]*|请问[，,：: ]*|请(?:简单|简短|用一句话)?|用一句话|麻烦你)', '',q)
    if re.fullmatch(r'(?:请问[，, ]*)?你(?:能|能够|可以|能不能|能否)(?:帮我|帮助我)(?:回答(?:一个)?问题|整理文字|处理文字|吗)[吗?？。 ]*',q):return True
    return bool(re.match(r'你是谁|你叫什么|你的名字|你是什么.{0,8}(?:助手|模型|AI|人工智能)|你(?:的|有什么|有哪些|主要有什么|主要有哪些)(?:作用|功能|能力)|你(?:能做什么|能够做什么)|(?:介绍|说明).{0,6}(?:你自己|自己)|你(?:是人|是机器人|是AI|是人工智能|有(?:自己的|一个)?身体|有没有身体)|你(?:能|能够|能不能|是否能|能否|可以|可不可以|会不会|现在能)(?:够|够不够)?(?:直接|实际|真的|亲自|自行|自己|替我|联网|上网|看到|看见|听见|感知|打开摄像头|到现场)|你(?:现在|目前)?(?:能|能够|能否|可以)(?:直接)?(?:查询网络|联网|上网)',q))

def invalid(messages, category):
    if not messages or len(messages)%2 or len(messages)>8:return 'roles'
    for i,m in enumerate(messages):
        if m.get('role')!=('user' if i%2==0 else 'assistant') or not isinstance(m.get('content'),str) or not m['content'].strip():return 'roles'
    text='\n'.join(m['content'] for m in messages)
    if '\ufffd' in text or re.search(r'<(?:\||/?think|bos|eos)|[\x00-\x08\x0b\x0c\x0e-\x1f]',text):return 'encoding'
    if re.search(r'颅骨|脊髓|诊断|用药|处方|养生|保健|血糖|糖尿|治疗|抗癌|癌症|蜂蜜|咳嗽|助眠|抑郁症|焦虑症|投资收益|股票|大学物理|量子|角色扮演|请扮演|假装你|作为一名',text):return 'specialized_or_roleplay'
    for m in messages[1::2]:
        a=m['content']
        if repetition(a):return 'repetition'
        if re.search(r'(?:我是|我叫|我的名字是|由.{0,12}(?:开发|训练)).{0,15}(?:MOSS|Qwen|通义|DeepSeek|ChatGPT|OpenAI|MiniMind|复旦)',a,re.I):return 'identity'
        if re.search(r'我(?:的朋友|的家人|父母|家里|住在|昨天|今天).{0,12}(?:吃|喝|去|买|住|上班)|我(?:去过|吃过|喝过|买了|陪你去|来你家)|我们(?:一起去|去公园|见面)',a):return 'physical_persona'
        if re.search(r'我(?:可以|能|会|将).{0,12}(?:到你家|到现场|替你去|帮你买|实时查询|打开摄像头|给你打电话)',a):return 'unsupported_capability'
    if category=='instruction' and re.search(r'以下|给定|这段|输入|提取|分类|照抄|改写|翻译',messages[-2]['content']) and not re.search(r'[:：\n“"「]',messages[-2]['content']):return 'missing_input'
    if category=='basic':
        from scripts.corpus_quality import arithmetic
        if arithmetic(messages[-2]['content'],messages[-1]['content']) is not True:return 'unverified_basic'
    return None

def validate_review(samples_path, marks_path, artifacts):
    samples=read_json(samples_path); marks=read_json(marks_path)
    if marks.get('samples_sha256')!=sha256(samples_path) or marks.get('artifacts')!=artifacts:
        raise ValueError('Review does not match frozen samples and data')
    items=marks.get('marks',[]); expected={r['id'] for r in samples}
    if len(items)!=len(expected) or {r['id'] for r in items}!=expected:raise ValueError('Incomplete review')
    for m in items:
        if any(type(m.get(k)) is not bool for k in ('severe_error','category_correct','usable')) or not m.get('reason'):raise ValueError('Explicit review decisions required')
    for split in ('train','validation'):
        ids={r['id'] for r in samples if r['split']==split}; selected=[m for m in items if m['id'] in ids]
        if sum(m['severe_error'] for m in selected) or sum(m['category_correct'] and m['usable'] for m in selected)/len(selected)<.95:
            raise ValueError('Content review gate failed')
    return dict(samples=len(items),severe_errors=0,status='review_passed',review_sha256=sha256(marks_path))

def weighted_score(rows):
    result={}
    for category,n in CATEGORIES.items():
        subset=[r for r in rows if r['category']==category]
        if not subset:raise ValueError('Missing comparison category')
        for r in subset:
            g=r.get('grading') or {}
            if any(type(g.get(k)) is not bool for k in ('relevant_readable','correct','looping')) or not g.get('reason'):
                raise ValueError('Semantic grading incomplete')
        result[category]=sum(r['grading']['relevant_readable'] and r['grading']['correct'] and not r['grading']['looping'] for r in subset)/len(subset)
    return dict(score=sum(CATEGORIES[k]/100000*v for k,v in result.items()),by_category=result,
        loop_rate=sum(r['grading']['looping'] for r in rows)/len(rows),
        eos_rate=sum(r['stop_reason']=='eos' for r in rows)/len(rows),
        replacement_count=sum(r['replacement_character'] for r in rows))
