"""Cheap, reproducible teaching-corpus rules. No generative judge or repair."""
from scripts.common import ROOT, read_json, sha256
from scripts.prepare_general_2b import valid_text, key
from eval.chat_suite import normalize, repeated
import re

VERSION = 'teaching-clean-v2'

def policy_identity(plan_path):
    files = ['scripts/corpus_quality.py','scripts/prepare_general_10b.py',
             'scripts/prepare_general_2b.py','scripts/prepare_sft_100k.py',
             'sft/template.py','data/prepare.py','eval/chat_suite.py','configs/known-bad-corpus.json','scripts/balanced_sft.py',
             'scripts/content_review_gate.py','scripts/run_general_10b.py','training/batches.py','training/train.py']
    return dict(version=VERSION, plan_sha256=sha256(plan_path),
                code={f:sha256(ROOT/f) for f in files})

def rejected_groups():
    result=set()
    for name in ('sft-review-decisions.json','sft-review-final-decisions.json'):
        path=ROOT/'docs/run-results/general-10b-20261005'/name
        if path.exists(): result.update(r['group'] for r in read_json(path).get('rejected',[]))
    path=ROOT/'configs/known-bad-corpus.json'
    if path.exists():result.update(read_json(path).get('sft_groups',[]))
    return result

def clean_document(text):
    original=text
    # Only remove isolated, clearly recognisable navigation/footer lines.
    lines=[line for line in text.splitlines() if not re.fullmatch(
        r'\s*(?:返回首页|返回顶部|上一篇|下一篇|相关阅读|责任编辑|点击收藏|分享至)[：:].{0,120}',line)]
    text='\n'.join(lines)
    if re.search(r'_ths_|\}\)\)|港澳发据中国消费者|改良成為，|f t re|str ct ral|l991|6o%',text):
        return None,'known_scrape_defect'
    if ('舒曼' in text and '1948' in text) or ('马斯特里赫特' in text and '单一欧洲' in text):
        return None,'known_factual_error'
    if re.search(r'(?:广告位招租|点击领取|扫码领取|购买链接|加入购物车)',text):
        return None,'commercial_noise'
    cleaned,reason=valid_text(text)
    return cleaned,reason

def arithmetic(question,answer):
    import ast
    from fractions import Fraction
    if not re.search(r'计算|算|等于|结果|答案|[=?？]',question):return None
    chunks=re.findall(r'[\d.()+×*/÷−-][\d.\s()+×*/÷−-]*',question)
    chunks=[x.strip() for x in chunks if re.search(r'\d\s*[+×*/÷−-]|\)\s*[+×*/÷−-]',x)]
    if len(chunks)!=1 or len(chunks[0])>100:return None
    expression=chunks[0].replace('×','*').replace('÷','/').replace('−','-')
    def calculate(node):
        if isinstance(node,ast.Constant) and type(node.value) in (int,float):return Fraction(str(node.value))
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.UAdd,ast.USub)):
            v=calculate(node.operand);return -v if isinstance(node.op,ast.USub) else v
        if isinstance(node,ast.BinOp) and isinstance(node.op,(ast.Add,ast.Sub,ast.Mult,ast.Div)):
            a,b=calculate(node.left),calculate(node.right)
            if isinstance(node.op,ast.Add):return a+b
            if isinstance(node.op,ast.Sub):return a-b
            if isinstance(node.op,ast.Mult):return a*b
            return a/b
        raise ValueError('unsupported arithmetic')
    try:expected=calculate(ast.parse(expression,mode='eval').body)
    except ZeroDivisionError:return False
    except (ValueError,SyntaxError,RecursionError):return None
    number=r'(-?\d+(?:\.\d+)?(?:/\d+)?)(?!\d|[./]\d)'
    explicit=re.findall(r'(?:=|等于|结果(?:是|为)?|答案(?:是|为)?|得)\s*[：:]?\s*'+number,answer)
    if not explicit:
        match=re.fullmatch(r'\s*'+number+r'\s*[。.!！]?\s*',answer)
        if not match:return None
        explicit=[match[1]]
    try:return all(Fraction(v)==expected for v in explicit)
    except (ValueError,ZeroDivisionError):return False

def verified_arithmetic_dialogue(messages):
    return bool(messages) and all(arithmetic(messages[i-1]['content'],messages[i]['content']) is True
        and not re.search(r'[A-Za-z]',messages[i-1]['content']) for i in range(1,len(messages),2))

def dialogue_reason(messages, task):
    text='\n'.join(m['content'] for m in messages)
    if any(x in text for x in ('\ufffd','<think>','</think>','<|','<bos>','<eos>')) or repeated(text):return 'control_or_repeat'
    if any(ord(c)<32 and c not in '\n\t' for c in text):return 'control_character'
    if sum('\u4e00'<=c<='\u9fff' for c in text)/max(1,len(re.sub(r'\s','',text)))<.4 and not (task=='arithmetic' and verified_arithmetic_dialogue(messages)):return 'language'
    if re.search(r'最新|近期.{0,6}新闻|实时股价|投资收益|诊断|用药|处方|疗效|临床|颅骨|枕骨|脊髓|量子霸权|肿瘤',text):return 'specialized_or_unstable'
    if re.search(r'扮演|假装|假设你|你的角色|角色是',text):return 'persona'
    if re.search(r'(?:我是|我叫|由.{0,12}(?:开发|训练)).{0,20}(?:Qwen|通义|DeepSeek|深度求索|ChatGPT|OpenAI|MiniMind|夸克)',text,re.I):return 'wrong_identity'
    if re.search(r'2/5的较大值|工作[：:]\s*2\s*\+\s*2',text):return 'known_bad_question'
    for i in range(1,len(messages),2):
        question,answer=messages[i-1]['content'],messages[i]['content']
        if arithmetic(question,answer) is False:return 'arithmetic_not_verified'
        if re.search(r'(?:不会|绝不会|不可能).{0,12}(?:损失|盗刷)',answer):return 'absolute_financial_claim'
        if re.search(r'亚马逊.{0,12}(?:最大|最大的).{0,10}森林',question) and '反驳' in question:return 'known_logic_error'
        if task=='instruction' and re.search(r'改写|润色|概括|总结|提取|翻译',question):
            if not re.search(r'[：:].{8}|\n.{12}',question,re.S):return 'missing_task_input'
            # New large numeric claims and inserted gender in faithful transformations.
            given=set(re.findall(r'\d{3,}(?:\.\d+)?',question))
            if set(re.findall(r'\d{3,}(?:\.\d+)?',answer))-given:return 'unsupported_numeric_addition'
            if re.search(r'男性|女性|男患者|女患者',answer) and not re.search(r'男性|女性|男患者|女患者',question):return 'unsupported_gender'
            if re.search(r'数以万计|数万|上万',answer) and not re.search(r'数以万计|数万|上万',question):return 'unsupported_numeric_addition'
            if re.search(r'填补缺失|插补|清洗数据|数据清洗',answer) and not re.search(r'填补缺失|插补|清洗数据|数据清洗',question):return 'unsupported_method'
        if re.search(r'我.{0,12}(?:住在|去过|吃过|我的朋友|我父母|亲身经历)',answer):return 'unsupported_personal_experience'
        if not answer.strip():return 'empty_answer'
    return None
