"""Deterministic context checks after encoding; never rewrite public answers."""
import json
import re
from functools import lru_cache
from scripts.common import ROOT, read_json
from scripts.corpus_quality import arithmetic, verified_arithmetic_dialogue

EXCLUSIONS = 'configs/sft-content-exclusions-20261007.json'
TEXT_REQUEST = re.compile(r'原句|写作|撰写|对话格式|小说|名言|改写|润色|编辑|校对|纠正|拼写|翻译|提取|摘要|总结|概括|填空|标点|主语|时态|语态|排序|分类|替换|删除|注释|语法|分词|分段|概要|梗概|原文[：:]|文字优化|(?:修改|改进|修复|扩展|简化).{0,18}(?:文字|文本|句|信|文|段|表达|人称)|(?:生成|编写|写|创作|仿写|续写|构造|构建|创建|描述).{0,18}(?:诗|故事|对话|段落|句子|短文|文章|文案|标题|贺卡|口号|信|邮件|广告|简历|消息)|(?:句子|段落|文本|文章|简历|邮件|广告|关键词|同义词|文字|标题)|开放式生成任务')
CONCEPT = re.compile(r'什么是|什么叫|原理|定义|含义|是什么|解释|区别|意思|为什么|摩尔|染色体|数据集|目标变量|回归模型|机器学习|数据挖掘|算法|神经网络|计算BMI')
LIFE = re.compile(r'生活|工作|朋友|心情|压力|失眠|焦虑|放松|周末|习惯|学习|休息|建议|社交|饮食|整理|感情|生日|道歉|感谢|聊天|孤独|难过|高兴|安慰|分心|沟通|效率|任务|自学|健身|家务|睡眠|同事|目标|减肥')
GREETING = re.compile(r'^(?:你好|您好|嗨|早上好|早安|晚安|再见|谢谢|感谢|好的|好|明白了|知道了)[，,！!。？?\s]*(?:你好吗|今天过得怎么样|最近好吗|谢谢|再见|下次聊)?[，,！!。？?\s]*$')
SELF = re.compile(r'^(?:请|能|可以|告诉我|我想知道|简要|简短|介绍一下|介绍下|请问|你好[，, ]*)*(?:你是谁|你叫什么(?:名字)?|你的名字|你是人(?:类)?吗|介绍.{0,4}你自己|自我介绍|你自己|你是由谁开发)')

@lru_cache(maxsize=1)
def excluded():
    return read_json(ROOT / EXCLUSIONS)['groups']

def context_task(messages):
    if verified_arithmetic_dialogue(messages):
        return 'arithmetic'
    questions = [m['content'].strip() for m in messages if m['role'] == 'user']
    substantive = [q for q in questions if not GREETING.fullmatch(q) and not SELF.match(q)]
    if not substantive:
        return 'identity' if any(SELF.match(q) for q in questions) else 'greeting'
    # A farewell never changes the task of the supervised earlier answers.
    first = substantive[0]
    if TEXT_REQUEST.search(first):
        return 'instruction'
    if LIFE.search(first) and not CONCEPT.search(first):
        return 'daily'
    return 'concept'

def postcheck(item):
    if item['group'] in excluded():
        return None, 'inspected_bad_family'
    messages = item['messages']
    writing_context = False
    user_context = ''
    for index in range(1, len(messages), 2):
        question, answer = messages[index-1]['content'], messages[index]['content']
        writing_context = writing_context or bool(TEXT_REQUEST.search(question))
        writing = writing_context
        user_context += '\n'+question
        if re.fullmatch(r'\s*token_?id\s*:\s*\d+\s*', question, re.I):
            return None, 'metadata_turn'
        if writing and re.search(r'论文|摘要|研究|测试|測試|实验|實驗', user_context) and not re.search(r'虚构|虛構|假设|假設|示例数据|示例數據', user_context):
            for metric in re.findall(r'\d+(?:\.\d+)?\s*[%％]', answer):
                if metric.replace(' ','') not in user_context.replace(' ',''):
                    return None, 'invented_research_metric'
        # Unknown precision operations are excluded, rather than guessed.
        if re.search(r'谜语|同音词|反向打印|从末尾|最长单词|单词之间的距离|(?:词|单词|句子).{0,16}按.{0,8}(?:字母|长度)|每个名词短语.{0,16}同义词|(?:冠词|介词).{0,12}删|两个数的和.{0,25}索引', question):
            return None, 'unverified_precision_operation'
        if re.search(r'(?:[0-9一二三四五六七八九十百两]+\s*(?:个?单词|个?字|词以上)|字数)', question):
            return None, 'unverified_length_constraint'
        # Do not mistake generated weather prose for a retrieved forecast.
        weather_query = re.search(r'天气|天氣|气温|氣溫|降雨|温度', question)
        forecast = re.search(r'今天|明天|后天|未來|未来|当地|當地|北京|上海|你那边|你那邊|你这边|你這邊', question)
        grounded = re.search(r'(?:数据|數據|记录|紀錄|文本|描述|例子|示例|假设|假設|小说|小說|故事|输入|輸入)[：:]|(?:气温|温度)[：=]\s*\d', question)
        concrete = re.search(r'\d.{0,5}(?:度|°|摄氏|攝氏)|(?:天气|天氣).{0,10}(?:晴|雨|雪|阴|陰)|(?:晴|雨|雪).{0,15}气温', answer)
        if weather_query and forecast and concrete and not grounded:
            return None, 'ungrounded_weather'
        if re.search(r'现在.{0,8}(?:什么季节|是什么季节)', question) and re.search(r'现在是.{0,4}[春夏秋冬]', answer) and not re.search(r'\d+月|日期|假设|假設', user_context):
            return None, 'ungrounded_season'
        if not writing:
            if re.search(r'我的家乡|我的家鄉|我这边|我這邊|我住在|带上家人|帶上家人|下次有机会我.{0,8}体验|等去了.{0,10}分享|早上坚持跑了|中午.{0,8}(?:吃了|吃了一)|(?:我的|我有).{0,8}(?:老师|父母|家人)|我.{0,15}(?:上班|去游泳|出门散步)', answer):
                return None, 'unsupported_physical_persona'
            if re.search(r'(?:我的名字|我叫|叫我|称呼我|我(?:由|是)|我的开发者|我的開發者).{0,30}(?:阿里云|阿里雲|通义千问|通義千問|晓晓|曉曉|匿名.{0,10}团队)', answer):
                return None, 'unsupported_model_identity'
            if SELF.match(question) and re.search(r'(?:我叫|我的名字(?:是|叫)|叫我|称呼我)(?!\s*(?:LubanLM|鲁班|魯班|AI|人工智能|智能助手|小助手))\s*[A-Za-z\u4e00-\u9fff]{2,}', answer, re.I):
                return None, 'unsupported_model_identity'
            if re.search(r'你正在学习.{0,6}课程|回到过去.{0,15}你已经做过', question) and not re.search(r'没有|沒有|并非|並非|不具备|不具備|作为AI|作为人工智能|作為人工智能', answer):
                return None, 'unsupported_physical_persona'
        # Specific professional/legal facts are not necessary for basic chat.
        if re.search(r'刑事责任|刑事責任|刑法|担保法|法律规定|法律規定|非法所得|罚款.{0,10}(?:万元|倍)|犯罪.{0,12}年龄|犯罪.{0,12}年齡|赌博.{0,16}(?:犯罪|违法)|賭博.{0,16}(?:犯罪|違法)', question):
            return None, 'unverified_legal_detail'
        if re.search(r'人口.{0,12}(?:排名|大小|排列)|(?:排名|排列).{0,12}人口|最辣.{0,8}辣椒|哥德尔|哥德爾', question):
            return None, 'unverified_specific_fact'
        if re.search(r'真性美区|循医|Disposition_switch_placeholder|(?:sparking|arouse).{0,50}(?:investigational|论坛)', question, re.I):
            return None, 'known_translation_noise'
        if len(answer) > 200 and re.search(r'(?:[，、：；]|促|和|以及|同时|例如|包括|通过|但是|一个|的|与|或|并)$', answer.strip()):
            return None, 'unfinished_long_answer'
        if ('json' in question.lower() or re.search(r'以\s*\{\s*"', question)) and answer.strip().startswith(('{', '[')):
            try:
                json.loads(answer)
            except ValueError:
                return None, 'invalid_requested_json'
        # Catch bad equations inside generated examples, not just math prompts.
        expression = r'(?<![A-Za-z\d])([-+]?\d+(?:\.\d+)?(?:\s*[+×x*/÷−-]\s*[()]?\d+(?:\.\d+)?[()]?)+)\s*=\s*(-?\d+(?:\.\d+)?)'
        for match in re.finditer(expression, answer):
            around = answer[max(0,match.start()-20):match.end()+20]
            if re.search(r'错误|錯誤|错的|不等|不是|不正确|不正確', around):
                continue
            expr = match[1].replace('x','*')
            if arithmetic('计算 '+expr+' 的结果', '= '+match[2]) is False:
                return None, 'false_embedded_equation'
    task = context_task(messages)
    return dict(item, task=task), None
