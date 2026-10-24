from scripts.common import ROOT,read_json,write_json,sha256,utc
from scripts.prepare_general_2b import key
from eval.chat_suite import normalize,excluded_questions,repeated
from sft.template import encode_messages
from tokenizers import Tokenizer
from pathlib import Path
from collections import Counter,defaultdict
import argparse
import json
import random
import re
import sqlite3
import shutil
from fractions import Fraction
import pyarrow.parquet as pq
from scripts.corpus_quality import dialogue_reason,arithmetic,verified_arithmetic_dialogue
from scripts.balanced_sft import select_balanced

def rows(path):
    if path.suffix=='.parquet':
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256): yield from batch.to_pylist()
    else:
        with path.open(encoding='utf-8') as stream:
            first=stream.read(1);stream.seek(0)
            if first=='[':
                decoder=json.JSONDecoder();buffer=stream.read(1<<20)[1:];need_value=True
                while True:
                    buffer=buffer.lstrip()
                    if not buffer:
                        buffer=stream.read(1<<20)
                        if not buffer:raise ValueError('Incomplete JSON array')
                        continue
                    if buffer[0]==']':return
                    if not need_value:
                        if buffer[0]!=',':raise ValueError('Missing JSON array separator')
                        buffer=buffer[1:];need_value=True;continue
                    while True:
                        try:value,end=decoder.raw_decode(buffer);break
                        except json.JSONDecodeError:
                            more=stream.read(1<<20)
                            if not more:raise
                            buffer+=more
                            if len(buffer)>64<<20:raise ValueError('Oversized JSON record')
                    yield value;buffer=buffer[end:];need_value=False
            for line in stream:
                if line.strip(): yield json.loads(line)

def messages(row):
    if 'conversations' in row:
        result=[]
        for m in row['conversations']:
            role=m.get('role',m.get('from'))
            role={'human':'user','gpt':'assistant'}.get(role,role)
            result.append(dict(role=role,content=m.get('content',m.get('value','')).strip()))
        return result
    return [dict(role='user',content=(row.get('instruction','')+'\n'+row.get('input','')).strip()),dict(role='assistant',content=row.get('output',row.get('response','')).strip())]

def category(filename,question):
    if len(question)<60 and re.search(r'你.{0,5}(?:介绍.{0,4}(?:自己|你自己)|是谁|叫什么|名字)|介绍.{0,4}你自己',question): return 'identity'
    if len(question)<40 and re.search(r'^(?:你好|您好|早上好|早安|晚安|再见|谢谢)[，,！!。 ]*(?:今天过得怎么样[？?]?|最近好吗[？?]?|你好吗[？?]?)?$',question):return 'greeting'
    if len(question)<40 and re.fullmatch(r'[请能可以否吗？?，, ]*(?:你是谁|你叫什么(?:名字)?|你的名字(?:是什么)?|你是人(?:类)?吗|介绍一下你自己)[呢吗？?。!！ ]*',question): return 'identity'
    if re.fullmatch(r'[你好您好早安晚安早上好再见谢谢感谢啦啊呀哦呢！!，,。？? ]{2,18}',question): return 'greeting'
    if filename in ('rewrite.parquet','editing.parquet','format-constrain.parquet','summary.parquet'): return 'instruction'
    # An everyday source can still contain a factual/definition question.
    if re.search('什么是|什么叫|原理|区别|是什么|解释.{0,6}(?:概念|含义)',question):return 'concept'
    if filename in ('everyday.parquet','advice-seeking.parquet','planning.parquet','role-playing.parquet'): return 'daily'
    if re.search('为什么|什么是|解释|原理|区别|是什么',question): return 'concept'
    if re.search('改写|提取|分类|照抄|翻译|列出|写一句|生成|总结|概括|编写|写一|写.{0,4}(?:声明|文章|故事|诗|文案|信)|提供.{0,8}(?:段落|例子)|编辑|修改|润色|排序|输出|标点|校对|填空|文本中|给定的|输入的',question): return 'instruction'
    if re.search('生活|工作|朋友|心情|压力|失眠|焦虑|放松|周末|习惯|学习|休息|怎么办|建议|社交|饮食|整理|感情|生日|道歉|感谢|聊天|孤独|难过|高兴|安慰|分心|沟通|今天|明天|天气',question): return 'daily'
    return 'concept'

def candidates(row,source,filename,file_key,row_index,blob,plan,tokenizer,blocked,bad_groups,count):
    for row in [row]:
        if 'smoltalk' in source['repo'] and (('score' in row and row['score']<4) or row.get('difficulty') in ('困难','非常困难')):
            count['source_quality']+=1;continue
        if source['repo'].startswith('Magpie') and (row.get('input_quality')=='poor' or row.get('difficulty') in ('hard','very hard') or row.get('instruct_reward',-1)<0 or row.get('task_category') in ('Coding & Debugging','Reasoning','Math','Data analysis')):
            count['source_quality']+=1;continue
        convo=messages(row)
        system='你是LubanLM，一个中文助手。'
        if convo and convo[0]['role']=='system':system=convo.pop(0)['content']
        if row.get('system_prompt'):
            if system!='你是LubanLM，一个中文助手。' and system!=row['system_prompt']:count['conflicting_system']+=1;continue
            system=row['system_prompt']
        if re.search(r'Qwen|通义|DeepSeek|ChatGPT|OpenAI|扮演|假装',system,re.I):count['incompatible_system']+=1;continue
        if not convo or convo[0]['role']!='user': continue
        if len(convo)%2 or any(m['role']!=('user' if i%2==0 else 'assistant') or not m['content'] for i,m in enumerate(convo)):count['invalid_roles']+=1;continue
        legacy_group=key(normalize(convo[0]['content']))
        group=key(file_key+':'+str(row_index))
        if legacy_group in bad_groups:count['known_rejected_group']+=1;continue
        for end in range(2,len(convo)+1,2):
            ms=convo[:end]
            question=ms[-2]['content'];question_key=key(json.dumps([system,*[(m['role'],normalize(m['content'])) for m in ms[:-1]]],ensure_ascii=False))
            task=category(filename,question)
            math_ok=arithmetic(question,ms[-1]['content'])
            if math_ok is False:count['arithmetic_not_verified']+=1;continue
            if math_ok is None and re.search(r'计算|算一下|等于多少',question) and re.search(r'\d\s*[+×*/÷−^%-]',question):count['unsupported_arithmetic']+=1;continue
            if math_ok is True:task='arithmetic'
            text='\n'.join(m['content'] for m in ms)
            reason=dialogue_reason(ms,task)
            if reason:count[reason]+=1;continue
            if re.search(r'(?:[一二三四五六七八九十百0-9]+个?字|字数|其他内容保持不变|只(?:需要)?修改|只.{0,5}替换)',question):
                count['unverified_exact_constraint']+=1;continue
            if re.search(r'颅骨|枕骨|脊髓|神经|临床|诊断|用药|药物|处方|大学物理|量子|投资收益|电影名称|剧情简介|评分|白茶.{0,8}未发酵|请.{0,4}担任|作为一名',text):
                count['specialized_or_persona']+=1;continue
            if re.search(r'出现(?:最)?频(?:率|繁)|出现次数|单词.{0,8}频率|鲸吞式并购|地球同步卫星|松鼠的种类',text):
                count['unverified_precision_task']+=1;continue
            if re.search(r'总结|概括|改写|提取|根据给定|给定文本|提供的文本|以下文本|去掉.{0,8}冗余|修改.{0,6}(?:句子|文字)|重写|校对|翻译句子',question) and not re.search(r'[：:].{8}|\n.{12}',question,re.S):
                count['missing_task_input']+=1;continue
            sentence_count=re.search(r'([1-5一二三四五两])(?:句话|个句子)',question)
            if sentence_count:
                number={'一':1,'二':2,'两':2,'三':3,'四':4,'五':5}.get(sentence_count[1],int(sentence_count[1]) if sentence_count[1].isdigit() else 0)
                if len(re.findall(r'[。！？!?]',ms[-1]['content']))!=number:count['sentence_constraint']+=1;continue
            if re.search(r'[^\x00-\x7f\u3400-\u9fff\u3000-\u303f\uff00-\uffef\u2000-\u206f\u00b0\u00d7\u00f7]',text):
                count['unexpected_script']+=1;continue
            if re.search(r'扮演|假装|假设你|你的角色|角色是|以.{0,10}身份',text):
                count['persona_roleplay']+=1;continue
            if re.search(r'最新|近期|最近.{0,8}(?:新闻|热点)|202[0-9]年|人均可支配|《|外交部发言|功效|抗氧化|治疗|疗效|肿瘤|量子霸权',text):
                count['unstable_or_specialized_fact']+=1;continue
            bad=False
            for message in ms:
                if message['role']!='assistant':continue
                answer=message['content']
                if task!='instruction' and re.search(r'我.{0,10}(?:喜欢|会|常常|经常|有时|通常).{0,12}(?:宅在家|看书|追剧|追追剧|朋友|出去|健身|运动|周末|周末|吃饭|喝茶|喝咖啡|做饭|散步|旅游|旅行)',answer):bad=True
                if re.search(r'不能.{0,8}推荐.{0,12}游戏|请提供需要翻译的句子',answer):bad=True
                english=re.findall(r'[A-Za-z]{3,}',answer)
                allowed={'CPU','GPU','RAM','ROM','USB','HTTP','HTTPS','DNS','JSON','AI','PDF','HTML','CSS','Linux','Windows','Android','iOS','LubanLM','Python','JavaScript','SQL','WiFi','UTF','TCP','UDP','IP','GHz','MHz','GPT','API'}
                supplied=' '.join(m['content'] for m in ms if m['role']=='user')
                supplied_words={w.lower() for w in re.findall(r'[A-Za-z]{3,}',supplied)}
                if any(w not in allowed and w.lower() not in supplied_words for w in english):bad=True
                if task!='instruction' and re.search(r'我.{0,15}(?:吃(?:了|过|一)|喝(?:了|过|一)|住在|穿着|买了|去过|看了|上班|上学|在家|我的朋友|我朋友|我家人|我父母)',answer):bad=True
                if task!='instruction' and (re.search(r'我(?:曾经|最近|昨天|小时候|平时|经常|自己|也|有时).{0,10}(?:试过|去过|读过|吃过|喝过|做饭|烹饪|参加|散步|旅行|放松|冥想|家乡|亲身|生活|成长)',answer) or re.search(r'我.{0,15}(?:喜欢做菜|喜欢做.{0,5}料理|亲自|准备饭菜)|寿司和沙拉马上就好|我都可以为您准备',answer)):bad=True
            if bad:count['unsupported_experience_or_english']+=1;continue
            if any(x in text for x in ('\ufffd','<think>','</think>','<|','<bos>','<eos>','```','http://','https://')) or repeated(text):
                count['invalid_or_complex']+=1;continue
            if re.search(r'(?:我是|我叫|由.{0,12}(?:开发|训练)).{0,20}(?:Qwen|通义|DeepSeek|深度求索|ChatGPT|OpenAI|MiniMind|夸克|Quark)',text,re.I):
                count['wrong_identity']+=1;continue
            if sum('\u4e00'<=c<='\u9fff' for c in text)/max(1,len(re.sub(r'\s','',text)))<.4 and not (task=='arithmetic' and verified_arithmetic_dialogue(ms)):
                count['language']+=1;continue
            if any(normalize(m['content']) in blocked for m in ms if m['role']=='user'):
                count['test_overlap']+=1;continue
            encoded=encode_messages(tokenizer,ms,plan['sft']['max_length'],plan['sft']['max_answer_tokens'],system=system)
            if not encoded:
                count['too_long_or_incomplete']+=1;continue
            item=dict(encoded,task=task,source=source['repo'],revision=source['revision'],file=filename,raw_sha256=blob['lfs']['sha256'],source_row=row_index,group=group,question_sha256=question_key,system=system,messages=ms)

            yield item

def main():
    from scripts.sft_disk import main as disk_main
    disk_main()

if __name__=='__main__':main()
