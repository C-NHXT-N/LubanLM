"""Extra source checks for the independently versioned short-chat supplement."""
import re

def row_reason(row, source):
    if source['repo']=='allenai/WildChat-1M':
        if row.get('language')!='Chinese' or row.get('toxic') or row.get('redacted') or any(m.get('toxic') or m.get('redacted') for m in row.get('conversation',[])):
            return 'source_language_or_risk'
    messages=row.get('conversations',row.get('conversation',[]))
    if isinstance(messages,list) and any(isinstance(m,dict) and (m.get('role') in ('tool','function') or m.get('tool_calls') or m.get('function_call') or m.get('tools')) for m in messages):
        return 'tool_protocol'
    if source['repo']=='opencsg/smoltalk-chinese' and ((isinstance(row.get('score'),(int,float)) and row['score']<4) or row.get('difficulty') in ('困难','非常困难')):
        return 'source_quality'
    if source['repo'].startswith('Magpie') and (row.get('input_quality')=='poor' or row.get('difficulty') in ('hard','very hard') or row.get('instruct_reward',-1)<0 or row.get('task_category') in ('Coding & Debugging','Reasoning','Math','Data analysis')):
        return 'source_quality'
    return None

def message_reason(messages, *, human=False):
    text='\n'.join(m['content'] for m in messages)
    if re.search(r'\ufffd|(?:┌|┐|└|┘|class\"?:)|-average-',text):return 'source_text_artifact'
    if re.search(r'实时|最近.{0,8}(?:上映|热播|新片)|最新.{0,8}(?:电影|新闻|价格)|近期.{0,8}新闻',text):return 'unstable_claim'
    for user,assistant in zip(messages[::2],messages[1::2]):
        if len(assistant['content'])>=40 and assistant['content'] in user['content'] and not re.search(r'改写|重写|编辑|照抄|复制|提取|引用|格式|润色|翻译|总结|概括|分类',user['content']):
            return 'answer_contaminated_user'
    for message in messages[1::2]:
        answer=message['content']
        if re.search(r'我(?:小时候|童年|上学|读书的时候|以前读书|老家|的生日|的头发|的皮肤|的身体|的同事|的老师|的父母|的家人)|(?:我.{0,8})?(?:忙里偷闲|生活还算有滋有味)|我(?:最近|今天|昨天|刚刚|前几天|周末).{0,18}(?:挺好|还好|不错|忙|生活|看了|看过|做菜|菜谱|做饭|出去|出去玩|旅游|旅行|吃|喝)|(?:看过|去过|吃过|喝过)(?:啊|呀|呢)|我.{0,8}(?:亲身|亲自体验|真人|家乡)|我们(?:一起|一块).{0,8}(?:去|吃|喝|买|散步)|我.{0,8}(?:陪你去|陪你走|帮你买|给你买|到你家|来你家)',answer):
            return 'unsupported_personal_experience'
        if re.search(r'(?:我是|我叫|我的名字|由.{0,12}(?:开发|训练)).{0,22}(?:jingyaogong|gongjy|MiniMind|MOSS|Qwen|通义|DeepSeek|ChatGPT|OpenAI|复旦)',answer,re.I):
            return 'source_identity'
        if re.search(r'我.{0,10}(?:训练数据|训练模型|模型训练|训练来源)|我的(?:训练|开发者|架构|参数数量)|我.{0,10}(?:基于|背后|母版本).{0,8}模型',answer):
            return 'unsupported_model_metadata'
    if human:
        if not messages or len(messages[0]['content'].strip())<6:return 'human_incomplete_context'
        if re.search(r'宝贝|亲爱的|傻|蠢|滚|卧槽|他妈|打死|开房|约炮|爱你|想你|学长|学姐|哥哥|姐姐|老公|老婆|照片|自拍|加微信|手机号|身份证',text):
            return 'human_relationship_or_sensitive'
        if re.search(r'@|https?://|\b\d{5,}\b',text):return 'human_external_context'
        for message in messages[1::2]:
            answer=message['content']
            if re.search(r'[?？]\s*$',answer) and not re.search(r'建议|可以先|不妨|试着',answer):return 'human_question_not_assistant_answer'
            if re.search(r'我|俺|咱|在家|回家|上班|下班|上课|开学|宿舍|你来|见面|见到|找你|来找|买回来|去买|吃饭|饿了',answer):return 'human_persona'
            if not re.search(r'可以|建议|不妨|试着|别着急|别担心|加油|祝你|祝福|放松|休息|慢慢|照顾|注意|希望你|理解你|愿你|保持|相信你|谢谢|不客气|没关系|不用谢|欢迎',answer):
                return 'human_not_assistant_response'
    return None
