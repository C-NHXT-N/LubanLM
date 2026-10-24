"""完整阅读v4开发原始输出后的逐题语义评分记录，不进行自动关键词判分。"""
import json
from pathlib import Path

records = {
    'greeting': [
        (True, True, False, '回应早上好并祝当天顺利，问候自然。'),
        (True, True, False, '主要内容明确回应晚上好；后半句你可以和你聊天的代词不当，但不改变核心问候。'),
        (True, True, False, '以你好和很高兴认识你回应见面，合理。'),
        (True, True, False, '不客气回应感谢，角色正确。'),
        (True, True, False, '再见并邀请有需要再聊，合理回应告别。'),
        (True, True, False, '回应你好并表示很高兴聊天，符合打招呼情境。'),
        (True, True, False, '答应陪聊并询问近况，形成合理交流。'),
        (True, True, False, '接受多关照的礼貌请求，接着询问需要何种知识帮助，属于合理的合作回应。'),
        (True, True, False, '下午好是可互相回应的问候，相同措辞不算循环或无效复述。'),
        (False, False, False, '只有我可以，没有说可以做什么，回答没有完成。'),
    ],
    'identity': [
        (True, False, False, '提到语言模型身份，但没有和你聊天的断言与当前交流不符，介绍混入错误。'),
        (True, True, False, '明确给出名字LubanLM。'),
        (True, False, False, '原样反问用户能帮我做什么，没有说明助手的能力范围。'),
        (True, True, False, '明确说明语言模型而非真人，正确回答人还是程序。'),
        (True, True, False, '说明没有身体且以文字交流，回应正确。'),
        (True, False, False, '说明类型但没有给出称呼，未完成怎么称呼你的请求。'),
        (True, False, False, '只说明语言模型而非真人，没有回答是否中文助手；不能仅凭使用中文代替对助手身份的回应。'),
        (True, True, False, '答应帮助理解基础知识，并询问具体需求。'),
        (True, True, False, '说明能够聊天、解释基础知识和提供日常建议，合理回答用途；实际可靠性由其他题检验。'),
        (False, False, False, '你对我很有用是在评价用户，没有说明自己的身份。'),
    ],
    'daily': [
        (True, True, False, '暂停工作、喝水、休息是合理的简单建议。'),
        (False, False, False, '今晚怎么和你想语义不完整，未提供早起前的睡前安排。'),
        (False, False, False, '您3个小目标、我将等片段语法混乱，没有实际桌面整理办法。'),
        (False, False, False, '只有句号，没有任何回答内容。'),
        (False, False, False, '每天读书的冒号后为空，没有可理解的读书建议。'),
        (True, True, False, '建议真诚说谢谢，合理回应对朋友的感谢。'),
        (True, True, False, '提出清单和核对关键物品的方法，能够执行。'),
        (True, True, False, '建议先设小目标并集中完成，是合理的减少分心办法。'),
        (True, True, False, '提醒鞋、天气和交通安全，符合散步准备。'),
        (True, True, False, '提供完整的生日祝福。'),
    ],
    'concept': [
        (True, True, False, '电脑处理信息的定义及学习工作交流用途正确。'),
        (True, True, False, '正确解释键盘输入文字、数字和指令。'),
        (True, True, False, '正确说明鼠标移动指针和点击操作。'),
        (True, True, False, '正确说明浏览器访问网页。'),
        (True, True, False, '用连接计算机和信息交流解释互联网，含义正确。'),
        (True, True, False, '文件夹整理和存放文件的说明正确。'),
        (True, True, False, '手机充电补充电量的说明正确。'),
        (True, True, False, '云中水滴增大后降落是合理的简短降雨解释。'),
        (True, True, False, '说明光合作用利用光能制造有机物，正确回答需要阳光的原因。'),
        (False, False, True, '冰乘积了既没有解释融化又重复三次，出现语义混乱和循环。'),
    ],
    'instruction': [
        (True, True, False, '只给出好和句末标点，沿用此前允许标点的口径。'),
        (True, True, False, '苹果与香蕉恰好两种水果。'),
        (True, False, False, '增添是我喜欢的学习，未原样抄写一遍；近义词冗余不判为持续循环。'),
        (True, False, False, '句子谈夏天，没有使用要求的春天，未遵守指定词语。'),
        (True, False, False, '输出小狗后附英文模型名LubanLM，没有明确给出一个中文宠物名字。'),
        (True, True, False, '准确列出红色与蓝色。'),
        (True, True, False, '给出合理的学习鼓励。'),
        (True, True, False, '用一句话向老师问好并祝愉快。'),
        (True, True, False, '仅输出谢谢和句末标点，没有解释。'),
        (True, True, False, '公交车是一种常见交通工具，数量符合要求。'),
    ],
    'arithmetic': [
        (True, True, False, '2加3等于5，数值正确。'),
        (True, True, False, '7减4等于3，数值正确。'),
        (True, True, False, '3乘2等于6，数值正确。'),
        (True, False, False, '回答2，8除以2应为4。'),
        (True, True, False, '5加4等于9，数值正确。'),
        (True, True, False, '9减3等于6，数值正确。'),
        (True, True, False, '4乘3等于12，数值正确。'),
        (True, True, False, '12除以3等于4，数值正确。'),
        (True, True, False, '6加1等于7，数值正确。'),
        (True, True, False, '10减8等于2，数值正确。'),
    ],
}
marks=[]
for category, rows in records.items():
    for i, (readable, correct, looping, reason) in enumerate(rows):
        marks.append(dict(id=f'dev-{category}-{i:02d}',seed=2026,
                          relevant_readable=readable,correct=correct,looping=looping,reason=reason))
Path(__file__).with_name('50m-chat-v4-dev.marks.json').write_text(
    json.dumps({'marks':marks},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
