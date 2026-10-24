"""保存完整阅读原始回答后逐题作出的语义评分；本文件不自动判分。"""
import json
from pathlib import Path

# 每项依次为相关且可读、核心正确、循环，以及该原始回答的具体理由。
records = {
    'greeting': [
        (True, True, False, '回应早安并表达愿意聊天，语义通顺。'),
        (True, True, False, '回应晚上好，随后补充你好，属于合理问候。'),
        (True, True, False, '表达同样很高兴见面，合理回应。'),
        (True, True, False, '用不客气回应感谢，角色与语境正确。'),
        (True, True, False, '以再见回应告别，前半句简短承接忙碌情境。'),
        (True, True, False, '接受聊天邀请，虽简短但回应合理。'),
        (True, True, False, '答应陪聊并询问今天过得怎样，合理发起交流。'),
        (False, False, True, '把初次见面的礼貌语当作注意事项，同一兴趣爱好句重复五次。'),
        (True, True, False, '用好、很高兴回应朋友的问候；措辞生硬，但能够理解且没有核心错误。'),
        (True, False, False, '句子通顺，但无喜好与情感不能推出无法回应聊天邀请，拒绝理由不成立。'),
    ],
    'identity': [
        (True, False, False, '提及人工智能身份，但以没有喜好情感为由拒绝自我介绍，核心回应不完整且理由不成立。'),
        (True, True, False, '明确回答模型名字LubanLM。'),
        (True, False, False, '反问能帮用户做什么，没有说明自己的帮助范围。'),
        (True, True, False, '说明自己是语言模型而非人；没有能力回答问题的自我限制不作为能力达标证据。'),
        (True, True, False, '明确说明人工智能没有身体，正确回答。'),
        (False, False, True, '没有给出称呼，反复说没有能力和能力，句子循环且不通顺。'),
        (True, True, False, '确认中文助手身份并说明简单聊天用途；实际能力仍须由后续题目检验。'),
        (True, True, False, '直接回应可以帮助理解简单概念，与提问相关。'),
        (False, False, True, '反复输出我可以回答与你好，未形成完整回应，达到生成长度上限。'),
        (False, False, False, '虽提到人工智能，但没有能力和能力这一残缺表述与提供帮助混杂，身份介绍不清楚。'),
    ],
    'daily': [
        (True, True, False, '暂时放下工作、喝水和休息是合理的简单休息建议。'),
        (True, False, False, '原样复述早起安排问题，没有提出今晚的安排。'),
        (False, False, False, '我需要我帮助我指代混乱，没有桌面整理步骤。'),
        (True, False, False, '把下雨出门提问改为陈述，没有说要带什么。'),
        (True, False, True, '没有给出每日读书办法，将读书与身体健康直接关联，并重复保持身体健康。'),
        (True, True, False, '建议真诚说谢谢，合理回应感谢朋友的需求。'),
        (True, False, False, '直接断言没有忘带东西，没有提供检查方法。'),
        (True, False, False, '听读写是学习活动，不能作为解决分心的具体办法；写重复一次不视为持续循环。'),
        (False, False, True, '把散步问题转到文本分类，文本分类重复三次，既离题又循环。'),
        (True, True, False, '给出完整、合理的生日祝福。'),
    ],
    'concept': [
        (True, True, False, '用处理信息的工具解释电脑并列出用途，符合一句话要求。'),
        (True, False, False, '反问键盘用途，没有解释输入文字或操作的作用。'),
        (True, True, False, '正确说明鼠标移动指针、点击和操作电脑的作用。'),
        (True, True, False, '正确说明浏览器用于访问网页。'),
        (True, True, False, '以连接计算机、分享信息交流解释互联网，基础含义正确。'),
        (True, True, False, '正确说明文件夹整理和存放文件的用途。'),
        (True, True, False, '正确说明充电补充电池电量。'),
        (True, True, False, '水滴在云中增大并降到地面是合理的基础降雨解释。'),
        (True, False, False, '阳光影响环境没有解释植物需要阳光的原因，末尾重述提问不算连续生成循环。'),
        (False, False, False, '冰电池的电池电池语义混乱，没有说明冰受热融化。'),
    ],
    'instruction': [
        (True, True, False, '只输出好并带句末标点，按既有评分口径接受标点。'),
        (True, False, False, '要求两种水果却给出苹果、香蕉、梨三种，数量不符。'),
        (True, False, True, '要求原样写一遍，却加入因为并重复同一我喜欢学习分句。'),
        (True, False, True, '重复春风吹过四次，未使用春天且没有形成所要求的单句。'),
        (True, False, False, '复述给小狗起中文名字的要求，没有实际名字。'),
        (True, True, False, '红色和蓝色两种颜色均已说出，蓝色多重复一次但不构成持续循环，核心请求完成。'),
        (True, True, False, '给出坚持练习、逐步进步的合理鼓励。'),
        (True, True, False, '用一句完整的话向老师问好，符合要求。'),
        (True, False, False, '要求输出谢谢，却回复不客气并加解释，未遵循指定输出。'),
        (True, False, False, '要求一种交通工具，却列出公交车和地铁两种，数量不符。'),
    ],
    'arithmetic': [
        (True, False, False, '回答2，正确结果应为5。'),
        (False, False, False, '2的2不是清楚的计算结果，7减4应为3。'),
        (True, False, False, '重复3乘2这个表达式，没有计算出6。'),
        (True, False, False, '回答1，8除以2应为4。'),
        (True, False, False, '回答1，5加4应为9。'),
        (True, False, False, '复述9减3得到几，没有给出6。'),
        (True, False, False, '回答4，4乘3应为12。'),
        (True, False, False, '将除法写成12等于12，没有给出正确商4。'),
        (True, False, False, '复述6加1等于几，没有给出7。'),
        (True, True, False, '计算结果为2，数值正确；多余的次字措辞不当但不改变主要运算结果。'),
    ],
}
marks = []
for category, rows in records.items():
    for i, (readable, correct, looping, reason) in enumerate(rows):
        marks.append(dict(id=f'dev-{category}-{i:02d}', seed=2026,
                          relevant_readable=readable, correct=correct,
                          looping=looping, reason=reason))
Path(__file__).with_name('50m-chat-v2-dev.marks.json').write_text(
    json.dumps({'marks': marks}, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
