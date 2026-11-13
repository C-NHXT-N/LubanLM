import unittest
from scripts.sft_content_checks import context_task, postcheck

def messages(*pairs):
    return [dict(role='user' if i % 2 == 0 else 'assistant', content=s) for i,s in enumerate(pairs)]

def check(*pairs):
    return postcheck(dict(group='unit-test',messages=messages(*pairs),task='concept'))

class ContextChecks(unittest.TestCase):
    def test_writing_and_definitions_not_daily(self):
        for q in ['请修改以下文字：我周末去工作。','检查语法：我想学习。','写封给朋友的感谢信。','给出机器学习与数据挖掘的区别。','解释休息的意思。']:
            self.assertNotEqual(context_task(messages(q,'回答。')),'daily')
        self.assertEqual(context_task(messages('如何提高工作学习效率？','认真制定计划。')),'daily')
        for q in ['原句：我们的沟通不够充分。','为未来工作场所会议撰写议程。','将内容整理为日常对话格式。','编写小说开头，描述主人公。','从给定的数据集预测目标变量。']:
            self.assertNotEqual(context_task(messages(q,'回答。')),'daily')

    def test_followup_metrics_and_metadata(self):
        self.assertEqual(check('改写论文摘要：测试方法有效。','方法有效。','补充准确率。','准确率95%。')[1],'invented_research_metric')
        self.assertIsNone(check('改写论文摘要：准确率95%。','测试准确率95%。')[1])
        self.assertEqual(check('tokenId: 6','继续修改。')[1],'metadata_turn')
        self.assertIsNone(check('解释日志中的tokenId: 6。','这是一个标识符。')[1])

    def test_names_season_and_past(self):
        self.assertEqual(check('你叫什么名字？','我叫智灵。')[1],'unsupported_model_identity')
        self.assertIsNone(check('你叫什么名字？','我叫LubanLM。')[1])
        self.assertEqual(check('现在是什么季节？','现在是春季。')[1],'ungrounded_season')
        self.assertEqual(check('回到过去改变你已经做过的事情？','大学选专业。')[1],'unsupported_physical_persona')

    def test_farewell_does_not_hide_context(self):
        self.assertEqual(context_task(messages('什么是引力？','物体之间的相互作用。','谢谢！','不客气。')), 'concept')
        self.assertEqual(context_task(messages('如何提高学习效率？','制定计划。','再见！','再见！')), 'daily')

    def test_identity_is_not_a_quoted_name(self):
        self.assertEqual(context_task(messages('写一句代码注释：name=input("你叫什么名字？")','获取名字。')), 'instruction')
        self.assertEqual(context_task(messages('你是谁？','我是LubanLM，一个中文助手。')), 'identity')

    def test_real_greeting_and_verified_math(self):
        self.assertEqual(context_task(messages('你好！','你好！')), 'greeting')
        self.assertEqual(context_task(messages('计算3+4。','7。')), 'arithmetic')

    def test_rewrite_is_not_daily_because_of_birthday(self):
        self.assertEqual(context_task(messages('改写：我明天参加朋友的生日聚会。','明天我将参加朋友的生日聚会。')), 'instruction')

    def test_weather_fabrication_and_clarification(self):
        self.assertEqual(check('明天北京天气怎么样？','明天北京晴，气温15到25度。')[1], 'ungrounded_weather')
        self.assertIsNone(check('北京天气怎么样？','我没有实时天气数据，请查看气象信息。')[1])
        self.assertIsNone(check('根据记录：明天气温20度，描述天气。','明天气温20度。')[1])

    def test_generated_equation_and_corrected_bad_equation(self):
        self.assertEqual(check('生成一个数学例子。','10 + 4 x 3 - 6 ÷ 2 = 16')[1], 'false_embedded_equation')
        self.assertIsNone(check('生成一个数学例子。','10 + 4 x 3 - 6 ÷ 2 = 19')[1])
        self.assertIsNone(check('指出错误：2+2=5。','2+2=5是错误的，正确结果是4。')[1])

    def test_persona_and_requested_fiction(self):
        self.assertEqual(check('你家乡在哪里？','我的家乡是福建。')[1], 'unsupported_physical_persona')
        self.assertIsNone(check('写一段故事，主角的家乡是福建。','我的家乡是福建。')[1])
        self.assertEqual(check('你叫什么名字？','你可以叫我晓晓。')[1], 'unsupported_model_identity')

    def test_precision_and_json(self):
        self.assertEqual(check('从末尾反向打印：甲乙丙。','丙甲乙。')[1], 'unverified_precision_operation')
        self.assertEqual(check('以JSON格式输出。','{"ANSWER": 英伦银行}')[1], 'invalid_requested_json')
        self.assertIsNone(check('以JSON格式输出。','{"ANSWER":"英伦银行"}')[1])

    def test_ordinary_bullet_list_not_truncation(self):
        self.assertIsNone(check('如何提高工作效率？','建议：'+('认真计划，合理安排时间。'*25)+'最后保持专注')[1])
        self.assertEqual(check('解释全球化。',('全球化促进文化交流。'*30)+'促')[1], 'unfinished_long_answer')

if __name__ == '__main__':unittest.main()
