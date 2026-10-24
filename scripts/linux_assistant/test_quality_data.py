"""Standalone regressions for hidden constraints and retired corpus builders."""
import unittest

from scripts.linux_assistant.dataset_quality import row_issues, split_issues


def row(prompt, answer='OK', **extra):
    return {'id': 'fixture', 'group': 'case-a', 'messages': [
        {'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': answer}],
        'reference': answer, **extra}


class DatasetQualityTests(unittest.TestCase):
    def test_json_structure_does_not_impose_unasked_reference_wording(self):
        from scripts.linux_assistant.semantic_grading import format_check
        spec={'kind':'json_schema','fields':{'status':'string',
              'missing_info':{'type':'array','items':'string'}},'additional_fields':False}
        for answer in ('{"status":"待查","missing_info":["模型路径"]}',
                       '{"missing_info":["启动参数","路径"],"status":"证据不足"}'):
            self.assertTrue(format_check(answer,spec)[0])
        for answer in ('{"status":false,"missing_info":[]}',
                       '{"status":"待查","missing_info":[1]}',
                       '{"status":"待查","missing_info":[],"extra":true}',
                       '```json\n{"status":"待查","missing_info":[]}\n```'):
            self.assertFalse(format_check(answer,spec)[0])

    def test_missing_previous_format_definition_is_rejected(self):
        r = row('已知驱动版本是0.9.8，但尚未验证。只输出JSON，使用最新信息和同样两字段。',
                '{"driver":"0.9.8","verified":false}')
        self.assertIn('missing_prior_format_definition', row_issues(r))
        self.assertIn('undefined_json_field:driver,verified', row_issues(r))

    def test_format_is_accepted_when_prior_context_defines_it(self):
        r = {'id': 'defined', 'group': 'case-b', 'messages': [
            {'role': 'user', 'content': '请使用JSON字段driver和verified。'},
            {'role': 'assistant', 'content': '好的。'},
            {'role': 'user', 'content': '版本是0.9.8，只输出同样两字段JSON。'},
            {'role': 'assistant', 'content': '{"driver":"0.9.8","verified":false}'}],
            'reference': '{"driver":"0.9.8","verified":false}'}
        self.assertNotIn('missing_prior_format_definition', row_issues(r))
        self.assertFalse(any(x.startswith('undefined_json_field:') for x in row_issues(r)))

    def test_missing_three_item_format_is_rejected(self):
        r = row('只要排查结果。仍用上述三项格式回复。', '检查：路径；判断：未知；下一步：读取日志。')
        self.assertIn('missing_prior_format_definition', row_issues(r))

    def test_filename_must_come_from_visible_input(self):
        r = row('请检查这个文件是否存在，并只给出命令。', 'test -f /opt/model.bin')
        self.assertIn('filename_missing_from_input', row_issues(r))

    def test_command_must_obey_read_only_constraint(self):
        r = row('只读检查目录状态，不要删除或修改文件。', 'rm -rf /tmp/cache')
        self.assertIn('command_conflicts_with_user_constraint', row_issues(r))
        multiline = row('只读检查目录状态，不要删除或修改文件。', '查看一下\nrm -rf /tmp/cache')
        self.assertIn('command_conflicts_with_user_constraint', row_issues(multiline))

    def test_command_only_rejects_explanatory_prose(self):
        from scripts.linux_assistant.semantic_grading import format_check
        self.assertTrue(format_check('ls -la', {'kind': 'command_only'})[0])
        self.assertFalse(format_check('请运行 ls -la。', {'kind': 'command_only'})[0])

    def test_source_command_is_a_machine_checked_content_condition(self):
        from scripts.linux_assistant.semantic_grading import validate_model_review
        r = row('页面示例命令：`grep -n error app.log`。', '可用grep查找。', grading={
            'format': {'kind': 'free'}, 'content': {'required_snippets': ['grep -n error app.log']}})
        r['answer'] = r['reference']
        decision = {'accepted': True, 'answer': r['reference'],
                    'axes': {'content_correct': True, 'steps_complete': True, 'format_compliant': True}}
        with self.assertRaisesRegex(ValueError, 'source-required text'):
            validate_model_review(r, decision)

    def test_board_scope_confusion_is_rejected_but_explicit_unknown_is_allowed(self):
        supported = {'board': 'LubanCat-3', 'soc': 'RK3576'}
        wrong = row('实测内容是什么？', '该结果已经证明鲁班猫4兼容。', evidence={'scope': supported})
        self.assertIn('board_scope_mismatch', row_issues(wrong))
        careful = row('实测内容是什么？', '设备为LubanCat-3/RK3576；现有记录没有验证鲁班猫4。', evidence={'scope': supported})
        self.assertNotIn('board_scope_mismatch', row_issues(careful))
        legacy = row('实测内容是什么？', '该结果证明鲁班猫4兼容。',
                     evidence={'scope': {'verified_board': 'LubanCat-3/RK3576'}})
        self.assertIn('board_scope_mismatch', row_issues(legacy))

    def test_cross_split_case_and_exact_input_are_rejected(self):
        a = row('检查Linux系统版本。', 'cat /etc/os-release', id='a', group='same')
        b = row('检查Linux系统版本。', 'cat /etc/os-release', id='b', group='same')
        issues = split_issues({'train': [a], 'dev': [b]})
        self.assertEqual({x['issue'] for x in issues}, {'cross_split_case_group', 'cross_split_exact_input'})

    def test_duplicate_exact_input_within_split_is_rejected(self):
        a = row('列出当前目录中的文件。', 'ls', id='a', group='case-a')
        b = row('列出当前目录中的文件。', 'find . -maxdepth 1 -type f', id='b', group='case-b')
        issues = split_issues({'train': [a, b]})
        self.assertIn('duplicate_exact_input', {x['issue'] for x in issues})

    def test_batch_encoding_token_count_uses_input_ids(self):
        from scripts.linux_assistant.audit_rebuilt_dataset import input_token_count
        self.assertEqual(input_token_count({'input_ids': [1, 2, 3], 'attention_mask': [1, 1, 1]}), 3)

    def test_eval_context_excludes_reference_answer(self):
        from scripts.linux_assistant.modeling import evaluation_messages
        r = row('问题', 'DO_NOT_LEAK')
        rendered = evaluation_messages({'system_prompt': '只按题目回答'}, r)
        self.assertEqual(rendered[-1], {'role': 'user', 'content': '问题'})
        self.assertNotIn('DO_NOT_LEAK', str(rendered))

    def test_duplicate_assistant_answer_is_blocked(self):
        r = row('请说明检查步骤。', '目标答案')
        r['messages'].append({'role': 'assistant', 'content': '目标答案'})
        self.assertIn('non_alternating_conversation', row_issues(r))
        self.assertIn('duplicated_adjacent_assistant_target', row_issues(r))

    def test_missing_user_turn_is_blocked(self):
        r = row('问题', '回答')
        r['messages'] = [{'role': 'assistant', 'content': '历史回答'}, *r['messages']]
        self.assertIn('non_alternating_conversation', row_issues(r))

    def test_legacy_builders_fail_closed(self):
        from scripts.linux_assistant import (build_unified_dataset, build_quality_dataset,
            build_dataset_v2_candidate, curate_training, prepare, quality_audit, review_training_data)
        calls = [build_unified_dataset.main, build_quality_dataset.build,
                 build_dataset_v2_candidate.main, curate_training.main, prepare.main,
                 quality_audit.finalize, review_training_data.prepare, review_training_data.finalize]
        for call in calls:
            with self.subTest(call=call.__module__ + '.' + call.__name__), self.assertRaises(RuntimeError):
                call()

    def test_default_experiment_config_targets_rebuilt_unified_dataset(self):
        from scripts.linux_assistant.common import DEFAULT_CONFIG, config
        c = config(DEFAULT_CONFIG)
        self.assertEqual(DEFAULT_CONFIG.name, 'linux-assistant-qwen35-2b-quality.json')
        self.assertEqual(c['data'].name, 'linux-assistant-unified')
        self.assertEqual(c['training_review'], 'reports/linux-assistant-rebuild-20261020/content-approval.json')


if __name__ == '__main__':
    unittest.main()
