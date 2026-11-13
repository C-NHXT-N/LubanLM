"""Preparation diagnostics may collect failures but must not approve content."""
import unittest
from scripts.linux_assistant.preparation_audit import inspect_records,review_coverage
from scripts.linux_assistant.common import fingerprint


class PreparationAuditTests(unittest.TestCase):
    def test_reference_cannot_violate_its_own_explicit_format(self):
        row=self.row();row.update(reference='```sh\npwd\n```',
                                 grading={'format':{'kind':'command_only'}})
        row['messages'][-1]['content']=row['reference']
        result=inspect_records({}, {'train':[row]})
        self.assertIn('reference_violates_own_format',{x['issue'] for x in result['issues']})

    def row(self):
        return {'id':'a','group':'g','case_id':'c','created_by':'author',
                'messages':[{'role':'user','content':'只输出 pwd 命令。'},
                            {'role':'assistant','content':'pwd'}],
                'reference':'pwd','category':'linux_operation',
                'skill_objective':'选择当前目录命令','input_dependencies':['当前shell'],
                'counterfactual_change':'改为主目录时不能只使用pwd',
                'skill_tags':['command_generation'],'evidence_refs':['manual'],
                'rubric_support':{'命令':'题面明确要求pwd'},'context_sufficient':True}

    def test_collects_missing_metadata_and_cross_split_case_together(self):
        row=self.row();row.pop('skill_objective');row.pop('input_dependencies')
        other={**self.row(),'id':'b'}
        report=inspect_records({}, {'train':[row],'dev':[other]})
        issues={x['issue'] for x in report['issues']}
        self.assertIn('missing_skill_objective',issues)
        self.assertIn('missing_input_dependencies',issues)
        self.assertIn('cross_split_case_group',issues)
        self.assertIn('cross_split_exact_input',issues)

    def test_changing_group_name_does_not_waive_case_isolation(self):
        first=self.row()
        second={**self.row(),'id':'b','group':'renamed','messages':[
            {'role':'user','content':'换种说法给出命令'},{'role':'assistant','content':'pwd'}]}
        report=inspect_records({}, {'train':[first],'test':[second]})
        self.assertIn('cross_split_case_id',{r['issue'] for r in report['issues']})

    def test_changing_case_and_group_does_not_waive_source_entry_isolation(self):
        first={**self.row(),'source_atom_id':'official:entry-1'}
        second={**self.row(),'id':'b','case_id':'new','group':'renamed',
                'source_atom_id':'official:entry-1','messages':[
                    {'role':'user','content':'改写同一来源题目'},{'role':'assistant','content':'pwd'}]}
        report=inspect_records({}, {'train':[first],'test':[second]})
        self.assertIn('cross_split_source_atom',{r['issue'] for r in report['issues']})

    def test_hash_change_self_review_and_conflicting_decision_block_acceptance(self):
        row=self.row()
        decision={'id':'a','decision':'accept','reviewer_id':'independent',
                  'record_sha256':fingerprint(row),'reason':'内容逐条核对'}
        self.assertFalse(review_coverage([row],[decision],'primary'))
        self.assertTrue(review_coverage([row],[{**decision,'reviewer_id':'author'}],'primary'))
        self.assertTrue(review_coverage([{**row,'reference':'changed'}],[decision],'primary'))
        self.assertTrue(review_coverage([row],[{**decision,'decision':'reject'}],'primary'))
        self.assertTrue(review_coverage([row],[decision,decision],'primary'))

    def test_crossreview_needs_a_different_reviewer(self):
        row=self.row()
        decision={'id':'a','decision':'accept','reviewer_id':'primary',
                  'record_sha256':fingerprint(row),'reason':'checked'}
        self.assertTrue(review_coverage([row],[decision],'cross',{'a':{'primary'}}))

    def test_digest_aliases_are_compatible_but_conflicts_are_rejected(self):
        row=self.row();decision={'id':'a','decision':'accept','reviewer_id':'other',
                                 'final_row_sha256':fingerprint(row),'reason':'checked'}
        self.assertEqual(review_coverage([row],[decision],'primary'),[])
        decision['record_sha256']='different'
        issues=review_coverage([row],[decision],'primary')
        self.assertIn('primary_conflicting_digest_aliases',{item['issue'] for item in issues})


if __name__=='__main__':unittest.main()
