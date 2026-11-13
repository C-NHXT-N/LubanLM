"""Focused tests for the controlled data-release gate."""
import json
from pathlib import Path
import tempfile
import unittest

from scripts.common import WORKSPACE
from scripts.linux_assistant.release_contract import validate_release_contract,validate_cross_review
from scripts.linux_assistant.common import fingerprint
from scripts.common import sha256


class ReleaseContractTests(unittest.TestCase):
    def _row(self, ident, category, tags, capability=None, multi=False):
        messages=[{'role':'user','content':'条件已完整。'}]
        if multi:
            messages += [{'role':'assistant','content':'请补充日志。'},
                         {'role':'user','content':'日志已经补充。'}]
        messages += [{'role':'assistant','content':'答案。'}]
        row={'id':ident,'category':category,'skill_tags':tags,
             'evidence_refs':['fixture://evidence'],'context_sufficient':True,
             'rubric_support':{'答案':'题面'},'stage_review_status':'accepted',
             'skill_objective':'解决所给条件的问题','input_dependencies':['用户条件'],
             'counterfactual_change':'条件改变时须重新推导',
             'dataset_source':'fixture',
             'messages':messages,'case_id':ident}
        if capability: row['evaluation_capability']=capability
        if capability=='general': row['evaluation_subskill']=tags[0]
        return row

    def test_recomputes_counts_and_token_share(self):
        with tempfile.TemporaryDirectory(dir=WORKSPACE/'.cache') as tmp:
            data=Path(tmp)
            train=[self._row('e','environment_check',['command_generation'],multi=True),
                   self._row('l','linux_operation',['command_generation']),
                   self._row('d','deployment',['document_explanation']),
                   self._row('t','troubleshooting',['log_troubleshooting']),
                   self._row('i','interaction',['instruction_following']),
                   self._row('g','general',['general_arithmetic'])]
            capabilities=['command_generation','log_troubleshooting','sdk_grounded','interaction','general']
            subtype=['general_arithmetic','general_reasoning','general_instruction','general_knowledge']
            held=[]
            for index,capability in enumerate(capabilities):
                tags=[subtype[index % 4]] if capability=='general' else [capability]
                held.append(self._row('h'+str(index),'general' if capability=='general' else 'linux_operation',tags,capability))
            for name in ('dev.jsonl','test.jsonl'):
                (data/name).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in held),encoding='utf8')
            encoded=[{'labels':[-100,1,2]} for _ in train]
            c={'data':data,'dataset_contract':{
                'training_total':6,'training_scenarios':{'environment_check':1,'linux_operation':1,
                    'deployment':1,'troubleshooting':1,'interaction':1,'general':1},
                'multiturn_min':1,'document_records_max':1,'document_supervised_token_fraction_max':1/6,
                'command_and_troubleshooting_min':3,
                'evaluation':{'total':5,'capabilities':{name:1 for name in capabilities},
                    'general_subtypes':{'general_arithmetic':1}}}}
            result=validate_release_contract(c,train,encoded)
            self.assertTrue(result['passed'])
            self.assertEqual(result['command_or_log_troubleshooting_records'],3)

    def test_pending_review_is_rejected(self):
        row=self._row('x','general',['general_arithmetic'])
        row['stage_review_status']='pending_independent_review'
        with self.assertRaisesRegex(ValueError,'Independent review'):
            validate_release_contract({'data':Path('.'),'dataset_contract':{
                'training_total':1,'training_scenarios':{},'multiturn_min':0,
                'document_records_max':0,'document_supervised_token_fraction_max':0,
                'command_and_troubleshooting_min':0,'evaluation':{}}},[row],[{'labels':[1]}])

    def test_rejected_and_unknown_review_states_cannot_release_training(self):
        for state in ('rejected','failed','pending','',True):
            row=self._row('x','general',['general_arithmetic'])
            row['stage_review_status']=state
            with self.subTest(state=state),self.assertRaisesRegex(ValueError,'Independent review'):
                validate_release_contract({'dataset_contract':{'training_total':1}},[row],[{'labels':[1]}])

    def test_crossreview_is_pinned_stratified_and_independent(self):
        with tempfile.TemporaryDirectory(dir=WORKSPACE/'.cache') as tmp:
            data=Path(tmp)
            rows=[{'id':str(i),'category':'a' if i<10 else 'b','split':'train',
                   'dataset_source':'fixture','created_by':'author'} for i in range(20)]
            primary=[{'id':r['id'],'reviewer_id':'primary'} for r in rows]
            (data/'content-review.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in primary),'utf8')
            cross=[{'id':r['id'],'decision':'accept','reason':'independent content check',
                    'reviewer_id':'cross','record_sha256':fingerprint(r)} for r in (rows[0],rows[10])]
            def save(decisions):
                path=data/'cross-review.jsonl'
                path.write_text(''.join(json.dumps(r)+'\n' for r in decisions),'utf8')
                (data/'manifest.json').write_text(json.dumps({'cross_review_file':path.name,
                    'cross_review_sha256':sha256(path)}),'utf8')
            save(cross)
            self.assertEqual(validate_cross_review(data,rows)['reviewed_records'],2)
            save(cross[:1])
            with self.assertRaisesRegex(ValueError,'coverage'):
                validate_cross_review(data,rows)
            cross[0]['reviewer_id']='primary';save(cross)
            with self.assertRaisesRegex(ValueError,'rejected or stale'):
                validate_cross_review(data,rows)
            cross[0]['reviewer_id']='cross';save(cross)
            (data/'cross-review.jsonl').write_text('[]','utf8')
            with self.assertRaisesRegex(ValueError,'missing or changed'):
                validate_cross_review(data,rows)


if __name__=='__main__':unittest.main()
