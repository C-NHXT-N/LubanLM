import unittest
from scripts.linux_assistant.qualify_pilot import qualify
from scripts.linux_assistant.preparation_audit import CATEGORIES
from scripts.linux_assistant.common import fingerprint


class QualifyPilotTests(unittest.TestCase):
    def fixture(self):
        rows=[{'id':f'{c}-{i}','category':c,'created_by':'author'} for c in CATEGORIES for i in range(10)]
        def decision(row,reviewer):
            return {'id':row['id'],'record_sha256':fingerprint(row),'decision':'accept',
                    'reviewer_id':reviewer,'reason':'fixture content checked'}
        primary=[decision(r,'primary') for r in rows]
        source=[{**decision(r,'primary'),'source_verified':True,'source_hash_verified':True,
                 'license_reviewed':True,'safety_reviewed':True,'license':'fixture-only',
                 'safety_decision':'clear'} for r in rows]
        cross=[decision(rows[i],'cross') for i in range(0,60,10)]
        engineering={'engineering_passed':True,'row_fingerprints':{r['id']:fingerprint(r) for r in rows}}
        return rows,primary,source,cross,engineering

    def test_complete_independent_pilot_allows_expansion_but_never_training(self):
        result=qualify(*self.fixture())
        self.assertTrue(result['pilot_method_qualified'])
        self.assertFalse(result['training_approved'])

    def test_rejected_row_or_stale_hash_cannot_qualify_method(self):
        rows,primary,source,cross,engineering=self.fixture()
        primary[0]['decision']='reject'
        self.assertFalse(qualify(rows,primary,source,cross,engineering)['pilot_method_qualified'])
        primary[0]['decision']='accept'; rows[0]['content']='changed'
        self.assertFalse(qualify(rows,primary,source,cross,engineering)['pilot_method_qualified'])

    def test_same_reviewer_and_missing_license_block_qualification(self):
        rows,primary,source,cross,engineering=self.fixture()
        cross[0]['reviewer_id']='primary'; source[1]['license']=''
        result=qualify(rows,primary,source,cross,engineering)
        self.assertFalse(result['pilot_method_qualified'])
        issues={r['issue'] for r in result['issues']}
        self.assertIn('cross_reviewer_not_independent',issues)
        self.assertIn('source_license_safety_review_incomplete',issues)


if __name__=='__main__':unittest.main()
