"""V3 regression tests for actual false classifications and fail-closed exports."""
import unittest
import tempfile
from pathlib import Path
from unittest.mock import Mock,patch
from scripts.shortchat_common import classify

class V3Tests(unittest.TestCase):
    def test_initialization_hash_uses_actual_weights(self):
        import torch
        from scripts.compare_shortchat import model_weight_hash
        state={'model':{'w':torch.tensor([1.,2.])},'optimizer':{'step':100}}
        other={'model':{'w':torch.tensor([1.,2.])},'optimizer':{'step':999}}
        self.assertEqual(model_weight_hash(state),model_weight_hash(other))
        other['model']['w'][0]=3
        self.assertNotEqual(model_weight_hash(state),model_weight_hash(other))

    def test_outer_request_vs_quoted_identity(self):
        for q in ('生成一封求职申请邮件，包括介绍自己、联系信息和职位申请。',
                  '编辑一段旅游场所的文字。原文本：在这个城市，你可以看到很多美丽的景点。',
                  '根据给定文本，添加一个有趣的结尾。文本内容：你可以看到很多戴帽子的人。'):
            self.assertEqual(classify([dict(role='user',content=q),dict(role='assistant',content='样例')]),'instruction')
        for q in ('你是谁，你有什么作用？','你的名字是什么？','请用一句话介绍你自己。','你能直接看到我的房间吗？','你现在能直接查询网络吗？'):
            self.assertEqual(classify([dict(role='user',content=q),dict(role='assistant',content='样例')]),'boundary')
        self.assertNotEqual(classify([dict(role='user',content='我家有龟龟你可以来我家看'),dict(role='assistant',content='你家欢迎西藏朋友吗')]),'boundary')
        for q in ('最近在学习模拟电路，你能帮我解释一下基本概念吗？','你能帮我安排明天的行程吗？','疫苗的副作用，你有什么看法？','这个功能很有用，你觉得呢？'):
            self.assertNotEqual(classify([dict(role='user',content=q),dict(role='assistant',content='样例')]),'boundary')
        self.assertNotEqual(classify([dict(role='user',content='原文本：在城市里，你可以看到景点。'),dict(role='assistant',content='样例')]),'boundary')

    def test_invalidating_removes_success_and_review_binding(self):
        from scripts.shortchat_preparation import invalidate
        from scripts.shortchat_io import write_json
        cache=Path('.cache/shortchat-tests');cache.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as tmp:
            root=Path(tmp);b=Mock(folder=root/'processed',runs=root/'runs',policy='new-policy');b.folder.mkdir();(b.runs/'reports').mkdir(parents=True)
            for name in ('manifest.json','review-samples.json','review-marks.json'):write_json(b.folder/name,{'old':True})
            write_json(b.runs/'reports/preparation_complete.json',{'status':'success'})
            invalidate(b,'repair')
            self.assertFalse((b.folder/'manifest.json').exists());self.assertFalse((b.folder/'review-marks.json').exists())
            self.assertFalse((b.runs/'reports/preparation_complete.json').exists());self.assertEqual(len(list((b.folder/'superseded').iterdir())),3)

    def test_partial_manifest_cannot_pass_training(self):
        from scripts.compare_shortchat import check_data
        with patch('scripts.common.read_json',return_value={'artifacts':{'hashes':{'train':'hash'}}}):
            with self.assertRaisesRegex(ValueError,'Partial exports'):check_data({'data_root':'unused'})

    def test_public_boundary_missing_or_failed_review_cannot_pass(self):
        from scripts.shortchat_preparation import validate_boundary_review
        sample=[{'id':'one'}];marks=dict(policy='policy',samples_sha256='hash',marks=[])
        with patch('scripts.shortchat_preparation.read_json',side_effect=[sample,marks]),patch('scripts.shortchat_preparation.sha256',return_value='hash'):
            with self.assertRaisesRegex(ValueError,'incomplete'):validate_boundary_review(Path('unused'),'policy')

    def test_partial_export_survives_dev_shortfall_and_resumes(self):
        import sqlite3,json
        from scripts.shortchat_preparation import export_selection
        from scripts.shortchat_io import write_json
        from scripts.shortchat_common import CATEGORIES
        cache=Path('.cache/shortchat-tests');cache.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as tmp:
            root=Path(tmp);b=Mock(folder=root/'processed',runs=root/'runs',policy='policy',plan={'selection_seed':20261010})
            b.folder.mkdir();(b.runs/'reports').mkdir(parents=True);b.db=sqlite3.connect(':memory:')
            b.db.execute('CREATE TABLE candidates(id TEXT,split TEXT,category TEXT,source TEXT,short INT,payload TEXT,rank TEXT,answer_cost INT,chosen INT DEFAULT 0,subset INT DEFAULT 0)')
            for c,n in CATEGORIES.items():
                for j in range(n*20//100000):
                    id=c+str(j);r=dict(id=id,category=c,source='fixture',answer_tokens=[4],assistant_tokens=5)
                    b.db.execute('INSERT INTO candidates(id,split,category,source,short,payload,rank,answer_cost) VALUES(?,?,?,?,?,?,?,?)',(id,'train',c,'fixture',1,json.dumps(r),id,-2))
            b.feasible.return_value={'dev':{'missing':3}};b.inventories.return_value=[]
            counts={(s,c,v):n for s,c,v,n in b.db.execute('SELECT source,category,answer_cost,count(*) FROM candidates GROUP BY source,category,answer_cost')}
            with patch('scripts.shortchat_preparation.SPLITS',{'train':20,'dev':200}),patch('scripts.shortchat_preparation.allocate_quality',return_value=(20,counts,-40)) as allocator:
                self.assertFalse(export_selection(b,['train','dev']))
                self.assertEqual(len((b.folder/'train-tokens.jsonl').read_text().splitlines()),20)
                self.assertFalse((b.folder/'manifest.json').exists())
                self.assertFalse(export_selection(b,['train','dev']));self.assertEqual(allocator.call_count,1)
            b.db.close()

if __name__=='__main__':unittest.main()
