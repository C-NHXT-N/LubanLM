"""验收边界回归：不能因总分达标而忽略算术、类别、重复或缺少评分。"""
from scripts.common import ROOT
from eval.chat_suite import suite
from eval.grade_chat_suite import grade
import json
from pathlib import Path

def main():
    folder=ROOT/'.cache/acceptance-verification'
    folder.mkdir(parents=True,exist_ok=True)
    raw=folder/'synthetic-answers.json'
    marks_path=folder/'synthetic-marks.json'
    rows=[dict(r,seed=2026,answer='synthetic fixture only',eos=True) for r in suite('test')]
    raw.write_text(json.dumps(dict(split='test',results=rows)),encoding='utf-8')
    baseline=[dict(id=r['id'],seed=2026,relevant_readable=True,correct=True,looping=False,reason='synthetic gate test') for r in rows]
    def run(modifications):
        marks=[dict(r) for r in baseline]
        for i,change in modifications.items():
            marks[i].update(change)
        marks_path.write_text(json.dumps(dict(marks=marks)),encoding='utf-8')
        return grade(raw,marks_path)
    assert run({})['passed']
    assert run({50:dict(correct=False),51:dict(correct=False)})['passed']
    assert not run({50:dict(correct=False),51:dict(correct=False),52:dict(correct=False)})['passed']
    assert not run({i:dict(correct=False) for i in range(5)})['passed']
    assert run({i:dict(looping=True) for i in range(3)})['passed']
    assert not run({i:dict(looping=True) for i in range(4)})['passed']
    assert not run({i:dict(relevant_readable=False) for i in range(7)})['passed']
    try:
        run({0:dict(correct=None)})
    except ValueError:
        pass
    else:
        raise AssertionError('missing grade accepted')
    print('Acceptance boundaries passed (synthetic fixtures, no model quality claim)')

if __name__=='__main__':
    main()
