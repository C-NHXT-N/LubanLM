"""Choose only a semantically reviewed development candidate, never a final-test winner."""
import argparse
import json
from pathlib import Path
from scripts.common import write_json,sha256
from scripts.linux_assistant.common import config,DEFAULT_CONFIG,jsonl,reviewed_scores,gate


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    p.add_argument('--run',choices=['lr-1e-4','lr-5e-5'],default='lr-1e-4');a=p.parse_args();c=config(a.config)
    baseline=json.loads((c['report']/'baseline-reviewed.json').read_text(encoding='utf8'))
    b=baseline['models']['qwen35']['scores'];folder=c['runs']/a.run;candidates=[]
    progress=json.loads((folder/'progress.json').read_text(encoding='utf8'))
    if not progress.get('completed'):raise ValueError('Wait for the complete two-epoch run before selecting or starting a second run')
    for entry in baseline['models'].values():
        if sha256(Path(entry['file']))!=entry['sha256']:raise ValueError('Reviewed baseline changed')
    files=list(folder.glob('dev-*.reviewed.jsonl'))
    expected={int(x.name.split('-')[-1]) for x in folder.glob('checkpoint-*') if x.is_dir()}
    found={int(x.name.split('-')[1].split('.')[0]) for x in files}
    if not expected or found!=expected:raise ValueError('Every saved checkpoint needs a complete documented development review')
    for path in files:
        step=int(path.name.split('-')[1].split('.')[0]);rows=jsonl(path)
        expected=jsonl(c['data']/'dev.jsonl')+jsonl(c['data']/'general-dev.jsonl')
        if [r['id'] for r in rows]!=[r['id'] for r in expected]:raise ValueError('Checkpoint selection requires the full current development set')
        scores=reviewed_scores(rows);adapter=folder/f'checkpoint-{step}'
        candidates.append({'step':step,'adapter':str(adapter),'scores':scores,'review_sha256':sha256(path),
                           'adapter_sha256':sha256(adapter/'adapter_model.safetensors'),
                           'passed':gate(b['task'],scores['task'],b['general'],scores['general'],c['acceptance'],b['general_content'],scores['general_content'])})
    # The general-retention gate is part of eligibility, not a secondary tie-break override.
    eligible=[r for r in candidates if r['passed']]
    selected=max(eligible,key=lambda r:(r['scores']['task'],r['scores']['general'],-r['step'])) if eligible else None
    result={'passed':selected is not None,'baseline':b,'candidates':candidates,'selected':selected}
    write_json(c['report']/('first-run-decision.json' if a.run=='lr-1e-4' else 'second-run-decision.json'),result)
    if selected:write_json(c['report']/'candidate-selection.json',selected)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
