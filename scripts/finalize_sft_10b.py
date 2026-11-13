"""Apply recorded startup review decisions without redownloading a candidate pool."""
from scripts.common import ROOT,read_json,write_json,sha256,utc
from pathlib import Path
from collections import Counter
import argparse,json,re

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',default='configs/50m-general-pretrain-10b.json');p.add_argument('--decisions',required=True);args=p.parse_args()
    plan=read_json(args.plan);folder=Path(plan['data_root'])/'processed'/plan['sft']['data_version'];runs=Path(plan['runs_root'])
    report=read_json(folder/'manifest.json');decisions=read_json(args.decisions)
    if report['train_sha256']!=decisions['reviewed_train_sha256']:raise ValueError('Review refers to a different train file')
    rejected={r['group'] for r in decisions['rejected']}
    patterns=re.compile(r'相似度|分词|相反的含义|不到.{0,8}单词|拿铁|冰咖啡|接近沸腾|冷水冲泡|血型不合|癫痫|起身水|PDF文档原有的自动打字|共和党|民主党|[\u4e00-\u9fff]\.[a-zA-Z]{3,}')
    counts=Counter()
    def eligible(row):
        if row['group'] in rejected:counts['review_rejected']+=1;return False
        if patterns.search('\n'.join(m['content'] for m in row['messages'])):counts['review_pattern']+=1;return False
        return True
    rows={};reserved=set();questions=set()
    for split in ('train','validation','test'):
        with (folder/f'{split}-tokens.jsonl').open(encoding='utf-8') as stream:
            original=[json.loads(x) for x in stream if x.strip()]
        rows[split]=[r for r in original if eligible(r)]
        reserved.update(r['group'] for r in original) # Never move an old heldout group into train.
        questions.update(r['question_sha256'] for r in original)
    replacements=[]
    with (folder/'qualified-candidates.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            r=json.loads(line)
            if r['task']!='instruction' or len(r['input_ids'])>384 or not re.search(r'改写|修改|润色|精简|简洁|通俗|概括|提取',r['messages'][-2]['content']):continue
            if r['group'] in reserved or r['question_sha256'] in questions or not eligible(r):continue
            split=next((s for s,n in [('train',100000),('validation',5000),('test',1000)] if len(rows[s])<n),None)
            if split is None:break
            rows[split].append(r);reserved.add(r['group']);questions.add(r['question_sha256'])
            replacements.append(dict(split=split,group=r['group'],task=r['task'],messages=r['messages']))
    assert [len(rows[s]) for s in ('train','validation','test')]==[100000,5000,1000], 'Not enough independent filtered alternatives'
    sets=[{r['group'] for r in rows[s]} for s in ('train','validation','test')]
    assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
    assert sum(r['task']=='daily' for r in rows['train'])>=40000
    # Manifest is the completion marker. Only the CPU controller is active.
    (folder/'manifest.json').unlink()
    for split,items in rows.items():
        tmp=folder/f'{split}-tokens.jsonl.tmp'
        with tmp.open('w',encoding='utf-8') as stream:
            for r in items:stream.write(json.dumps(r,ensure_ascii=False)+'\n')
        tmp.replace(folder/f'{split}-tokens.jsonl')
        report[split+'_sha256']=sha256(folder/f'{split}-tokens.jsonl')
    report.update(categories={s:dict(Counter(r['task'] for r in v)) for s,v in rows.items()},sources=dict(Counter(r['source'] for r in rows['train'])),postfilter_sha256=sha256(Path(__file__)),review_decisions_sha256=sha256(args.decisions),review_removed=dict(counts),quality_status='filtered_and_known_review_errors_removed_semantic_gate_pending')
    write_json(folder/'manifest.json',report);write_json(runs/'reports/sft_filter_report.json',report)
    retained=[r for r in read_json(runs/'reports/sft_semantic_samples.json') if r['group'] not in rejected and not patterns.search('\n'.join(m['content'] for m in r['messages']))]
    write_json(runs/'reports/sft_review_retained_samples.json',retained)
    write_json(runs/'reports/sft_review_replacements.json',replacements)
    write_json(runs/'reports/sft_postfilter.json',dict(created_utc=utc(),manifest_sha256=sha256(folder/'manifest.json'),replacements=len(replacements),counts=dict(counts)))
    print(json.dumps(dict(train=100000,validation=5000,test=1000,replacements=len(replacements),counts=dict(counts))))

if __name__=='__main__':main()
