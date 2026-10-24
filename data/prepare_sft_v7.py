"""由训练侧语义抽查提出的公共任务筛选；不读取开发/最终原始答案。"""
from scripts.common import DATA,RUNS,read_json,write_json,sha256,utc
from data.prepare import load_jsonl,save_jsonl
from collections import Counter
import re

def public_rejection(row):
    q,a=row['user'],row['assistant']
    if any(s in q for s in ('GPT模型','GPT-','ChatGPT')):return 'external_model_meta_instruction'
    if any(s in q for s in ('同义词','近义词','反义词','主语','谓语','宾语','句子成分')):
        return 'specialized_lexical_or_grammar_task'
    if any(s in q for s in ('最近','最新','近年来','当前','今年','目前','实时')) and not re.search(r'\d{4}年',q):
        return 'unanchored_time_sensitive_question'
    if any(s in q for s in ('行车路线','预计到达','天气预报','空气质量','空气指数','治疗','服药','投资','股票')):
        return 'unverified_dynamic_or_high_stakes_task'
    if any(s in q for s in ('最大公约数','最小公倍数','求和','求根','计算','数字','等于','平方根','概率','微分','积分')):
        return 'public_calculation_outside_verified_curriculum'
    if any(s in q+a for s in ('假设的文本','假设文本','没有提供','未提供','没有给出','示例文章','一篇包括')):
        return 'missing_context_or_placeholder'
    return None

def main():
    parent=DATA/'processed/sft-v6';target=DATA/'processed/sft-v7'
    if target.exists():raise ValueError('v7 already exists; retain immutable version')
    rejected=[];rows={}
    for split in ('train','validation'):
        rows[split]=[]
        for row in load_jsonl(parent/f'{split}.jsonl'):
            reason=public_rejection(row) if row['task']=='public' else None
            if reason:
                rejected.append(dict(split=split,question_sha256=row['question_sha256'],reason=reason));continue
            rows[split].append(row)
    assert not {r['family_group'] for r in rows['train']} & {r['family_group'] for r in rows['validation']}
    target.mkdir(parents=True)
    for split in rows:
        save_jsonl(target/f'{split}.jsonl',rows[split])
        save_jsonl(target/f'{split}-tokens.jsonl',[{k:r[k] for k in ('input_ids','labels','assistant_tokens','encoding_version','prompt_tokens','source','task','family_group','question_sha256')} for r in rows[split]])
    save_jsonl(target/'rejected.jsonl',rejected)
    result=dict(created_utc=utc(),parent_manifest_sha256=sha256(parent/'manifest.json'),
                counts={s:len(r) for s,r in rows.items()},tasks={s:dict(Counter(r['task'] for r in v)) for s,v in rows.items()},
                rejected=len(rejected),rejection_types=dict(Counter(r['reason'] for r in rejected)),encoding_version='prefix-stable-v1',
                train_sha256=sha256(target/'train-tokens.jsonl'),validation_sha256=sha256(target/'validation-tokens.jsonl'),
                policy='Task-scope filters motivated by seeded training-side semantic audit, not test answers. Remaining public rows are not claimed fully fact-certified. Preserve all v6 files and same authored curriculum. Validation definition changes, no direct loss comparison across versions.')
    write_json(target/'manifest.json',result);write_json(RUNS/'reports/sft-v7.json',result)
    print(result)

if __name__=='__main__':main()
