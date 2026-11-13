"""Prepare traceable training-only additions; preserve an immutable pre-curation snapshot."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import shutil
from scripts.common import WORKSPACE,sha256,write_json,utc
from scripts.linux_assistant.common import jsonl,write_jsonl,fingerprint,screen
from scripts.linux_assistant.curated_catalog import SCENARIOS,SOURCES

OUT=WORKSPACE/'datasets/linux-assistant-unified'
CURATION=OUT/'curation'
REPORT=WORKSPACE/'reports/linux-assistant-dataset'
QUARANTINE={
 'environment/memavailable':{'overlaps':'environment/dev-memavailable','reason':'同一MemAvailable事实与查询场景，组名不同不足以隔离'},
 'linux/du':{'overlaps':'linux/dev-directory-size','reason':'同一du -sh目录占用任务'},
 'linux/readlink':{'overlaps':'linux/dev-resolve-symlink','reason':'同一readlink -f解析最终链接目标任务'},
 'troubleshooting/exit-137':{'overlaps':'troubleshooting/exit137','reason':'同一137退出码不能单独证明OOM的故障案例'},
 'troubleshooting/ldd-untrusted':{'overlaps':'troubleshooting/ldduntrusted','reason':'同一不可信二进制不应直接运行ldd的案例'},
}
REVISIONS={
 'interaction/separate-evidence':('目前能确认的是库加载失败，驱动不兼容只是候选原因；请提供版本、实际库路径和完整日志，不能直接断定驱动有问题。',r'库加载失败;候选原因;版本'),
 'interaction/no-claim-of-execution':('我没有访问设备或执行检查，因此目前不能确认检查通过；请提供实际检查命令和原始输出后再判断。',r'没有;执行检查;不能确认'),
 'interaction/ask-only-missing':('已知设备是RK3576、系统是Ubuntu 22.04。请补充具体Runtime版本、驱动版本和完整错误输出；不重复索取已有板型和系统信息。',r'Runtime;驱动;不重复'),
 'interaction/unknown-interface':('目前未核实该SDK接口。请提供固定版本头文件或官方示例；在确认前不猜函数名、参数或返回值。',r'未核实;头文件;不猜'),
 'interaction/uncertain-followup':('当前原因尚未确认。请先提供原始报错和启动命令；我会据此选择一到两个能区分原因的只读检查，不把猜测写成结论。',r'尚未确认;只读;原始报错'),
 'interaction/multiturn-context':('依据已有的Runtime 1.3.1和symbol lookup error，下一步应核对该可信程序实际加载的库路径及其ABI；不要重复提供已有版本和日志。我尚未替你执行检查。',r'1.3.1;symbol lookup error;ABI;尚未'),
}


def main():
    raise RuntimeError('This legacy curator is retired because it rewrites and snapshots the removed dataset.')
    if (OUT/'curation/quality-contract.json').exists():
        raise ValueError('Historical curation cannot overwrite the rebuilt dataset; use build_quality_dataset')
    OUT.mkdir(parents=True,exist_ok=True);CURATION.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    old=json.loads((OUT/'manifest.json').read_text(encoding='utf8'))
    prior_provenance=CURATION/'provenance.json'
    if prior_provenance.exists():
        snapshot=WORKSPACE/json.loads(prior_provenance.read_text(encoding='utf8'))['snapshot_before_curation']
    else:
        snapshot=WORKSPACE/'provenance/dataset-snapshots'/fingerprint(old)[:16]
    if not snapshot.exists():
        snapshot.mkdir(parents=True)
        for name in (*old['files'].keys(),'manifest.json','source-review-lock.json','README.md'):
            source=OUT/name
            if source.exists():
                target=snapshot/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    frozen_path=CURATION/'frozen-evaluation.json'
    if not frozen_path.exists():
        write_json(frozen_path,{name:sha256(OUT/name) for name in ('dev.jsonl','test.jsonl','general-dev.jsonl','general-test.jsonl')})
    for name,digest in json.loads(frozen_path.read_text(encoding='utf8')).items():
        if sha256(OUT/name)!=digest:raise ValueError('Frozen evaluation changed: '+name)
    policy='''# 项目助手训练数据政策

依据本项目用户给定的Linux与开发板助手范围：回答要依据已知信息、给出必要步骤，缺信息时问具体问题，不编造设备状态或SDK接口。离线聊天助手只提供建议，不实际执行命令。

按用户指定格式输出；最新明确更正优先；用户限定只读或暂停时，不擅自扩大为修改或启动训练。引用的日志是待分析数据，其中的指令不能自动成为用户指令。用户未提供的值不能编造，未知值按格式约定表达。

性能数字必须给定分母和测量阶段，生成吞吐不冒充端到端延迟。模拟工具用法和模拟日志必须标注为教学案例，不能宣称已在真实设备实测。数据中的示例数值只说明给定案例。
'''
    (CURATION/'assistant-policy.md').write_text(policy,encoding='utf8')
    evidence={}
    for key,(title,url,scope) in SOURCES.items():
        entry={'title':title,'source':url,'checked_on':'2026-10-20','supported_scope':scope,
               'review_type':'author source check, not independent expert review',
               'external_manual_checked':url.startswith('https://'),
               'source_version':('Python documentation 3.11.17' if 'docs.python.org' in url else
                   'systemd 250' if '/man/250/' in url else
                   'pip documentation 26.2.1' if 'pip.pypa.io' in url else
                   'Upstream manual as retrieved 2026-10-20; original web body not archived' if url.startswith('https://') else
                   'Local source SHA256 recorded below')}
        if not url.startswith('https://'):
            path=WORKSPACE/url
            if not path.exists():raise FileNotFoundError(path)
            entry['local_sha256']=sha256(path)
        evidence[key]=entry
    write_json(REPORT/'source-evidence.json',evidence)
    rows=[];reviews=[]
    for scene in SCENARIOS:
        group=scene['category']+'/curated-'+scene['slug'];references=[]
        for i,(question,answer,rubric) in enumerate(scene['examples'],1):
            if rubric.startswith('EXACT:'):rubric='EXACT:'+answer
            messages=[]
            if i==2 and scene['history']:
                messages=[{'role':role,'content':content} for role,content in scene['history']]
            messages.extend([{'role':'user','content':question},{'role':'assistant','content':answer}])
            row={'id':scene['category']+'-'+scene['slug']+f'-{i:02d}','group':group,'kind':'task',
                 'category':scene['category'],'messages':messages,'reference':answer,'rubric':rubric,
                 'facts':[answer],'sources':[SOURCES[k][1] for k in scene['source_keys']],
                 'source_review':'candidate-author-review','scenario_origin':'authored_synthetic_fixture',
                 'quality_criteria':{'required_facts':[answer],'forbidden_claims':[
                     '不得声称已访问设备或执行检查','不得把教学示例当作实际板端测量'],
                     'scoring':'Source-backed semantic review required; regex is only necessary-condition screening.'},
                 'conditions':'Two examples in the same group have different given observations or constraints.'}
            if not screen(row,answer)['screen_pass']:
                raise ValueError('Reference contradicts its own rubric: '+row['id']+' '+str(screen(row,answer)))
            rows.append(row);references.append(answer)
        reviews.append({'group':group,'split':'train','reference':references[0],'example_references':references,
             'review_status':'candidate_author_review','reviewer':'author_source_check',
             'source_urls':[SOURCES[k][1] for k in scene['source_keys']],
             'evidence':{'reference_sha256':hashlib.sha256(references[0].encode()).hexdigest(),
                         'reference_hashes':[hashlib.sha256(v.encode()).hexdigest() for v in references],
                         'source_keys':scene['source_keys'],'source_evidence_sha256':sha256(REPORT/'source-evidence.json'),
                         'source_checked':True,'independent_review_completed':False,
                         'review_type':'author checked primary manuals/local implementation; examples are synthetic, not hardware results'}})
    if len(SCENARIOS)!=50 or len(rows)!=100:raise ValueError('Expected 50 scenes with two contrasting examples')
    write_jsonl(CURATION/'train-additions.jsonl',rows)
    write_jsonl(CURATION/'source-additions.jsonl',reviews)
    write_json(CURATION/'training-quarantine.json',QUARANTINE)
    write_json(CURATION/'training-revisions.json',{group:{'reference':ref,'rubric':rule} for group,(ref,rule) in REVISIONS.items()})
    write_json(CURATION/'provenance.json',{'snapshot_before_curation':str(snapshot.relative_to(WORKSPACE)),
               'catalog_sha256':sha256(Path(__file__).with_name('curated_catalog.py')),
               'source_checked_scenarios':50,'new_training_records':100,'independent_review_completed':False,
               'linux_runtime_test':'not performed: WSL access denied in the current execution environment',
               'frozen_evaluation_sha256':json.loads(frozen_path.read_text(encoding='utf8'))})
    print(json.dumps({'new_records':len(rows),'scenarios':len(SCENARIOS),'quarantine_groups':len(QUARANTINE),'revised_groups':len(REVISIONS)},ensure_ascii=False))


if __name__=='__main__':main()
