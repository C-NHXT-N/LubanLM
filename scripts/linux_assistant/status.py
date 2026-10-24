"""Summarize evidence without promoting implemented code to passed experimental milestones."""
import json
from collections import Counter,defaultdict
from pathlib import Path
from scripts.common import write_json
from scripts.linux_assistant.common import config,jsonl,screen


def main():
    c=config();report=c['report']
    def read(path):
        return json.loads(path.read_text(encoding='utf8')) if path.exists() else {}
    baseline=read(report/'baseline-reviewed.json')
    smoke=read(c['runs']/'smoke/progress.json')
    selection=read(report/'candidate-selection.json');merge=read(report/'merge-check.json')
    conversion=read(report/'conversion-Qwen3.5-2B.json')
    perf=read(report/'board-performance.summary.json');stability=read(report/'board-stability.summary.json')
    boundary=read(report/'board-boundary.summary.json')
    parity=read(report/'board-parity.json');final=read(report/'final-acceptance.json')
    milestones={
        'M1':bool(baseline.get('completed')),
        'M2':bool(smoke.get('completed') and smoke.get('reload_passed') and smoke.get('frozen_base_unchanged') and smoke.get('adapter_changed')),
        'M3':bool(selection.get('passed') and merge.get('passed')),
        'M4':bool(conversion.get('completed') and perf.get('passed') and stability.get('passed') and boundary.get('passed') and parity.get('parity_passed') and (report/'board-dev.reviewed.jsonl').exists()),
        'M5':bool(final.get('passed')),
    }
    tests=read(report/'tests.json');structure=read(report/'lora-structure-check.json')
    experiments=[];second_screen_bounds=[]
    for name in ('lr-1e-4','lr-5e-5'):
        folder=c['runs']/name;progress=read(folder/'progress.json')
        if not progress:continue
        scores=[]
        for path in sorted(folder.glob('dev-*.reviewed.scores.json'),key=lambda p:int(p.name.split('-')[1].split('.')[0])):
            scores.append({'step':int(path.name.split('-')[1].split('.')[0]),**read(path)})
        if name=='lr-5e-5':
            eval_paths=[p for p in folder.glob('dev-*.jsonl') if p.stem.removeprefix('dev-').isdigit()]
            for path in sorted(eval_paths,key=lambda p:int(p.stem.split('-')[1])):
                rows=jsonl(path);task=[r for r in rows if r['kind']=='task']
                second_screen_bounds.append({'step':int(path.stem.split('-')[1]),
                    'task_screen_pass':sum(screen(r,r['answer'])['screen_pass'] for r in task),
                    'task_total':len(task)})
        experiments.append({'run':name,'completed':progress.get('completed',False),
            'step':progress.get('step',progress.get('history',[{}])[-1].get('step')),
            'peak_cuda_allocated_mib':max((r.get('cuda_peak_allocated_mib',0) for r in progress.get('history',[])),default=0),
            'scores':scores})
    state={'milestones':milestones,'tests':tests,'source_review':'author review complete, not independent external review',
           'dataset':str(c['data']),'data_limit':'500 training records from 50 independent authored scenarios',
           'final_tests_run':any((report/f'final-{name}.jsonl').exists() for name in ('original','adapter','merged','board')),
           'final_candidate_locked':(report/'candidate-lock.json').exists(),
           'board_login':'Key authentication unavailable at initial check; do not modify proxy/network configuration.',
           'training_failure':read(report/'training-failure.json'),'experiments':experiments,
           'first_run_decision':read(report/'first-run-decision.json'),
           'second_run_decision':read(report/'second-run-decision.json'),
           'second_run_necessary_condition_bounds':second_screen_bounds}
    write_json(report/'status.json',state)
    diagnostics={}
    review_files=[*report.glob('qwen*-dev.reviewed.jsonl'),*c['runs'].glob('lr-*/dev-*.reviewed.jsonl')]
    for path in review_files:
        rows=[r for r in jsonl(path) if r['kind']=='task'];groups=defaultdict(list)
        for row in rows:groups[row['group']].append(row['accepted'])
        categories={category:{'passed':sum(r['accepted'] for r in rows if r['category']==category),
            'total':sum(r['category']==category for r in rows)} for category in sorted({r['category'] for r in rows})}
        diagnostics[str(path)]={'categories':categories,'independent_groups':len(groups),
            'all_variants_pass_groups':sum(all(values) for values in groups.values()),
            'some_variants_pass_groups':sum(any(values) for values in groups.values()),
            'missing_conditions':dict(Counter(condition for row in rows if not row['accepted'] for condition in row['failed_conditions'])),
            'failed_examples':[{'id':r['id'],'answer':r['answer'],'reason':r['review_reason']} for r in rows if not r['accepted'] and r['id'].endswith('-00')]}
    write_json(report/'development-diagnostics.json',diagnostics)
    lines=['# Linux助手首轮实验：实际执行记录','',
      '实验始于2026-10-19。以下状态由已保存证据生成；程序已实现与实验通过分别记录。','',
      '| 阶段 | 已通过 |','|---|---|',*[f'| {key} | {"是" if value else "尚未确认"} |' for key,value in milestones.items()], '',
      '数据：训练500条、任务开发100条、任务封存验收100条、通用开发和验收各40条。训练来自50个独立场景的10种问法，开发和验收各20个场景的5种问法。统计不能按100个独立任务解释，不能据此声称覆盖整个Linux领域。',
      '', '作者逐场景核对来源，记录于数据目录source-review.jsonl；独立外部人工审阅未完成。草案保留在旧数据目录；训练前将一条与训练事实重合的命令题换为独立用户名命令题，开发输入保持不变。','',
      '原模型开发集结果：','', '| 模型 | 任务通过率 | 严格通用题通过率 | 通用内容正确率 | 已生成回答数 |', '|---|---:|---:|---:|---:|']
    for model in ('qwen35','qwen3'):
        path=report/f'{model}-dev.jsonl';count=len(jsonl(path)) if path.exists() else 0
        score=baseline.get('models',{}).get(model,{}).get('scores',{})
        if not score:
            score=read(report/f'{model}-dev.reviewed.scores.json')
        lines.append(f'| {model} | {score.get("task","评审待完成")} | {score.get("general","评审待完成")} | {score.get("general_content","评审待完成")} | {count}/140 |')
    lines.extend(['','严格字面题仅比较原生助手结束标记之前的内容；JSON题要求直接解析及指定字段。事实内容正确率与严格格式分数分别记录，并同时约束通用能力退化。关键词筛查不等于语义验收，逐题评审另存原回答摘要、结论与理由。结束标记修复前的旧分数已撤回，见stop-policy-incident.json。','',
      f'本地协议与数据测试：{tests.get("passed","待运行")}；测试数：{tests.get("test_cases",0)}。20组原生与轻量分词对照已记录在tokenizer-parity.json，板端一致性另验。',
      f'LoRA结构检查只使用元数据：语言目标层{structure.get("target_module_count","待检查")}个，可训练参数{structure.get("trainable_parameters","待检查")}个；没有据此宣称梯度更新通过。','',
      '板端C++已通过ARM64交叉编译和与固定Runtime的链接检查，见build-check/result.json；该结果不代表板端加载与运行。初次连接已使用原有cat主机公钥校验，但SSH认证失败，需要可用密钥登录后才能执行板端验证。','',
      '独立训练环境、原推理环境包清单与网络配置的前后对照见environment/。模型原文件保持独立，未公开上传。','',
      '复现命令、每个前置条件及验收目标见 ../../LubanLM/docs/Linux助手微调与交付.md。'])
    lines.extend(['','正式实验开发集检查点：','', '| 学习率 | 参数更新 | 任务 | 严格通用 | 通用内容 |','|---|---:|---:|---:|---:|'])
    for experiment in experiments:
        for score in experiment['scores']:
            lines.append(f'| {experiment["run"]} | {score["step"]} | {score["task"]:.0%} | {score["general"]:.1%} | {score["general_content"]:.1%} |')
    if second_screen_bounds:
        lines.extend(['','第二组任务必要条件筛查上界（不是语义通过率）：','','| 学习率 | 参数更新 | 最多可能通过 | 任务题数 |','|---|---:|---:|---:|'])
        for bound in second_screen_bounds:
            lines.append(f'| lr-5e-5 | {bound["step"]} | {bound["task_screen_pass"]} | {bound["task_total"]} |')
    if state['second_run_decision'].get('passed') is False:
        lines.extend(['','第二组5e-5的八个检查点已完成生成；必要条件筛查最高仅25/100条任务答案可能通过，低于至少80/100的任务门槛。筛查是语义通过数的上界，不冒充完整语义评分；因此没有对这组全部1,120条回答制作逐题语义评审。所有检查点均不可能达到门槛，停止在M3。第一组已有完整逐题语义评分，也未达门槛。两组约定实验未产生合格候选，不合并、量化或发布未达标模型；最终验收集继续封存。不得用继续试验旧验收题的方式调参。'])
    elif state['first_run_decision'].get('passed') is False:
        lines.extend(['','第一组未达到开发门槛，按预案仅允许第二组5e-5，其余设置保持相同。'])
    lines.extend(['','评审修正记录：必要条件缺失不能仅以大意正确计为通过；原评审与逐项同义证据保留在required-conditions-review-audit.json所列目录。'])
    if state['training_failure']:lines.extend(['','训练阶段记录了失败，见training-failure.json；按计划停在该阶段诊断，不自动切换模型或扩大数据。'])
    (report/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    m1=['# M1：数据与原模型基线','',*lines[lines.index('原模型开发集结果：'):lines.index('正式实验开发集检查点：')],
        '', '划分与重复检查：data-audit.json；数据摘要与来源复核：datasets/linux-assistant-v1-reviewed/manifest.json、source-review-lock.json。',
        '最终题未运行。通过率仅适用于本轮内部题集与作者评审，不代表独立第三方认证。']
    (report/'M1-data-baseline.md').write_text('\n'.join(m1)+'\n',encoding='utf8')
    m2=['# M2：最小训练、冻结与恢复','',f'通过：{milestones["M2"]}。',
        '16条不同训练记录，微批次1、累积8，共4次更新；第2步恢复后完成第3、4步。',
        f'适配器更新：{smoke.get("adapter_changed")}；冻结原权重摘要不变：{smoke.get("frozen_base_unchanged")}；重新加载推理：{smoke.get("reload_passed")}。',
        '恢复计数、AdamW状态和学习率证据：runs/linux-assistant-v1-20261019/smoke/resume-check.json。',
        f'GPU框架峰值分配：{smoke.get("cuda_peak_allocated_mib")} MiB；当前样本最长164 Token，未认证1024长度的峰值。',
        '训练环境版本、锁与原环境保持对照：environment/；目标模块完整列表：各run/configuration.json。']
    (report/'M2-training-correctness.md').write_text('\n'.join(m2)+'\n',encoding='utf8')
    m3=['# M3：开发集选择与合并','',*lines[lines.index('正式实验开发集检查点：'):],
        '',f'通过选择的候选：{selection.get("adapter","尚无")}；合并验证：{merge.get("passed","未进入")}。']
    (report/'M3-development-selection.md').write_text('\n'.join(m3)+'\n',encoding='utf8')
    (report/'M4-board-delivery.md').write_text('# M4：板端交付\n\n实际板端验收通过：'+str(milestones['M4'])+
        '\n\n已实现轻量分词、常驻C++流式推理、历史裁剪、离线依赖锁、续传校验和独立性能/稳定性/边界测试。'+
        '\nARM64编译链接证据：build-check/result.json；协议替身证据：build-check/worker-stub-test.json。替身不代表NPU实测。'+
        '\n板端初次主机公钥匹配，但SSH认证失败；需可用密钥登录。候选质量、合并、转换、板端一致性及实际60分钟稳定性全部须通过。鲁班猫4仍待验证。\n',encoding='utf8')
    (report/'M5-final-acceptance.md').write_text('# M5：最终验收与教程\n\n最终验收通过：'+str(milestones['M5'])+
        '\n\n最终题是否运行：'+str(state['final_tests_run'])+'；候选是否锁定：'+str(state['final_candidate_locked'])+
        '\n\n已更新README、项目状态与第7—11章实验入口，完整复现说明在Linux助手微调与交付.md。教程更新不能代替尚未达到的质量、部署与验收结果。\n',encoding='utf8')
    print(json.dumps(milestones))


if __name__=='__main__':main()
