"""最终评分汇总与交付。只接受三个完整且针对同一权重的独立评分。"""
from scripts.common import ROOT, RUNS, DATA, read_json, write_json, sha256, utc
from training.train import load_checkpoint
import argparse
import subprocess
import sys
import shutil

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export',action='store_true',help='only export after all capability gates pass')
    args=parser.parse_args()
    state=read_json(RUNS/'reports/pipeline_status.json')
    prefix=state.get('acceptance_prefix','50m-chat-v2')
    if state['phase'] not in ('awaiting_final_acceptance','accepted','not_accepted','capability_passed_export_pending'):
        raise ValueError('pipeline has not produced final acceptance outputs')
    reviews={}
    for split in ('test','core','sampling'):
        raw=RUNS/'reports'/f'{prefix}-{split}.json'
        review=read_json(raw.with_suffix('.review.json'))
        if review.get('answers_sha256')!=sha256(raw) or not review.get('review_complete'):
            raise ValueError('missing, incomplete or mismatched review')
        raw_result=read_json(raw)
        if raw_result['checkpoint_sha256']!=sha256(state['checkpoint']):
            raise ValueError('reviews are not for final checkpoint')
        reviews[split]=review
    capability=all(r['passed'] for r in reviews.values())
    exports_passed=False
    if args.export and capability:
        for kind,checkpoint in [('Base',state['base_checkpoint']),('Chat',state['checkpoint'])]:
            subprocess.run([sys.executable,'-m','export.prepare','--checkpoint',checkpoint,'--output',str(RUNS/f'exports/LubanLM-50M-{kind}')],cwd=ROOT,check=True)
        subprocess.run([sys.executable,'-m','scripts.verify_exports','--base-checkpoint',state['base_checkpoint'],'--chat-checkpoint',state['checkpoint']],cwd=ROOT,check=True)
        exports_passed=read_json(RUNS/'reports/export_checks.json')['passed']
    result=dict(created_utc=utc(),capability_passed=capability,exports_passed=exports_passed,closed_loop_complete=capability and exports_passed,checkpoint=state['checkpoint'],base_checkpoint=state['base_checkpoint'],rates={k:{name:r[name] for name in ('qualified_rate','relevant_readable_rate','looping_rate','eos_rate','category_rates')} for k,r in reviews.items()})
    write_json(RUNS/'reports/final_acceptance.json',result)
    write_json(RUNS/'reports'/f'{prefix}-final_acceptance.json',result)
    state.update(phase='accepted' if result['closed_loop_complete'] else 'capability_passed_export_pending' if capability else 'not_accepted',capability_status='accepted' if capability else 'not_accepted',updated_utc=utc())
    write_json(RUNS/'reports/pipeline_status.json',state)
    results=ROOT/'docs/run-results/chat-v2'
    results.mkdir(parents=True,exist_ok=True)
    for path in (RUNS/'reports').glob('*.json'):
        if not path.name.startswith('verification-') and path.name!='resume-config.json':
            shutil.copyfile(path,results/path.name)
    base=load_checkpoint(state['base_checkpoint'])
    chat=load_checkpoint(state['checkpoint'])
    chat_total=chat.get('cumulative_sft_tokens')
    if chat_total is None:
        for plan_path in (RUNS/'reports').glob('*plan.json'):
            plan=read_json(plan_path)
            if plan.get('name')==state.get('candidate') and plan.get('initial_chat_supervised_tokens') is not None:
                init=chat['settings'].get('init')
                if init and sha256(init)==plan.get('init_sha256'):
                    chat_total=plan['initial_chat_supervised_tokens']+chat['supervised_tokens']
                    break
    tokens=base.get('cumulative_pretrain_tokens',base['supervised_tokens'])
    ledger=read_json(RUNS/'reports/gpu_budget.json')
    lines=['# 50M 聊天能力验收报告','',f"生成时间（UTC）：{utc()}。完成状态：{'通过能力与工程验收，闭环完成' if result['closed_loop_complete'] else '能力通过，工程交付待验证' if capability else '未通过能力验收，闭环未完成'}。",'',f"选定 Base 累计预训练 {tokens:,} Token，本阶段 {base['supervised_tokens']:,} Token；Chat 监督 {chat['supervised_tokens']:,} Token。新实验累计 GPU {sum(r['seconds'] for r in ledger)/60:.2f} 分钟。",'','|检查|合格率|相关可读|循环重复|通过|','|---|---:|---:|---:|---|']
    lines[4]=f"选定Base累计预训练 {tokens:,} Token，本阶段 {base['supervised_tokens']:,} Token；Chat本SFT阶段监督 {chat['supervised_tokens']:,} Token，已核查SFT链累计 {format(chat_total,',') if chat_total is not None else '未知，不能将本阶段冒充全链'}。新实验累计GPU {sum(r['seconds'] for r in ledger)/60:.2f} 分钟。"
    for split,r in reviews.items():
        lines.append(f"|{split}|{r['qualified_rate']:.1%}|{r['relevant_readable_rate']:.1%}|{r['looping_rate']:.1%}|{r['passed']}|")
    lines+=['','原始输出与逐题理由：']
    for split in reviews:
        lines.append(f'- [{split} 原始回答与评分](run-results/chat-v2/{prefix}-{split}.review.json)')
    lines+=['','## 五道基础题原始输出','']
    for row in reviews['core']['results']:
        lines+=[f"用户：{row['question']}",'',f"模型原始回答：{row['answer']}",'',f"评分理由：{row['grading']['reason']}",'']
    lines+=['## 适用范围','', '仅验收中文单轮基础聊天，不据此声称多轮、复杂推理或广泛知识可靠。旧实验的重复输出保持在 [旧流程报告](Windows实跑报告.md)。','']
    if not capability:
        failed=[row for r in reviews.values() for row in r['results'] if not row['qualified']]
        lines+=['## 失败与后续','',f'本次保留 {len(failed)} 条不合格输出及其原因。不得针对最终题补答案后再复用同一验收；下一轮应按错误类型修正数据或训练，另建独立验收集。未自动追加 GPU 预算。','']
    (ROOT/'docs/50M聊天验收报告.md').write_text('\n'.join(lines),encoding='utf-8')
    for folder in ('C4','C5','C6','C7','C8','C12'):
        for path in (ROOT/folder).glob('*.md'):
            text=path.read_text(encoding='utf-8')
            marker='> 最新能力验收结果：'
            text='\n'.join(line for line in text.splitlines() if not line.startswith(marker))+'\n'
            text+=f"\n{marker}{'闭环完成' if result['closed_loop_complete'] else '未完成'}。实际训练量、独立评分和原始回答见 [50M 聊天验收报告](../docs/50M聊天验收报告.md)。\n"
            if result['closed_loop_complete']:
                text=text.replace('当前正在进行以独立聊天验收为终点的续训；','本轮续训已通过独立聊天验收与工程交付检查；')
            path.write_text(text,encoding='utf-8')
    for name in ('Windows章节清单.md','Windows实跑报告.md','Windows复现说明.md'):
        path=ROOT/'docs'/name;text=path.read_text(encoding='utf-8')
        text+='\n最新能力结果见 [50M 聊天验收报告](50M聊天验收报告.md)，旧日志和输出保留以便复查。\n'
        if result['closed_loop_complete']:
            text=text.replace('正文与流程已验证；50M 能力待验收','正文、流程与50M基础聊天验收通过').replace('当前正在进行以独立聊天验收为终点的续训；','本轮续训已通过独立聊天验收与工程交付检查；')
        path.write_text(text,encoding='utf-8')
    if result['closed_loop_complete']:
        for kind in ('Base','Chat'):
            card=f'# LubanLM-50M-{kind} 本地教学模型\n\n'+f'49,558,400 参数，固定32k词表，从本项目随机初始化Base继续训练。选定Base累计 {tokens:,} 预训练Token，Chat本阶段监督 {chat["supervised_tokens"]:,} Token，已核查SFT链累计 {format(chat_total, chr(44)) if chat_total is not None else "未知"} Token。\n\n'+('Chat通过本轮中文单轮基础聊天与工程交付验收，逐题数据见课程docs/50M聊天验收报告.md。' if kind=='Chat' else 'Base用于文本续写与SFT初始化，不能把Chat的问答分数归于Base。')+' 未验收多轮记忆、复杂推理或通用知识可靠性。\n\n数据：Wikipedia中文固定快照、OASST1、BELLE合成短问答及自建模板样本。来源版本、许可及限制见数据清单与来源数据卡。仅本地实验，未公开发布。\n'
            (ROOT/f'docs/ModelCard-50M-{kind}.md').write_text(card,encoding='utf-8')
            (RUNS/f'exports/LubanLM-50M-{kind}/README.md').write_text(card,encoding='utf-8')
        for path in (ROOT/'目录.md',ROOT/'环境与验证记录.md',ROOT/'章节正文要求.md'):
            text=path.read_text(encoding='utf-8')
            text=text.replace('当前正在进行以独立聊天验收为终点的续训；','本轮续训已通过独立聊天验收与工程交付检查；')
            text+='\n本轮50M通过能力与工程交付验收；原始评分及实际训练量见 [验收报告](docs/50M聊天验收报告.md)。该结论只适用于报告记录的选定权重与基础单轮范围。\n'
            path.write_text(text,encoding='utf-8')
    print(result)

if __name__=='__main__':
    main()
