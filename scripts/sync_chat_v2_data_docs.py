"""将已完成的数据准备统计同步到对应章节，不预填模型能力结果。"""
from scripts.common import ROOT, RUNS, DATA, read_json, write_json, sha256
import shutil

def main():
    wiki=read_json(RUNS/'reports/wiki-v2.json')
    if 'legacy_seed' not in wiki:
        wiki['legacy_seed']=wiki['source']
        wiki['source']={k:wiki['source'][k] for k in ('repo','revision','config','license')}
        wiki.update(requested_documents=250000,selection='first requested records of fixed source revision, streaming; legacy prefix reused',processed_sha256={s:sha256(DATA/f'processed/wiki-{s}.jsonl') for s in ('train','validation')})
        write_json(RUNS/'reports/wiki-v2.json',wiki)
        data=read_json(RUNS/'reports/data-v2.json');data['wiki']=wiki
        write_json(RUNS/'reports/data-v2.json',data)
    sft=read_json(RUNS/'reports/sft-v2.json')
    public=ROOT/'docs/run-results/chat-v2'
    public.mkdir(parents=True,exist_ok=True)
    for name in ('data-v2.json','wiki-v2.json','sft-v2.json','training_checks.json','continue_checks.json'):
        source=RUNS/'reports'/name
        if source.exists():
            shutil.copyfile(source,public/name)
    updates={
        'C4/C4S4.md':f"本轮已读取 {wiki['raw_documents']:,} 篇固定版本中文 Wikipedia，清洗编码后的训练集有 {wiki['train']['documents']:,} 篇、{wiki['train']['tokens']:,} Token，验证集有 {wiki['validation']['documents']:,} 篇、{wiki['validation']['tokens']:,} Token。处理使用流式读取与分片写入，未重新训练 Tokenizer。数量来自 [新数据清单](../docs/run-results/chat-v2/data-v2.json)，旧的 3 万篇结果保留为流程验证历史。",
        'C6/C6S1.md':'当前新阶段从旧 Base 参数与 AdamW 状态继续学习，目标为新增 4 亿 Token、预训练预算最多 20 小时；这两个上限都不等于能力验收通过。控制器入口为 `python -u -m scripts.train_chat_v2`，先加载 `scripts/chat_v2_environment.ps1`。同一实验中断用严格 `--resume`，另开训练阶段才用 `--continue-from`。',
        'C6/C6S2.md':'新阶段在 1 亿与 2 亿 Token 检查点运行短 SFT，并用固定开发题观察回答是否相关、可读、正确及是否循环重复。最后一个候选也参加相同检查；保存原始输出和逐题理由后选择候选。模型不会因为走到下一章就自动达到聊天目标。',
        'C6/C6S3.md':'新预训练评估使用固定随机种子采样的至少 50 万验证 Token，SFT 评估覆盖全部独立验证问答。旧报告只取 8 个固定批次，不能将两种验证设置的 loss 直接连成同一条曲线。验证 loss 用于观察学习状态，独立聊天题用于判定课程模型是否达到目标。',
        'C7/C7S2.md':f"新 SFT 数据清洗后有 {sft['statistics']['train']['examples']:,} 组训练问答、{sft['statistics']['validation']['examples']:,} 组验证问答。训练监督位置共 {sft['statistics']['train']['supervised_tokens']:,}，验证监督位置 {sft['statistics']['validation']['supervised_tokens']:,}。数据包括保留原划分的 OASST、BELLE 合成短问答和自建规则/模板样本，不能统一称为人工问答。按规范化问题去重与分组划分，并排除开发及验收的同文问题；记录见 [SFT 数据清单](../docs/run-results/chat-v2/sft-v2.json)。",
        'C7/C7S3.md':'本轮先对候选 Base 做固定子集短 SFT，再选择开发题表现最好的候选做最多两轮完整 SFT。两种运行均新建优化器，只监督回答与 EOS，问题及 padding 标签仍为 -100。未通过开发题门槛时先定位失败，不直接把最新 `last.pt` 改名当作训练好的 Chat。',
        'C8/C8S4.md':'新交付必须通过独立聊天验收后，再从导出目录重载并运行真实 CLI。程序能启动、逐字打印或保存 KV Cache 都是工程检查，不能证明回答正确。当前只验收单轮对话，每次问题独立生成；没有验证多轮记忆。',
        'C12/C12S1.md':'新版 Model Card 必须列出选定 Base 的真实累计预训练量、Chat 监督量、六类独立验收结果、原始输出链接和适用范围。未达标时写明未完成；不能把 24 小时预算或 4 亿 Token 目标写成已经完成的训练量。',
        'C12/C12S3.md':'新实验入口与旧流程分开：加载 `scripts/chat_v2_environment.ps1`，运行持久化控制器，并每 30 分钟执行 `python -m scripts.chat_v2_monitor --update-docs` 核对真实状态。开发题评分后控制器继续；最终 test/core/sampling 全部通过后，运行 `python -m scripts.finalize_chat_v2 --export` 完成标准导出、重载和 CLI 检查。',
        'C12/C12S4.md':'课程的 Windows 模型训练终点是独立基础聊天验收与工程交付均通过。旧实验已证明代码路径可以运行，却没有完成能力闭环；本轮继续训练，只有实际评分达标才更新完成结论。预算耗尽未达标时保存结果、分析失败并保留未完成状态。',
    }
    for relative,paragraph in updates.items():
        path=ROOT/relative;text=path.read_text(encoding='utf-8')
        title='## 本轮能力训练的对应步骤'
        if title not in text:
            text+='\n'+title+'\n\n'+paragraph+'\n'
            path.write_text(text,encoding='utf-8')
    print('Synced actual v2 data statistics and training/acceptance steps')

if __name__=='__main__':
    main()
