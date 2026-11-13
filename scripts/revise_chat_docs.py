"""原位修正文档完成表述，保留旧实验数值与 Obsidian 公式。"""
from pathlib import Path
from scripts.common import ROOT

def main():
    affected=[]
    banner='> 2026-10-02 能力验收修订：旧 50M 实验仅完成工程流程验证，实际回答存在明显重复，聊天能力未达标。当前正在进行以独立聊天验收为终点的续训；状态、门槛与失败后改进步骤见 [50M 聊天训练闭环]({link})。正文完成和代码跑通不能替代模型能力验收。\n\n'
    for folder in ('C4','C5','C6','C7','C8','C12'):
        for path in (ROOT/folder).glob('*.md'):
            text=path.read_text(encoding='utf-8')
            if '2026-10-02 能力验收修订' not in text:
                first,rest=text.split('\n',1)
                text=first+'\n\n'+banner.format(link='../docs/50M聊天训练闭环.md')+rest.lstrip('\n')
            if path.name=='C5S4.md':
                text=text.replace('完整的 50M 预训练另用 `50m-base` 目录。','旧短训验证另用 `50m-base` 目录；该实验没有完成模型能力验收。')
                old='本次 GPU 全流程总上限为 4 小时，其中预训练最多 130 分钟、SFT 最多 40 分钟，其余用于检查、基准、附录实验和评测。累计账本在 `gpu_budget.json`；每个实验持有同一把锁，避免两个进程同时使用预算。到时停止并保存，未完成的目标如实注明，不把时间上限写成已经训练完所有语料。'
                new='旧流程验证的累计 GPU 上限为 4 小时。现在的 50M 聊天训练另开实验，首轮上限为 24 小时：预训练最多 20 小时、SFT 连同候选检查最多 3 小时，最终验收预留 1 小时。加载 `scripts/chat_v2_environment.ps1` 设置新目录和显式预算，旧账本保留。每个 GPU 实验持有同一把锁。到时保存但未通过聊天验收的模型，仍标记为未完成。\n\n`--resume` 用于同一实验的精确恢复，不能修改目标和学习率计划。`--continue-from` 用于开启新的预训练阶段，载入权重与 AdamW 状态，重置新语料采样器和阶段学习率，记录父检查点与累计 Token。两种操作不能混用。'
                text=text.replace(old,new)
            if path.name=='C7S4.md':
                text=text.replace('下一章不再更新权重，拆开推理时“怎样选出一个 Token，又怎样继续下去”的过程。','当前权重未通过聊天验收，仍需要继续预训练与 SFT。通过验收后再作为能力达标的 Chat 交付物；下一章解释推理时“怎样选出一个 Token，又怎样继续下去”的过程。')
                if '## 从检查结果到训练完成' not in text:
                    text+='\n## 从检查结果到训练完成\n\n旧模型在“你好”后反复生成“我可以让你的……”，五道观察题没有证明它能聊天。本轮先用独立开发题选择候选，再完成 60 道验收题、五道基础题和多种子采样检查。每条答案必须保留原文及相关性、正确性、循环重复的评分理由。门槛与命令见 [闭环说明](../docs/50M聊天训练闭环.md)。\n\n如果未通过，先对失败类型分类：连续重复需要检查基础预训练、数据重复与解码；答非所问需要检查问题/回答模板和监督掩码；算术错误需要核查答案和训练覆盖。修改训练后先复查开发题，再进入独立验收。验收未通过时可以导出调试权重，但不得把它写成已经训练好的聊天模型。\n'
            if path.name=='C6S4.md' and '## Base 阶段的检查与后续门槛' not in text:
                text+='\n## Base 阶段的检查与后续门槛\n\nBase 应先用普通文章续写和独立验证集检查语言学习，不能只套聊天模板判断。若语言仍明显破碎或循环重复，应继续改进预训练，再进行短 SFT 检查。阶段 Token 目标完成意味着一次运行结束；完整课程模型的终点，还包括 SFT 后的独立聊天验收。当前短训 Base 未达到这个终点。\n'
            path.write_text(text,encoding='utf-8');affected.append(str(path.relative_to(ROOT)))
    guide=ROOT/'章节正文要求.md'
    text=guide.read_text(encoding='utf-8')
    if '## 模型能力与训练闭环' not in text:
        text+='\n## 模型能力与训练闭环\n\n- 训练章节必须有实际能力检查。解释训练输入与目标，展示原始模型回答，按独立验收规则判断，再写失败分析和改进。不能到保存检查点、导出文件或 CLI 打印文字就宣布闭环完成。\n- 分别说明“正文已完成”“工程流程通过”“模型能力达标”。Base/Chat 是权重阶段名，不是质量保证。模型尚未达标时，在章节状态与 Model Card 明确写未完成，并保留真实失败样例。\n- 预先固定开发题与独立验收题。开发题用于选模型和排错；最终验收题不用于补训练答案或调参。每条原始输出附相关性、可读性、正确性、循环重复与评分理由；不能用关键词命中代替答案理解。\n- 到达步数、Token 或时间上限只意味着运行停止。若能力未达标，先保存可恢复状态并定位问题，在约定预算内改进；预算耗尽仍未达标则如实说明，不把“暂时结束”写成“训练完成”。\n- 本课程当前 50M 采用 [聊天训练闭环](docs/50M聊天训练闭环.md) 中的单轮门槛。实跑结果必须从日志与评分取数，未知结果保持待验收，禁止预填理想回答。\n'
        guide.write_text(text,encoding='utf-8');affected.append(guide.name)
    for name in ('Windows实跑报告.md','Windows章节清单.md','Windows复现说明.md','ModelCard-50M-Base.md','ModelCard-50M-Chat.md'):
        path=ROOT/'docs'/name;text=path.read_text(encoding='utf-8')
        if '2026-10-02 能力验收修订' not in text:
            first,rest=text.split('\n',1)
            text=first+'\n\n'+banner.format(link='50M聊天训练闭环.md')+rest.lstrip('\n')
        if name=='Windows章节清单.md':
            for chapter in ('C6 预训练','C7 SFT','C8 推理','C12 本地发布准备'):
                lines=text.splitlines()
                text='\n'.join(line.replace('|完成|','|正文与流程已验证；50M 能力待验收|') if line.startswith('|'+chapter+'|') else line for line in lines)+'\n'
        path.write_text(text,encoding='utf-8');affected.append(str(path.relative_to(ROOT)))
    for name in ('目录.md','环境与验证记录.md'):
        path=ROOT/name;text=path.read_text(encoding='utf-8')
        if '2026-10-02 能力验收修订' not in text:
            first,rest=text.split('\n',1)
            text=first+'\n\n'+banner.format(link='docs/50M聊天训练闭环.md')+rest.lstrip('\n')
        path.write_text(text,encoding='utf-8');affected.append(name)
    print('Revised',len(affected),'documents')

if __name__=='__main__':
    main()
