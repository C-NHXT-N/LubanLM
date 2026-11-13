"""由实际日志生成中文报告、模型说明与轻量复现材料。"""
from scripts.common import ROOT, DATA, RUNS, read_json, write_json, utc
import importlib.metadata
import platform
import shutil
import subprocess
from pathlib import Path
import psutil
import torch

def load(relative):
    path = RUNS / relative
    return read_json(path) if path.exists() else None

def main():
    docs = ROOT / 'docs'
    results = docs / 'run-results'
    results.mkdir(parents=True, exist_ok=True)
    environment = {'created_utc': utc(), 'platform': platform.platform(), 'python': platform.python_version(), 'torch': torch.__version__, 'cuda_runtime': torch.version.cuda, 'RAM_GiB': psutil.virtual_memory().total/2**30, 'GPU': subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total', '--format=csv,noheader'], text=True).strip().splitlines(), 'git': shutil.which('git'), 'vscode': shutil.which('code'), 'WSL_RKLLM_RK3576': 'not_run', 'dependencies': {name: importlib.metadata.version(name) for name in ('transformers','tokenizers','numpy','datasets','safetensors','lm-eval')}}
    write_json(RUNS / 'reports/environment.json', environment)
    for path in (RUNS / 'reports').glob('*.json'):
        if path.name != 'resume-config.json':
            shutil.copyfile(path, results/path.name)
    for name in ('50m-base', '50m-chat'):
        for file in ('summary.json', 'run_config.json'):
            path = RUNS / 'checkpoints' / name / file
            if path.exists():
                shutil.copyfile(path, results/f'{name}-{file}')
    for size in (16000,32000,40000):
        path = RUNS / f'reports/tokenizer-{size}/metrics.json'
        if path.exists():
            shutil.copyfile(path, results/f'tokenizer-{size}.json')
    data = load('reports/data_stats.json')
    tokens = load('reports/tokenization_stats.json')
    base = load('checkpoints/50m-base/summary.json')
    chat = load('checkpoints/50m-chat/summary.json')
    bench = load('reports/benchmark_0.5b.json')
    observed = load('reports/generation_observations.json')
    ceval = load('reports/ceval_results.json')
    lab = load('reports/lab_results.json')
    budget = load('reports/gpu_budget.json') or []
    lines = ['# Windows 实跑报告', '', '> 本报告记录旧短训流程验证，50M 聊天能力未通过验收。新实验的能力门槛与进度见 [50M 聊天训练闭环](50M聊天训练闭环.md)。章节正文完成、代码流程通过和模型能力达标必须分别记录。', '', f'生成时间（UTC）：{utc()}。数值读取本机日志；以下权重均为本地教学产物。', '', '## 正文与流程范围', '', 'C4～C8、C12 的 24 节与附录 A/B/C 的 Windows 内容。50M 为真实预训练、SFT、推理与导出流程验证，尚未达到简单聊天门槛；0.5B 为随机输入短性能基准。WSL、RKLLM、RK3576 未运行。', '', '## 环境', '', f"Python {environment['python']}；PyTorch {environment['torch']}；wheel CUDA runtime {environment['cuda_runtime']}；系统 RAM {environment['RAM_GiB']:.2f}GiB。", '', 'GPU：', '']
    lines += ['- '+row for row in environment['GPU']]
    lines += ['', '依赖已通过 `pip check`，精确版本见 [锁定文件](../configs/requirements-gpu.lock.txt)。本次通过官方 CUDA 12.4 wheel 使用现有驱动，未额外安装 CUDA Toolkit。', '', '## 数据与词表', '']
    if data and tokens:
        wiki = data['wiki']
        lines += [f"Wikipedia 原始 {wiki['raw_documents']:,} 篇，保留 {wiki['kept_documents']:,} 篇，长度过滤 {wiki['rejected'].get('length',0):,} 篇；精确去重，没有执行近似去重。", '', f"训练 {tokens['train']['documents']:,} 篇 / {tokens['train']['tokens']:,} Token；验证 {tokens['validation']['documents']:,} 篇 / {tokens['validation']['tokens']:,} Token。", '', f"BPE 使用 {wiki['tokenizer_training']['records']:,} 个训练侧片段，共 {wiki['tokenizer_training']['characters']:,} 字符；每隔一篇取前 512 字符。完整预训练正文仍保留。", '', f"OASST 长度过滤后训练 {tokens['sft_train']['examples']} 组、验证 {tokens['sft_validation']['examples']} 组；训练监督位置 {tokens['sft_train']['supervised_tokens']:,}，验证 {tokens['sft_validation']['supervised_tokens']:,}。", '', '来源：'+f"[Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) revision `{data['sources']['pretraining']['revision']}`，CC-BY-SA 3.0 / GFDL；[OASST1](https://huggingface.co/datasets/OpenAssistant/oasst1) revision `{data['sources']['sft']['revision']}`，Apache-2.0。", '', '|目标/实际词表|中文诊断 Token（100篇）|英文诊断 Token（1条）|代码诊断 Token（1条）|往返还原|', '|---|---:|---:|---:|---|']
        for size in (16000,32000,40000):
            path = RUNS/f'reports/tokenizer-{size}/metrics.json'
            if path.exists():
                metric = read_json(path)
                by = metric['by_category']
                lines.append(f"|{size}/{metric['vocab_size_including_special_tokens']}|{by['中文']['tokens']}|{by['英文']['tokens']}|{by['代码']['tokens']}|{'通过' if all(x['roundtrip_ok'] for x in by.values()) else '失败'}|")
        lines += ['', '英文/代码是自编诊断，不代表混合预训练比例。主流程选择 32k，并不据这些少量样例宣称它质量最优。']
    lines += ['', '## 训练与基准', '', '|阶段|实际更新步|输入 Token|监督 Token|初始验证 loss|最终验证 loss|训练段秒数|峰值分配 MiB|', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for kind, summary in [('50M Base',base), ('50M Chat',chat)]:
        if summary:
            lines.append(f"|{kind}|{summary['step']}|{summary['input_tokens']:,}|{summary['supervised_tokens']:,}|{summary['initial_validation_loss']:.6f}|{summary['final_validation_loss']:.6f}|{summary['elapsed_seconds']:.2f}|{summary['peak_allocated_mib']:.2f}|")
    lines += ['', '验证默认读取 8 个固定批次，不是全量验证集。输入 Token 计数含 SFT padding；监督位置不包含问题及 padding。训练段耗时包含循环中的校验/保存及最终保存，初始化另计入累计预算。']
    if base:
        lines += ['', f"Base 目标 8,000,000 Token，实际 {base['input_tokens']:,}，停止原因 `{base['stop_reason']}`；未完整遍历全部语料。平均训练段输入速度约 {base['input_tokens']/base['elapsed_seconds']:.2f} tokens/s。"]
    if bench:
        for row in bench['attempts']:
            if row['status']=='passed':
                lines += ['', f"0.5B {row['parameters']:,} 参数，序列 {row['sequence_length']}、微批次 1、累积 4、BF16、Block 检查点。预热 {row['warmup_steps']} 步，计时 {row['measured_steps']} 步：{row['tokens_per_second']:.2f} tokens/s；峰值分配 {row['peak_allocated_mib']:.2f}MiB，峰值保留 {row['peak_reserved_mib']:.2f}MiB。随机 ID 仅测性能。", '', '按短基准速度线性外推：'+ '；'.join(f"{hour} 约 {count/1e6:.2f}M Token" for hour,count in row['training_tokens_estimate'].items())+'。不保证长期运行维持该速度。']
            else:
                lines += ['', f"基准失败设置：序列 {row['sequence_length']}，状态 `{row['status']}`。"]
    lines += ['', '## 数值与交付检查', '']
    for file,label in [('gpu_checks.json','GPU / 因果 / padding / cache / 梯度重算'),('training_checks.json','训练恢复 / 数据隔离 / 回答掩码 / 模板'),('export_checks.json','原生导出重载与真实 CLI')]:
        record = load('reports/'+file)
        lines.append(f"- {label}：{'通过' if record and record.get('passed') else '未完成或未通过'}，见 [原始记录](run-results/{file})。")
    for kind in ('Base','Chat'):
        path = RUNS / f'exports/LubanLM-50M-{kind}/reload_verification.json'
        if path.exists():
            verification = read_json(path)
            lines.append(f"- {kind} 标准 Qwen3 重新加载 logits 最大绝对误差：{verification['logits_max_abs_error']:.8g}。")
    if observed:
        lines += ['', '## 原始生成观察', '', '以下内容由当前模型生成，可能乱码、重复、答错或不能稳定回应；没有人工改写答案。', '', '|权重|问题|原始输出|', '|---|---|---|']
        for row in observed['base_chat']:
            output = row['output'].replace('|','\\|').replace('\n','<br>')
            lines.append(f"|{row['model']}|{row['question']}|{output}|")
        lines += ['', '所有保存点的固定续写、采样差异及生成 ID 数见 [生成记录](run-results/generation_observations.json)。下面 cache 对照使用 FP32；独立 prefill 用 BF16，两类计时不能直接横比。', '', '|cache|总生成秒数（含prefill）|新Token数|新Token/s（含prefill）|', '|---|---:|---:|---:|']
        for row in observed['timing']:
            lines.append(f"|{row['use_cache']}|{row['total_seconds']:.4f}|{row['generated_tokens']}|{row['tokens_per_second_including_prefill']:.2f}|")
        lines += ['', f"Prefill：{observed['prefill']['tokens']} Token / {observed['prefill']['seconds']:.6f} 秒；缓存实际 {observed['cache']['bytes']:,} 字节。两条生成输出一致性由程序断言检查。"]
    if ceval:
        lines += ['', '## C-Eval 工具体验', '', '零样本，计算机网络与操作系统各 10 题，仅用于展示工具流程，不作能力排名。', '', '|任务|acc|acc_norm|', '|---|---:|---:|']
        for name,row in ceval['results'].items():
            lines.append(f"|{name}|{row.get('acc,none',0):.4f}|{row.get('acc_norm,none',0):.4f}|")
        lines += ['', '题目级输出和评分轨迹见 [评测记录](run-results/ceval_results.json)。']
    if lab:
        lines += ['', '## 附录短实验', '', '每个设置重复同一小段训练文本十步，参数量不匹配，词表与初始化可能不同，不作一般质量结论。', '', '|设置|参数量|首步loss|末步loss|tokens/s|峰值MiB|', '|---|---:|---:|---:|---:|---:|']
        for row in lab:
            lines.append(f"|{row['case']}|{row['parameters']:,}|{row['initial_loss']:.4f}|{row['final_loss']:.4f}|{row['tokens_per_second_after_first_2_steps']:.2f}|{row['peak_allocated_mib']:.2f}|")
    lines += ['', '## 时间预算与调整', '', f"累计 GPU 实验时间（保守按各阶段墙钟计时）：{sum(r['seconds'] for r in budget)/60:.2f} 分钟，上限 240 分钟。下载、清洗、CPU 词表训练、CPU 导出与写作不计入该预算。", '', '首次完整正文 BPE 学习在系统内存分配处失败，已调整为有界训练侧片段并成功重跑三组词表；预训练完整正文没有缩小。Miniconda 默认源要求条款，已改用 conda-forge。lm-eval 0.4.9.2 导入 Transformers 5.8 已移除的接口，GPU 环境已改为兼容的 Transformers 4.57.6 并重锁依赖；原 CPU 环境保留。', '', '## 产物位置', '', f'- Base：`{RUNS / "exports/LubanLM-50M-Base"}`', f'- Chat：`{RUNS / "exports/LubanLM-50M-Chat"}`', f'- Tokenizer：`{DATA / "tokenizers/vocab-32000"}`', f'- 检查点与完整日志：`{RUNS / "checkpoints"}`', '', '运行命令见 [Windows 复现说明](Windows复现说明.md)。后续尚未完成的是 C9～C11 的 WSL/RK3576 正文、工具链和板端实测。']
    (docs/'Windows实跑报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    chapter_titles = {4:'数据与语料',5:'训练系统',6:'预训练',7:'SFT',8:'推理',12:'本地发布准备'}
    flags = {4: bool(data and tokens),5: bool(load('reports/gpu_checks.json') and load('reports/training_checks.json') and bench),6: bool(base and ceval and load('reports/export_checks.json')),7: bool(chat and load('reports/export_checks.json')),8: bool(observed and load('reports/export_checks.json')),12: bool(base and chat and ceval and lab and load('reports/export_checks.json'))}
    index = ['# Windows 章节清单','','本次范围：C4～C8、C12 共24节，附录 A/B/C 的 Windows 内容。正文与代码已补齐，执行状态由报告文件判断。','','|章节|正文|对应代码实跑|','|---|---|---|']
    for chapter in chapter_titles:
        notes=[]
        for section in range(1,5):
            path=ROOT/f'C{chapter}/C{chapter}S{section}.md'
            title=path.read_text(encoding='utf-8').splitlines()[0].removeprefix('# ')
            notes.append(f'[{title}](../C{chapter}/C{chapter}S{section}.md)')
        index.append(f"|C{chapter} {chapter_titles[chapter]}|"+'<br>'.join(notes)+f"|{'完成' if flags[chapter] else '运行中或待检查'}|")
    index += ['', '|附录|正文|执行|','|---|---|---|','|A|[深度学习基础](../附录/附录A%20深度学习基础.md)|张量/梯度/交叉熵实跑|','|B|[开发环境](../附录/附录B%20开发环境.md)|Windows GPU 环境验证；WSL与板端未运行|',f"|C|[LubanLM-Lab](../附录/附录C%20LubanLM-Lab.md)|{'短实验完成' if lab else '待运行'}|",'','后续未写/未跑：**C9、C10、C11 的12节**，分别为 WSL 模型转换与量化、RK3576部署、板端性能观察。0.5B正式长期预训练、Base/Chat公开发布也不在本次完成量内。','','完整事实与产物：[实跑报告](Windows实跑报告.md)；命令：[复现说明](Windows复现说明.md)。']
    (docs/'Windows章节清单.md').write_text('\n'.join(index)+'\n',encoding='utf-8')
    if data:
        (docs/'DatasetCard.md').write_text('# LubanCorpus-v1-teaching\n\nWikipedia 中文文章预训练，OASST 中文人工问答 SFT。来源、许可、清洗、划分、token 数和限制见 [实跑报告](Windows实跑报告.md) 与 `run-results/data_stats.json`。\n\n本地保存原始文章 URL、内容哈希、消息编号和对话树编号。精确去重不等于近似去重；按内容哈希划分 Wikipedia，沿用 OASST 原划分。Tokenizer 使用独立于验证侧的有界文章片段。原始数据未再分发。\n',encoding='utf-8')
    for kind, summary in [('Base',base),('Chat',chat)]:
        directory=RUNS/f'exports/LubanLM-50M-{kind}'
        if directory.exists() and summary:
            text=f'# LubanLM-50M-{kind} 教学模型\n\n从随机参数训练的手写 Qwen3 Dense 参考结构，49,558,400 参数，32,000 词表。上下文容量2048，主要预训练窗口256，SFT样本上限512。没有使用Qwen预训练权重。\n\n当前阶段处理输入 {summary["input_tokens"]:,} Token，监督 {summary["supervised_tokens"]:,}，更新 {summary["step"]} 步；最终固定小样本验证loss {summary["final_validation_loss"]:.6f}。Chat从本地Base继续两轮人工问答监督；Base只进行普通文章预训练。\n\n数据来自Wikipedia中文快照（CC-BY-SA 3.0 / GFDL）及OASST1（Apache-2.0），版本与处理记录见课程docs。短教学训练，不能保证事实、身份、算术或对话可靠，可能重复或生成无效字节。\n\n共享聊天模板：系统、用户、助手普通文本标题，BOS起始、EOS结束；详见随附tokenizer_config.json。权重严格映射为标准Qwen3，已重新加载并比对logits。RKLLM转换和RK3576尚未运行。\n\n本地发布准备包，未上传；公开模型许可尚未指定。\n'
            (directory/'README.md').write_text(text,encoding='utf-8')
            (docs/f'ModelCard-50M-{kind}.md').write_text(text,encoding='utf-8')
    print('Generated actual reports and local model cards')

if __name__ == '__main__':
    main()
