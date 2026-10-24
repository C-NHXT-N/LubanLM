# Linux助手微调与板端交付

当前数据与后续实验应使用[统一数据说明](数据与实验状态.md)及[重建报告](数据与实验状态.md)。现为部分发布：训练606条、开发82条、封存验收65条，通用数据均为0；没有训练批准，默认入口会阻止基线和微调。本页后文保留的首轮教学配置、数字和旧数据集路径均为历史记录，不代表当前有效数据。当前数据工作没有启动基线、微调或模型评测。

本实验把“怎样确认板端程序与系统架构一致”作为任务样例：先保存原模型回答，再用有来源的问答训练LoRA，最后检查微调是否改善回答。实验配置是 `configs/linux-assistant-v1.json`。以下命令对应已经新增的入口；有入口不等于已经训练、合并或通过板端验收，实际完成情况以工作区 `reports/linux-assistant-v1-20261019/` 的证据为准。

第一版使用已下载的Qwen3.5-2B，Qwen3-4B仅作为原模型比较。50M继续承担前六章的从零训练案例。实际可连接设备此前识别为LubanCat-3/RK3576，不能据此写成鲁班猫4已经验收。本实验只提供操作建议，不自动执行用户提出的命令。

## 在本机建立可比较的输入

从工作区根目录打开PowerShell，设置源码目录并使用独立训练解释器：

```powershell
$env:PYTHONPATH='C:\Lab\LubanLM-lab\LubanLM'
$env:PYTHONUTF8='1'
$assistantPython='C:\Lab\LubanLM-lab\.conda\qwen-lora-py311\python.exe'
```

该环境由现有Qwen环境离线克隆，再安装PEFT等训练依赖。实际安装版本、依赖检查和原环境前后对照保存在本轮 `environment/`。不要将训练依赖装入50M或原Qwen环境。当前环境使用完整解释器路径，不依赖Conda全局环境登记。

（历史实验）当时的数据目录为 `datasets/linux-assistant-v1-reviewed/`：训练500条、任务开发100条、封存任务验收100条，另有开发和验收通用题各40条。训练样本来自50个独立场景，每个场景有10种问法；开发和验收分别来自20个场景，每个场景有5种问法。**这些是措辞变体，不是500条独立知识事实。** 该目录已退役，不是当前活动数据集。

同一场景的改写不会跨划分。150条训练记录含先前对话。每条记录保留来源、参考事实和规则；逐场景来源核对记录为 `source-review.jsonl`，作者复核不冒充独立外部人工复核。初次数据草案目录仍保留；在训练前发现的命令题重合已在reviewed版本修订，最终验收题未用于生成或调参。

准备入口拒绝覆盖已有清单。需要新实验时先复制配置，改为新的数据、运行、报告和导出目录，再使用 `--config`，不要修改已冻结版本。检查现有数据与模板可运行：

```powershell
& $assistantPython -m unittest scripts.linux_assistant.test_pipeline -v
& $assistantPython -m scripts.linux_assistant.parity
```

输入先由原生聊天模板渲染，再编码一次。训练标签仅包含最后一轮助手正文及结束符，问题、系统提示、历史和填充位置不计算损失。第一版500条样本的最大实际输入长度为164 Token；配置允许1024，不代表已经实测1024长度的训练显存。超过上限的样本会报错，不会静默截断答案。

运行原模型开发集基线：

```powershell
& $assistantPython -m scripts.linux_assistant.evaluate --model qwen35
& $assistantPython -m scripts.linux_assistant.evaluate --model qwen3
```

两次运行串行使用GPU。固定相同系统提示、greedy和最多256个输出Token，保留输入ID、完整原始回答和停止原因。`screen_pass`仅检查必要的字面条件，不能证明事实正确；正确同义词可能漏判，包含错误扩展内容的回答也可能通过筛查。逐题评审另存决定，必须绑定原回答摘要、明确通过与否并给出理由，不能把筛查分数直接用于选模型。

```powershell
& $assistantPython -m scripts.linux_assistant.review `
  --answers reports/linux-assistant-v1-20261019/qwen35-dev.jsonl `
  --decisions reports/linux-assistant-v1-20261019/qwen35-dev.decisions.jsonl
& $assistantPython -m scripts.linux_assistant.review --baseline
```

最后一条命令要求两个模型均有完整的140题评审。明确要求精确字面输出的题按原生助手结束标记之前的字符比较，包括正文中的尾部换行；结束标记之后的模板换行不属于回答。每项预设必要条件均须满足；正则未命中的同义表达，评审必须在 `equivalents` 中逐项说明实际文本证据，不能用“大意正确”免除缺失步骤。精确字符、JSON和行数要求不能以同义为由豁免。每道通用题必须同时填写 `content_correct`，严格通过率和事实正确率均不得下降超过5个百分点。科学题接受正确同义表达，也检查附加解释是否有错误。

后续检查点可用 `python -m scripts.linux_assistant.review_template --answers 回答文件` 起草评审。只有问题、原始回答、参考事实和规则逐字相同，且旧评审没有冲突时，才携带旧决定与来源文件摘要；其他项保留为空，必须逐题评审。正式评审入口再次校验复用来源，不能以草案或关键词筛查替代语义判断。

本轮发现并修复过停止策略错误：Qwen3.5的语言配置结束ID为248044，而原生助手结束标记为248046。训练监督位置始终包含正确的248046；早期评测却只等待248044，导致继续生成模板尾部甚至下一轮。现同时接受两个结束ID。原始轨迹完整保留，只重建首次结束标记之前的贪心前缀，并针对两个原模型及前两个检查点各重放20个独立开发场景；Token ID全部相同后才采纳。未重放题的修正耗时留空，不能冒充重新测量。详见 `stop-policy-incident.json` 和各 `replay-check.json`。早期宽松的必要条件评审也已保留并纠正，见 `required-conditions-review-audit.json`；报告只使用修正后的分数。

## 先检查一次更新和恢复

LoRA保留原权重，新增rank16、alpha32、dropout0.05的低秩增量。程序从实际模块中列出语言解码器的线性层，包括线性注意力和完整注意力的投影与MLP；视觉模块、词嵌入和输出头保持冻结。新增参数名称和数量写入每次运行配置，不用普通Qwen的目标层短名单代替混合架构检查。

```powershell
& $assistantPython -m scripts.linux_assistant.train --stage smoke
```

检查使用16条来自不同训练场景的记录，每条见两次；微批次1、累积8，完成4次参数更新。第2步保存，重新加载原模型和适配器，恢复优化器、调度器、计数和随机状态，再完成第3、4步。程序对冻结参数计算摘要，核对适配器变化、梯度与有限损失，并进行一次生成。任何检查失败都停在此阶段，记录 `training-failure.json`；不自动切换模型。

正式训练使用相同批次和BF16，学习率1e-4、2个epoch、warmup比例0.05，之后线性衰减。按助手监督Token数加权累积损失，避免短长回答的微批次产生非预期权重。训练入口要求来源复核和双模型基线评审均完成：

```powershell
& $assistantPython -m scripts.linux_assistant.train --stage formal
```

每16次更新及训练结束保存检查点并运行开发集。对应回答逐题评审后，按任务通过率、通用题通过率、较早步数顺序选择满足门槛的候选。

```powershell
& $assistantPython -m scripts.linux_assistant.select --run lr-1e-4
```

只有第一组被记录为开发集未通过时，才允许 `train --stage formal --second` 以5e-5从原模型重训；最多两组。恢复原实验使用 `--resume` 指向本实验的检查点，身份不匹配会拒绝恢复。低验证损失不能代替任务得分，封存验收题不参与选择。

## 将增量变成可部署模型

```powershell
& $assistantPython -m scripts.linux_assistant.merge
```

合并入口只接受通过开发门槛的候选。用20个独立开发场景对照合并前和保存重载后的下一Token分布，报告最大logit差、JS散度和首选Token变化。预设门槛为每题JS散度不超过0.001，至少19题首选Token相同；BF16合并不要求位级一致。随后运行完整开发题，产生128条训练侧校准记录，排除开发和验收题。

WSL转换沿用已验证的独立SDK1.3.1环境；该GPU环境与SDK的两项Torch依赖差异仍需披露，不能称为官方完整兼容环境。下面路径对应当前工作区。转换命令只有在合并检查通过、校准文件生成后才可运行：

```powershell
wsl -d Ubuntu-22.04 -- /home/aiyong/.venvs/lubanlm-rkllm-gpu310/bin/python `
  /mnt/c/Lab/LubanLM-lab/LubanLM/scripts/convert_qwen_rk3576.py `
  --model Qwen3.5-2B --device cuda `
  --model-dir /mnt/c/Lab/LubanLM-lab/exports/linux-assistant-v1-20261019/merged `
  --calibration /mnt/c/Lab/LubanLM-lab/exports/linux-assistant-v1-20261019/data_quant.json `
  --output-dir /mnt/c/Lab/LubanLM-lab/exports/linux-assistant-v1-20261019/rk3576 `
  --report-dir /mnt/c/Lab/LubanLM-lab/reports/linux-assistant-v1-20261019
& $assistantPython -m scripts.linux_assistant.board_dependencies
& $assistantPython -m scripts.linux_assistant.bundle
& $assistantPython -m scripts.linux_assistant.transfer --verify-only
```

新转换固定W4A16、GRQ、双NPU核心和2048上下文，不复用原模型GRQ参数，也不覆盖原量化文件。SDK可能在合并目录产生派生缓存；模型权重、配置和来源另有摘要记录。若内存不足，记录失败并定位，不把先前原模型成功当成本次成功。

## 板端自己处理文本

板端只需要Python、`tokenizers`、Jinja2和C++程序。Python用随包原生模板和Tokenizer处理文本；C++常驻加载RKLLM，每次接收完整输入Token，清理运行时历史后推理，回传流式字节与性能数据。字节在Python中增量解码，避免中文字跨回调边界时损坏。

传输入口使用已保存的cat主机公钥，要求可用SSH密钥登录。部分文件以 `.part` 保留；续传前核对已有前缀摘要，全部接收后检查完整SHA256，再形成正式文件。不修改Clash、TUN、路由或设备系统配置。

```powershell
& $assistantPython -m scripts.linux_assistant.transfer
```

打包前在Windows运行 `python -m scripts.linux_assistant.board_dependencies`，从既有实体WLAN下载通道准备Python3.10 ARM64 wheels及带摘要的完整依赖锁。打包将这些文件一并携带，板端安装无需联网，也不包含PyTorch。传输后在板端实验目录执行以下命令；安装只针对项目虚拟环境，不修改系统Python。

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --no-index --find-links wheels --require-hashes -r requirements.lock.txt
PATH="$PWD/.venv/bin:$PATH" sh build.sh
.venv/bin/python frontend.py --root . --parity tokenizer-parity.json > board-parity.json
.venv/bin/python frontend.py --root .
```

`/clear`清空会话，`/quit`退出，Ctrl+C中断当前回答。输入超过1792 Token时，只删除最早完整的用户/助手轮次，保留系统提示和当前问题；当前问题仍太长则提示缩短。中断、空回答和错误回答不进入后续历史。

开发题、性能和稳定性分别测量：

```sh
.venv/bin/python acceptance.py --root . --mode dev --questions dev-questions.jsonl --output board-dev.jsonl
.venv/bin/python acceptance.py --root . --mode performance --output board-performance.jsonl
.venv/bin/python acceptance.py --root . --mode stability --output board-stability.jsonl
.venv/bin/python acceptance.py --root . --mode boundary --output board-boundary.jsonl
```

性能题预热后计时，输入含系统提示且不超过128 Token。端到端首字延迟包括本地模板、分词和流式文本解码；decode统一以非结束Token数量与首末Token回调的墙钟间隔计算，包含回调开销，Runtime原始性能字段另外保留。原生助手结束标记到达时主动停止，不等待另一个文本结束标记。Python与C++进程的RSS每0.1秒采样相加，报告峰值及系统可用内存；驱动内存不一定全部计入RSS。稳定性包含10组五轮会话，持续至少60分钟且至少100次请求；模型质量仍需另行评审。

## 最后一轮只用来验收

开发评审、合并、板端分词、质量与稳定性全部通过后，回传相应报告并锁定候选：

```powershell
& $assistantPython -m scripts.linux_assistant.final_round lock
& $assistantPython -m scripts.linux_assistant.final_round pc
& $assistantPython -m scripts.linux_assistant.transfer --final-round
```

锁定文件绑定配置、数据、适配器、合并目录及板端模型摘要。生成的140道最终题及锁定文件再传到板端，运行 `acceptance.py --mode final --questions final-questions.jsonl --final-lock candidate-lock.json --output final-board.jsonl`。板端为该轮建立不可重复启动标记，PC只允许在相同锁下恢复同一轮未完成部分。所有原始结果评审后运行 `final_round assess`。

目标是：任务基线不足80%时，微调后至少80%且提升至少10个百分点；基线已达80%时，错误减少至少30%。通用题下降不超过5个百分点，量化后任务下降不超过5个百分点。板端短输入生成速度中位数至少4 token/s、首字P95不超过3秒，合计RSS峰值不超过3GiB。它们都是工程验收目标，实际未达到时如实保存失败，不改写成通过。
