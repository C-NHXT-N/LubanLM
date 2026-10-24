# Windows 11 复现说明

> 历史实验记录：下面的环境、路径与进行中状态属于记录时点。当前学习顺序见[教材基准](教材基准.md)，保留范围见[历史记录说明](历史记录说明.md)。

> 2026-10-02 能力验收修订：旧 50M 实验仅完成工程流程验证，实际回答存在明显重复，聊天能力未达标。当前正在进行以独立聊天验收为终点的续训；状态、门槛与失败后改进步骤见 [50M 聊天训练闭环](50M聊天训练闭环.md)。正文完成和代码跑通不能替代模型能力验收。

所有命令在 `D:\BaiduSyncdisk\知识库\LubanLM` 的 PowerShell 执行。本机环境是 `D:\miniconda3\envs\lubanlm-gpu`，RTX 4060 Ti 通过 UUID 选择。换电脑时先核对 `nvidia-smi` 的名字与 UUID，再修改环境脚本。

## 环境与已有产物

```powershell
. .\scripts\environment.ps1
python -c "import sys,torch; print(sys.executable); print(torch.__version__,torch.version.cuda); print(torch.cuda.get_device_name(0))"
python -m pip check
```

首次安装过程见 [附录 B](../附录/附录B%20开发环境.md)。环境已创建时不要重新覆盖。精确依赖见 [锁定文件](../configs/requirements-gpu.lock.txt)；若从该文件安装，CUDA wheel 源也要明确提供：

```powershell
python -m pip install -r .\configs\requirements-gpu.lock.txt --extra-index-url https://download.pytorch.org/whl/cu124
```

当前 Base/Chat 在 `D:\LubanLM-Runs\exports`，数据和 Tokenizer 在 `D:\LubanLM-Data`，完整训练检查点在 `D:\LubanLM-Runs\checkpoints`。验证本地交付可直接运行：

```powershell
python -m inference.chat --checkpoint D:\LubanLM-Runs\exports\LubanLM-50M-Chat --question "你好" --temperature 0 --max-new-tokens 24
```

这会增加当前 GPU 账本时间。预算用尽后程序拒绝继续，另开实验应使用新运行目录，保留本次记录。

## 数据与词表

以下步骤是准备流程。当前产物已经完成；在已有目录重新训练 Tokenizer 会更新词表文件，训练好的检查点依赖它的哈希，不应随意替换。完整重做时先保存旧数据版本并使用新数据目录；仅复现训练则复用已审计的同一数据与词表。

```powershell
python -m data.prepare --documents 30000 --sft-pairs 2000
foreach ($vocab in 16000,32000,40000) {
    python .\C1\program\1.3\train_tokenizer.py --input D:\LubanLM-Data\processed\tokenizer-train.jsonl --vocab-size $vocab --output "D:\LubanLM-Data\tokenizers\vocab-$vocab"
    if ($LASTEXITCODE -ne 0) { throw 'Tokenizer training failed' }
    python .\C1\program\1.3\evaluate_tokenizer.py --tokenizer "D:\LubanLM-Data\tokenizers\vocab-$vocab" --input D:\LubanLM-Data\processed\tokenizer-eval.jsonl --report-dir "D:\LubanLM-Runs\reports\tokenizer-$vocab" --hidden-size 512
    if ($LASTEXITCODE -ne 0) { throw 'Tokenizer evaluation failed' }
}
python -m data.tokenize
```

Wikipedia 按已保存内容哈希划分，OASST 保留原消息树划分。原始数据已有时复用固定版本；初次下载的 revision 写入来源记录。BPE 使用 6,463,443 字符的有界片段，完整预训练集约 56M Token，二者不是同一个处理数量。

## 新运行目录中的检查与训练

下面重训命令使用新根目录，不覆盖已交付权重。先确认新目录没有其他训练结果：

```powershell
$env:LUBAN_RUNS = 'D:\LubanLM-Runs-reproduction'
python -m scripts.verify_gpu
python -m scripts.basics
python -m scripts.verify_training
python -m benchmark.train_benchmark --config .\configs\config_0.5b.json --warmup 3 --steps 10
python -m training.train --config .\configs\config_50m.json --tokens 8000000 --minutes 130 --length 256 --micro-batch 2 --accumulation 4 --lr 3e-4 --output D:\LubanLM-Runs-reproduction\checkpoints\50m-base
python -m training.train --config .\configs\config_50m.json --stage sft --init D:\LubanLM-Runs-reproduction\checkpoints\50m-base\last.pt --output D:\LubanLM-Runs-reproduction\checkpoints\50m-chat --epochs 2 --minutes 40 --micro-batch 2 --accumulation 4 --lr 5e-5
```

每步成功后再继续。`verify_training` 在新目录中运行四步微型模型，并从中间点恢复，比较最终参数、优化器和随机状态。它的输入词表与数据仍来自已准备版本。0.5B 用随机 ID 进行性能测量，没有进入正式文章预训练。

恢复正式预训练时保持原来的 Token 上限、批次、长度、累积、学习率与总步数，只增加 `--resume ...\last.pt` 并指向原输出目录。若改变这些设置，需要设计新的继续训练实验；当前严格恢复不会静默改变它们。

## 导出、生成与评测

```powershell
python -m export.prepare --checkpoint D:\LubanLM-Runs-reproduction\checkpoints\50m-base\last.pt --output D:\LubanLM-Runs-reproduction\exports\LubanLM-50M-Base
python -m export.prepare --checkpoint D:\LubanLM-Runs-reproduction\checkpoints\50m-chat\last.pt --output D:\LubanLM-Runs-reproduction\exports\LubanLM-50M-Chat
python -m eval.observe
python -m eval.lm_harness
python -m benchmark.lab
python -m scripts.verify_exports
```

标准目录不覆盖已有导出。验证同时检查严格权重映射、HF 重新加载 logits、原生重新加载和真实 CLI。C-Eval 各取 10 题，是工具体验；附录实验重复短文本十步，是结构与成本观察。报告不能把它们当成一般质量结论。

当前已经交付的事实报告见 [Windows 实跑报告](Windows实跑报告.md)。新实验先保留独立运行根目录，不用新结果覆盖这份历史报告。本次尚未执行 WSL 转换、量化、RK3576 和公开上传。

最新能力结果见 [50M 聊天验收报告](50M聊天验收报告.md)，旧日志和输出保留以便复查。
