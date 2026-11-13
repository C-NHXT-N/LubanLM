# LubanLM 实战项目

这里是教程共用的完整代码。我把从零训练小模型、成熟模型 LoRA 微调、推理与 RK3576 部署放在同一处，方便你沿着数据、模型和运行入口阅读实现。

## 代码结构

| 目录 | 用途 |
|---|---|
| `tokenizer/`、`data/` | 分词器训练与评估、语料清洗和编码 |
| `model/`、`training/`、`sft/` | 模型结构、预训练、监督微调与保存恢复 |
| `inference/`、`eval/`、`benchmark/` | 文本生成、聊天、评测与性能测量 |
| `export/`、`deploy/` | 模型导出、转换和板端应用 |
| `configs/` | 模型规格、训练参数与依赖清单 |
| `scripts/` | 环境准备、实验入口、检查和测试 |

## 快速开始

以下 PowerShell 命令从仓库根目录执行。先按[环境说明](../docs/新电脑环境复现.md)准备 Python 3.11、PyTorch、Tokenizers 和 NumPy；已有环境也可以使用。CPU 小实验只用随代码提供的文本，无需下载模型权重或正式数据集。

```powershell
Set-Location project
. .\scripts\workspace_env.ps1
python -m training.reader_lab --output ..\..\runs\reader-lab\part --stop-after 8
python -m training.reader_lab --output ..\..\runs\reader-lab\resumed --resume ..\..\runs\reader-lab\part\last.pt --stop-after 16
```

请使用已安装依赖的 Python 环境；再次实验时换一个空的输出目录。实验会保存词表、检查点和逐步训练记录。更详细的解释见[小模型实验](../docs/低成本必修实验.md)。

## 运行与阅读入口

下面的 Python 模块均从 `project/` 执行，可以先用 `--help` 查看参数。

| 任务 | 入口 | 教程 |
|---|---|---|
| 训练与评估分词器 | `python -m tokenizer.train`、`python -m tokenizer.evaluate` | [第 01 章](../01/README.md) |
| 预训练小模型 | `python -m training.train` | [第 05—06 章](../05/README.md) |
| 成熟模型 LoRA 微调 | `scripts/linux_assistant/` 专用流程 | [第 07 章](../07/README.md)、[操作说明](../docs/Linux助手微调与交付.md) |
| 生成与聊天 | `python -m inference.generate`、`python -m inference.chat` | [第 08 章](../08/README.md) |
| 模型导出与转换 | `export/` | [第 09 章](../09/README.md) |
| 板端应用 | [部署说明](deploy/linux_assistant/README.md) | [第 10 章](../10/README.md) |

原生小模型与成熟 Qwen 模型使用各自的模型结构、Tokenizer 和入口。训练前按对应章节准备输入，使用 `configs/` 中的配置核对路径与参数。成熟模型的固定来源、数据准备和实验结果见[配套资料](../docs/README.md)。

## 文件与产物位置

项目目录存放源码；仓库父目录中的 `datasets/`、`models/`、`runs/`、`exports/` 和 `reports/` 分别存放数据、模型、训练结果、导出与报告。环境脚本会设置默认路径，也支持已有的 `LUBAN_DATA`、`LUBAN_RUNS` 等环境变量。

## 检查与打包

```powershell
python -m scripts.audit_docs
python -m scripts.verify_reader_route --output ..\..\reports\my-reader-route
& .\scripts\package_tutorial.ps1
```

打包范围是整个教程仓库，包含章节与实战代码；输出放在仓库父目录的 `archives/shares/`。私有写作资料、缓存、大型数据和模型权重不会进入发布包。
