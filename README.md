# LubanLM：从零训练教学到成熟模型微调与RK3576部署

本版从[教材基准](docs/教材基准.md)和[低成本必修实验](docs/低成本必修实验.md)进入；实际检查范围见[读者路线复现记录](docs/读者路线复现记录.md)。

本目录可作为独立教程发布，包内内容及验证边界见[发布说明](发布说明.md)。历史记录只用于追溯，不能覆盖现行基准。

原创代码与代码示例采用[MIT许可](LICENSE-CODE)；教材正文与原创插图采用[CC BY-SA 4.0](LICENSE-DOCS)。两者均允许商业使用，第三方材料保留原许可；完整范围与署名方式见[许可说明](LICENSING.md)。

课程面向有Python基础的读者，先用自训50M学习Tokenizer、现代Decoder实现、数据与预训练，再用成熟指令模型完成任务LoRA、推理、量化与鲁班猫4部署。前后两条路线使用不同权重和Tokenizer，以相同原理连接。

从 [课程目录](目录.md) 开始；写作遵循 [章节正文要求](章节正文要求.md)，状态以 [PROJECT_STATUS.md](PROJECT_STATUS.md) 为准。正文优先用连续实例解释输入、运算和输出，区分教学数值、历史实测和待验证步骤。

当前Linux助手采用[统一数据集](docs/数据与实验状态.md)，现为部分发布：606条训练、82条开发、65条封存验收，通用数据为0条。数据未获训练批准，默认基线和训练入口会阻断；复核证据与不足见[重建报告](docs/数据与实验状态.md)。旧数量和实验分数只用于对应历史实验。

| 路线 | 内容与状态 |
|---|---|
| 第0—6章：50M从零训练 | 已有代码、5B预训练与历史SFT；可靠通用问答未获证明 |
| 第7—8章：成熟模型应用 | Qwen3.5-2B作为首轮Linux助手LoRA对象，Qwen3-4B作原模型对照；已新增专用入口，验收状态见实验记录 |
| 第9—11章：量化、RK3576与性能 | 已有复现说明和专用程序；原模型短测完成，新任务模型质量与板端验收分别记录 |
| 第12章：整理与交付 | 分清自训成果、上游模型来源、适配器和板端交付 |
| [附录E](<附录/附录E 继承50M权重的扩容实验.md>) | 继承50M权重至约100M的可选研究，不承诺提升 |

目标板为鲁班猫4 RK3576、8GB内存与64GB存储；优先考察W4A16和短上下文，型号以实际质量、延迟、速度与内存定型。两个成熟原模型已下载、转换并在实际LubanCat-3/RK3576完成短问答；鲁班猫4仍待另验。首轮任务微调见 [Linux助手实验](docs/Linux助手微调与交付.md)，代码存在不等于训练及板端交付通过。0.5B规格和旧基准保留为参考，退出必修训练路线。

## 文件与运行位置

章节正文在 `C0/`—`C12/`，长程序在章节 `program/`，跨章原生实现集中在 `model/`、`training/`、`sft/`、`inference/`。`data/`、`tokenizer/` 是处理源码，`configs/`、`scripts/` 是现有配置与入口，`docs/` 保存教学材料与历史证据。

大数据、检查点、导出和报告分别放在上一级工作区的 `datasets/`、`runs/`、`exports/`、`reports/`。在教程目录的PowerShell中加载 `. .\scripts\workspace_env.ps1` 设置路径；它不安装依赖。正文使用该入口，带旧GPU和盘符的历史脚本只用于追溯。

当前电脑的Python 3.11独立环境、双Qwen推理环境与WSL转换环境已建立，详情见 [ENVIRONMENT.md](ENVIRONMENT.md) 和 [环境复现入口](docs/新电脑环境复现.md)。任务训练环境由Qwen环境另行克隆，保留原推理环境；下载继续绑定实体WLAN，不改代理配置。历史50M入口与能力记录见 [历史对照](docs/50M-SFT历史对照.md)。成熟Qwen使用 `scripts/linux_assistant/` 专用入口，不能直接交给原生训练器或50M导出器。

分享时以本目录为仓库根目录，或运行 `& .\scripts\package_tutorial.ps1` 生成教程ZIP与SHA256。它不包含工作区数据、模型权重或旧工作树；模型、数据和素材分别确认授权后再发布。

目录职责参考 [nanoGPT](https://github.com/karpathy/nanoGPT) 与 [MiniMind](https://github.com/jingyaogong/minimind)，并保留章节与教学程序相对路径。候选依据见 [Qwen模型卡](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) 与 [Rockchip工具链](https://github.com/airockchip/rknn-llm)。
