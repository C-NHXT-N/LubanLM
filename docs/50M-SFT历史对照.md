# 50M SFT与推理历史对照

> 历史实验记录：下面的环境、路径与进行中状态属于记录时点。当前学习顺序见[教材基准](教材基准.md)，保留范围见[历史记录说明](历史记录说明.md)。

本页提供现行路线与历史实验的索引。50M的训练阶段已结束，第7章以后转向成熟指令模型的任务适配；旧程序和失败样例用于教学，不再作为必须继续续训到可靠聊天的要求。

迁移记录中的最终Base为5B Token、step 406,902，参数量49,558,400；SFT v7完成6,723步，保留best和last。当前文件分别在工作区 `runs/pretrain/checkpoints/last.pt`、`runs/sft/checkpoints/best.pt`、`runs/sft/checkpoints/last.pt`。本页整理来源记录，没有重新加载或评测。

| 历史入口 | 用途与限制 |
|---|---|
| [原生训练器](../training/train.py) | 50M/0.5B预训练与全参数SFT，不能加载成熟Qwen做LoRA |
| [聊天模板](../sft/template.py) | 50M自定义模板，不能套到成熟模型 |
| [生成循环](../inference/generate.py) | 原生权重、32K词表、采样和KV Cache |
| [旧Chat CLI](../inference/chat.py) | 50M单轮交互，不是成熟模型CLI |
| [导出入口](../export/prepare.py) | 自训权重严格映射，不负责LoRA合并 |

历史原始结果见 [Windows实跑报告](Windows实跑报告.md)、[50M聊天验收报告](50M聊天验收报告.md)、[问题分析与修正](50M问题分析与修正.md)。[聊天训练闭环](50M聊天训练闭环.md) 和 [续训进度](50M续训进度.md) 中的预算、门槛和“当前”属于当时阶段，当前路线不沿用其续训承诺。

旧对话记载了本轮三模型生成实验及能力不足，最终迁移说明又称未找到本轮完整对照报告。因此未核对报告、权重哈希和配置前，不把历史报告自动归为最终SFT v7验收证据。最终状态与待核实项见 [项目状态](../PROJECT_STATUS.md)。

修订前的C7、C8、C12等正文已经保存在工作区 `archives/tutorial-route-before-*/`，原迁移工作树在 `archives/original-project-20261016/`，原始代码和检查点未改动。需要重跑历史实验时，先修正旧盘符、GPU限制与数据版本，不直接执行文档中的旧D盘命令。
