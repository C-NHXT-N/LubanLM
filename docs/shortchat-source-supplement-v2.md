# 中文短对话来源核查与补充

> 历史实验记录：下面的环境、路径与进行中状态属于记录时点。当前学习顺序见[教材基准](教材基准.md)，保留范围见[历史记录说明](历史记录说明.md)。

本次使用 `configs/50m-shortchat-base-compare-v2.json`。数据和运行目录分别是 `D:/LubanLM-Data/shortchat-base-compare-v2`、`D:/LubanLM-Runs/shortchat-base-compare-v2`。v1候选库保留，通过只读连接重新检查；原基座、50M模型结构、训练参数与数量要求均不变。

## 核查发现

- v1各来源能对应固定提交版本和文件SHA256。来源可信不代表每条回答正确。
- v1最终候选326,019条；训练多轮候选5,783条。Belle占训练单轮候选47,438/60,616，受15%上限影响，无法靠候选总量直接凑齐训练集。
- 补充流程只处理ShareGPT、WildChat、MOSS。SmolTalk等来源依赖旧筛选库，未按本轮需求重新检查所有原始记录。
- 对SmolTalk三个缓存分别检查前1,000条；缓存SHA256均与固定版本元数据一致。日常文件的样本中存在短多轮候选，也发现编造童年、生活经历、占位文字和近期电影推荐。这是诊断抽样，不是全量质量认证。
- MiniMind当前文件含工具调用、思考字段、模型身份和训练来源描述；不能原样全收。仅监督通过检查的最终回答，拒收工具协议与错误模型身份。
- LCCC是清华CoAI发布的人类社交对话；[官方加载器](https://huggingface.co/datasets/thu-coai/lccc/blob/main/lccc.py)指向`silver/lccc`的实际数据文件。不是面向AI助手写成的数据，必须额外拒收私事、人设、脏话、依赖外部上下文的片段，仅保留适合助手回答的自足会话。

## 补充措施

1. 在v2重新检查v1公开候选，规则数据另行生成并保留生成标记。
2. 校验并重用SmolTalk、Alpaca、COIG、Magpie原始缓存，恢复可能被旧筛选流程丢弃的短样本。
3. 从固定版本`312afb4f76391145c6902f765bb51691c09a12f5`补充MiniMind的`sft_t2t_mini.jsonl`（1,739,201,170字节）。
4. 从固定版本`5bd582fa28cd7143f2f9c852e08e23089d677c44`补充LCCC的`lccc_base_train.jsonl.gz`（369,854,377字节）。不引入官方验证/测试文件。
5. 全部新增候选使用相同精确/近重复去重、家族隔离、长度、EOS、类别及来源比例限制。每个原会话最多保留一份完整初始样本，不拼接、不生成普通日常对话、不截断回答凑数。

新增下载约1.97GiB，滚动缓存；私有候选库和输出另计，D盘始终保留20GiB。下载绑定`WLAN / 192.168.43.4`，经`hf-mirror.com`且不回退VPN。实际下载完整文件后校验SHA256再入库；网络探测的256字节响应不代表文件已完整下载。

## 状态与恢复

入口：

```powershell
& 'D:/miniconda3/envs/lubanlm-gpu/python.exe' -B -m scripts.compare_shortchat --plan configs/50m-shortchat-base-compare-v2.json --phase auto --wait-for-reviews --download-interface WLAN --download-bind-ip 192.168.43.4
```

后台隐藏运行；新进度以v2目录的`reports/controller.json`、`preparation_progress.json`为准。来源和网络核查记录见`source-supplement-plan.json`、`supplement-source-verification.json`、`network.jsonl`。原缓存只读，新下载入库后只清理本实验的临时文件。

数量不足仍报告缺口并停止；达到数量门槛后冻结数据并等待800条抽样审核。新增来源不是降低审核标准的理由。未完成抽审及GPU预检前不会开始三组微调；不自动开始10万条正式微调。
