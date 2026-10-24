# LubanCorpus-v1-teaching

> 历史实验记录：下面的环境、路径与进行中状态属于记录时点。当前学习顺序见[教材基准](教材基准.md)，保留范围见[历史记录说明](历史记录说明.md)。

> 本页记录旧短训数据。新版语料、来源与实际统计见 [聊天训练 v2 数据说明](DatasetCard-chat-v2.md)，语料库存不能当作已完成的训练量。

Wikipedia 中文文章预训练，OASST 中文人工问答 SFT。来源、许可、清洗、划分、token 数和限制见 [实跑报告](Windows实跑报告.md) 与 `run-results/data_stats.json`。

本地保存原始文章 URL、内容哈希、消息编号和对话树编号。精确去重不等于近似去重；按内容哈希划分 Wikipedia，沿用 OASST 原划分。Tokenizer 使用独立于验证侧的有界文章片段。原始数据未再分发。
