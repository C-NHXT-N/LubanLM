# 中文短对话与三个基座对照实验

> 历史实验记录：下面的环境、路径与进行中状态属于记录时点。当前学习顺序见[教材基准](教材基准.md)，保留范围见[历史记录说明](历史记录说明.md)。

配置：[50m-shortchat-base-compare.json](../configs/50m-shortchat-base-compare.json)。入口为 `scripts.compare_shortchat`，使用 `D:/miniconda3/envs/lubanlm-gpu/python.exe`。

本实验的数据目录为 `D:/LubanLM-Data/shortchat-base-compare-v1`，运行目录为 `D:/LubanLM-Runs/shortchat-base-compare-v1`。旧候选库通过只读连接访问；原2B、12B最佳、12B最终检查点只用于初始化和校验。不会追加预训练，也不会自动进行10万条正式微调。

## 程序阶段

1. `--phase prepare`：复用候选库，构造明确标记的基础任务和能力边界样本，按固定提交版本依次补充ShareGPT、WildChat、MOSS。新下载经过校验、入库后删除本实验临时原始文件。失败保留游标及下载断点。
2. 数量和来源限制通过后冻结100,000训练、5,000验证、200开发、1,000测试及20,000对照子集，并生成800条分层审核样本。数据不足时输出 `reports/data_shortfall.json`，不会降低标准或启动训练。
3. 审核人在 `processed/review-marks.json` 中逐条记录严重错误、分类正确、回答可用及理由。必须绑定审核样本哈希和完整数据哈希。训练600条、验证200条分别满足严重错误0、分类正确且可用≥95%才能继续。
4. `--phase auto` 全量核查来源版本、文件校验值、样本身份、编码、所有助手标签和EOS、来源上限、长度比例、家族及原始会话隔离，然后进行独立副本批量预检。
5. 三个基座顺序微调同一20,000条数据，连续两轮，每组2,500步。使用新优化器、学习率2e-5、预热125步、有效批量16、种子2026。统一使用16×1；预检不通过则统一8×2。第1,250和2,500步验证并保存 `epoch-01.pt`、`epoch-02.pt`。
6. 自动生成六份200题开发集输出，状态停在 `awaiting_dev_review`。离线盲审完成后重新运行同一 `--phase auto`，按固定规则选择每组较好轮次，再生成三份1,000题测试输出。
7. 测试盲审完成后重新运行同一入口，生成比较表。未完成评分、生成中断或预算暂停均不进入排名。差距不足2个百分点按并列处理；未达到任务门槛时不宣称达到目标。

## 启动与恢复

```powershell
& 'D:/miniconda3/envs/lubanlm-gpu/python.exe' -B -m scripts.compare_shortchat --plan configs/50m-shortchat-base-compare.json --phase auto --wait-for-reviews
```

指定本次物理网卡及其IPv4地址时，追加 `--download-interface WLAN --download-bind-ip 192.168.43.4`。地址必须当前属于指定的物理网卡且具有网关，否则明确报错，禁止改走VPN或其他网卡。控制器将参数传给下载子进程；也可通过 `LUBAN_DOWNLOAD_INTERFACE`、`LUBAN_DOWNLOAD_BIND_IP` 环境变量配置独立下载器。网卡地址再次变化时，使用新的参数恢复同一实验，沿用已提交游标及 `.part` 下载断点。

后台启动时必须设置隐藏窗口和D盘日志；物理网卡探测子进程也使用隐藏窗口。控制器锁防止重复启动。程序通过CPU/GPU阶段顺序执行控制资源，不需要agent实时监控。

`--wait-for-reviews` 使后台程序在三个审核门槛处仅等待完整标记文件，写入后自动继续。等待期间不占用GPU，也不启动新的终端窗口。未使用该选项时，审核完可手动重新运行同一入口。数量缺口、审核失败、运行错误或总预算耗尽仍会停止，并保留明确状态。

`--phase audit` 仅执行冻结数据及审核核验。审核不通过时，`scripts.repair_shortchat --plan configs/50m-shortchat-base-compare.json` 删除本实验问题家族并同类补齐、重新抽样，最多三次。必须重新审核；不能复用旧数据版本的审核标记。

正常预算暂停从最近的完整检查点恢复优化器、学习率、随机状态、消费者采样器和覆盖审计。优化器更新报错的试验禁止静默续跑；须定位错误后重新开始该组。两轮的样本顺序摘要和监督token数必须在三组间完全一致。

## 审核文件格式

数据审核标记包含 `samples_sha256`、原样复制的 `manifest.json/artifacts`，以及覆盖全部800个样本ID的 `marks`。每个标记提供布尔值 `severe_error`、`category_correct`、`usable` 和非空 `reason`。

生成的 `evaluations/<随机别名>/<dev或test>-blind.json` 隐藏基座和轮次，保留题目、参考、回答。语义审核标记写入同目录 `<dev或test>-marks.json`，包含 `raw_sha256` 与 `marks`；只填写未由程序核验的题目。每项提供 `id`、布尔值 `relevant_readable`、`correct`、`looping` 及非空 `reason`。禁止用EOS、损失或文本长度替代语义判断。

## 日志与资源

- `reports/preparation_progress.json`：当前准备阶段、当前会话处理计数和内存；`availability.json`/`data_shortfall.json`：已入库候选的实际库存与缺口。
- `reports/network.jsonl`：物理网卡、绑定IP、实际HTTPS目标与重试证据。禁止VPN或虚拟网卡回退。
- `reports/controller.json`：训练、评估、待审核、暂停或失败状态。
- `reports/experiment_binding.json`：冻结数据、配置、三个父检查点和训练/生成代码校验值。
- `reports/training_fairness.json`：三组实际采样顺序、完整覆盖和监督token一致性证据。
- `reports/gpu_budget.json`：累计GPU预算，最大8小时。预算不足则保存进度并暂停。
- `reports/comparison.json`、`comparison.md`：全部评分完成后的最终比较。

新增空间预算10–20GB；D盘最低保留20GiB；准备进程内存上限3GiB。训练部分原估20–40分钟，准备和语义审核另计，生成评估逐题记录实测剩余时间。数据门槛通过前不会产生正式对照训练结果。
