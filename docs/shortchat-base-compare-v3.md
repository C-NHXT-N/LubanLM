# 中文短对话三基座对照 v3

> 历史实验记录：下面的环境、路径与进行中状态属于记录时点。当前学习顺序见[教材基准](教材基准.md)，保留范围见[历史记录说明](历史记录说明.md)。

v3 复用 v2 的 443,176 条候选，原文件只读；目录为 `D:/LubanLM-Data/shortchat-base-compare-v3` 和 `D:/LubanLM-Runs/shortchat-base-compare-v3`。配置为 `configs/50m-shortchat-base-compare-v3.json`。

分类优先判断外层写作、编辑和提取任务；身份/能力只识别直接询问助手的意图。三条已知 Belle 写作样本经逐条语义检查，两条改回指令类，编造求职资历的一条删除整个家族。不合适的 LCCC 社交反问删除整个家族。新增加回答混入用户问题、LCCC 反问而没有助手回答等清洗检查。保留已有会话 ID、来源、上下文和标签，冻结前重新编码核对 EOS。

开发集身份/能力整组由人工编写的 10 题替换，问候、身份、现实行动、感知、联网各 2 题，明确计入生成上限。登记独立评测家族；逐字及 SimHash 近重复检查覆盖保留候选和旧测试题，冲突家族排除，不尝试改划分种子。

## 准备、审核和续接

准备入口支持 `--prepare-splits train validation dev test subset`，默认全部。开发集不足时先导出合格的其他集合；`processed/preparation-state.json` 分别记录完成文件和缺口，只有全部完成并通过结构检查才发布 `manifest.json`。子集从完整训练集抽取，取整时首先满足短回答比例，再比较分层余数和种子。

`public-boundary-review-samples.json` 是所有保留的公开来源身份/能力候选，必须逐条审核，评分绑定文件哈希和候选政策。`review-samples.json` 是按类别、来源和长短回答分层的 600 条训练及 200 条验证样本。两个集合分别需要严重错误 0、分类正确且回答可用 ≥95%。审核记录须逐条填写理由；不得批量填“通过”。

修复入口 `scripts.repair_shortchat` 先作废完整清单和审核绑定，再删除问题家族并重新选取，最多三次；选取失败不遗留成功清单。旧样本及旧审核保存在 `processed/superseded`。

## 后台运行

控制器 `scripts.compare_shortchat --plan configs/50m-shortchat-base-compare-v3.json --download-interface WLAN --download-bind-ip <当前物理地址> --wait-for-reviews`。地址明确作为参数，必须在物理网卡上实际存在；没有合法绑定时停止，禁止 VPN/虚拟网卡回退。无缺口时复用缓存，不下载。

控制器先验证数据和所有审核记录，再进行独立 GPU 批量预检。三组使用统一的 16×1 或 8×2、同一 20,000 条、全新优化器、连续两轮 2,500 更新、LR 2e-5、warmup 125、种子 2026，保存六个轮末检查点。训练和生成评测顺序执行，子进程隐藏窗口；D 盘保留 20 GiB，准备内存限 3 GiB，总 GPU 预算 8 小时。

开发生成完成后程序等待语义盲审，评分不齐不会选轮次；三组选定轮次的独立测试同样需盲审，缺失评分不得排名。控制器可以等待审核文件，不需要 agent 实时监控训练。正式 100,000 条微调不会自动开始。训练时长估计由 GPU 预检/训练实测更新，不能把原先 20–40 分钟当作保证。

进度以 `reports/preparation_progress.json`、`processed/preparation-state.json`、`reports/controller.json` 为准。“内容待审核”“部分生成”“评估待审核”均不代表实验完成。
