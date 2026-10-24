# Win11 训练效率优化实测

> 历史实验记录：下面的环境、路径与进行中状态属于记录时点。当前学习顺序见[教材基准](教材基准.md)，保留范围见[历史记录说明](历史记录说明.md)。

短测时间（UTC）：2026-10-02T14:17:29.203483+00:00 至 2026-10-02T14:30:59.251218+00:00。本报告只说明训练执行效率，聊天能力仍按独立评分验收。

从第 53,810 步、110,202,880 个本轮 Token 的同一检查点开始短测；有效批次为 8，序列长度 256。各候选预热 20 步，测量 100 步，重复三次。测试权重没有覆盖正式模型。

|设置|中位有效输入 Token/s|GPU 平均利用率|整卡峰值 MiB|损失绝对差|梯度相对 L2 误差|结果|
|---|---:|---:|---:|---:|---:|---|
|baseline-2x4|5569|36.1%|2476|0.000000|0.0000%|通过|
|optimized-2x4|5764|37.2%|2476|0.000000|0.0000%|通过|
|optimized-4x2|10606|41.8%|2854|0.000397|0.8340%|通过|
|optimized-8x1|18673|65.3%|3801|0.000043|0.8630%|通过|
|fused-adamw|22848|72.9%|3802|0.000043|0.8630%|通过|
|sdpa|25009|66.9%|3795|0.000521|0.9517%|通过|
|pinned-transfer|27231|72.9%|3795|0.000521|0.9517%|通过|
|sft-baseline-2x4|1784|34.5%|3814|0.000000|0.0000%|通过|
|sft-optimized-2x4|2220|33.6%|3816|0.001263|0.9243%|通过|
|sft-optimized-4x2|4452|35.4%|3816|0.001025|0.9905%|通过|
|sft-optimized-8x1|8927|56.4%|5243|0.000622|0.9555%|通过|

选定预训练设置：`pinned-transfer`，同期短测速度为原设置的 **4.89 倍**。SFT 的有效输入 Token 不包含 padding；其独立短测速度比为 5.00 倍。短测排除了验证与保存时间，正式运行的端到端速度以进度页为准。

```json
{
  "pretrain": {
    "execution": "optimized",
    "attention_backend": "sdpa",
    "optimizer_backend": "fused",
    "pin_memory": true,
    "micro_batch": 8,
    "accumulation": 1
  },
  "sft": {
    "execution": "optimized",
    "attention_backend": "sdpa",
    "optimizer_backend": "fused",
    "pin_memory": true,
    "micro_batch": 8,
    "accumulation": 1
  }
}
```

## 切换与恢复

正式恢复使用保存点 110,202,880 Token。保存时间比较最初漏掉秒以下精度，导致没有等到下一次保存；已修正为按秒比较。本次最近完整日志能确认至少 **9,605,120 Token** 的未保存更新需要重跑。原 GPU 时间全部结算保留，没有清空账本或把重跑次数重复计为已保存进度。

切换前权重保存在 `D:\LubanLM-Runs\chat-v2\checkpoints\performance-switch\source.pt`。迁移保留模型、Adam 状态、随机数、采样位置、步数和学习率；分组变化有浮点误差，不能称为逐位一致。

普通 `--resume` 仍拒绝改变执行配置。只有显式 `--migrate-execution` 才允许有效批次不变的分组和执行后端迁移；验证微批次保持 2，验证使用原手写注意力路径。旧权重没有新字段时使用原设置。

## 正确性、预算与后续监控

已核验采样顺序、因果掩码、padding、矩形 KV cache、SFT 监督 Token 加权、相同设置的状态恢复及非有限损失拦截。短测采用损失绝对差≤0.01、梯度相对 L2≤2%的门槛；不通过的候选不会上线。

实际 attention 运算：`['aten::scaled_dot_product_attention', 'aten::_scaled_dot_product_efficient_attention', 'aten::_efficient_attention_forward', 'aten::log_softmax', 'aten::_log_softmax', 'aten::_log_softmax_backward_data', 'aten::_scaled_dot_product_efficient_attention_backward', 'aten::_efficient_attention_backward']`。只有实际调用记录支持时才称为融合后端；平台不支持 CUDA profiler 时保留错误信息。

所有短测计入原预训练阶段和24小时累计GPU预算，合计最多20分钟。固定5000条开发阶段probe保持原执行设置；完整SFT使用独立v3数据及通过短测的SFT设置。固定验证损失比切换前增加超过0.1时保存异常权重并自动回退；用户已取消定时检查，之后仅按用户询问检查显存、速度、进程和能力评分。

0.5B尚未启动正式训练。本次代码可供它复用，但仍需独立测定微批次与显存。Win11本轮没有新增编译依赖或迁移WSL；GTX1650保留用于桌面。

最新正式训练状态见 [50M续训进度](50M续训进度.md)，能力完成条件见 [训练闭环](50M聊天训练闭环.md)。

上线核验（UTC 2026-10-02T14:36:40.249036+00:00）：控制器 62352、训练子进程 65308；真实磁盘预训练/SFT恢复均通过。全部性能与恢复检查实际GPU耗时 **13.72 分钟**，未超过20分钟。正式训练恢复后的资源原始采样见 `performance_rollout.json`。

2026-10-03追加SFT组批、预取、分桶短测：6,979 / 6,697 / 6,954非填充输入Token/s，未达到15%新收益门槛，预取与分桶均未启用。原始结果见 [第二次性能对照](run-results/chat-v2/sft-correction-performance.json)，教学解释见 [附录D](../附录/附录D%20训练性能诊断与优化.md)。

