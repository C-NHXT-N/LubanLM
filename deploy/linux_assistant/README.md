# Linux助手：板端离线启动

此说明随部署包分发。仅在候选通过开发、合并和转换检查后才制作模型包；源码目录本身不含模型，不代表已经验收。

已验证的原模型设备为LubanCat-3/RK3576。新任务模型需在该设备另行验收，鲁班猫4仍待验证。本包针对Ubuntu22.04、Linux ARM64与Python3.10，使用随包RKLLM Runtime 1.3.1。板端需要已有的Python虚拟环境支持和C++编译器。

## 安装与构建

在部署包目录中运行。离线依赖包含在wheels中，安装只进入当前目录的虚拟环境：

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --no-index --find-links wheels --require-hashes -r requirements.lock.txt
PATH="$PWD/.venv/bin:$PATH" sh build.sh
.venv/bin/python frontend.py --root . --parity tokenizer-parity.json > board-parity.json
.venv/bin/python frontend.py --root .
```

构建前校验全部包文件，构建后另存可执行文件摘要。摘要不符、缺少模型或库、分词对照不一致时停止并检查原始文件，不跳过校验。模型首次加载后C++进程保持常驻，聊天不需要PC、PyTorch或联网服务。

## 聊天操作

- 输入中文问题后回车，回答流式输出。
- `/clear`清空对话；`/quit`退出；Ctrl+C中断当前回答。
- 中断、错误和空回答不会写入后续历史；中断后会重新加载推理进程。
- 输入最多1792 Token，输出最多256 Token，总上下文2048。历史超限删除最早完整轮次，保留系统提示和当前问题；当前问题仍过长时提示缩短。

助手提供检查和操作建议，不自动执行问题中的命令。Runtime诊断保存在runtime.log，与聊天结果分开。

## 保存验收证据

每次输出使用新文件名，程序拒绝覆盖已有测量。以下命令分别生成原始回答及统计报告：

```sh
.venv/bin/python acceptance.py --root . --mode dev --questions dev-questions.jsonl --output board-dev.jsonl
.venv/bin/python acceptance.py --root . --mode performance --output board-performance.jsonl
.venv/bin/python acceptance.py --root . --mode stability --output board-stability.jsonl
.venv/bin/python acceptance.py --root . --mode boundary --output board-boundary.jsonl
```

稳定性测试持续至少60分钟并完成至少100次请求，包含10组五轮对话。性能测试预热后运行20条短输入。边界测试会主动中断一次生成并验证随后还能请求。质量必须逐题评审，成功返回或关键词命中不能视为质量通过。

保留board-parity.json、全部原始JSONL、对应summary.json、runtime.log和build-manifest.json。最终题只在候选锁定后单独传入，按锁定流程运行一次；不要自行拿最终题调参。

包内bundle-manifest.json列出模型、分词器、运行库、依赖、源码及说明的SHA256。传输不完整时保留.part续传文件；先核对前缀和完整摘要，再形成正式文件。
