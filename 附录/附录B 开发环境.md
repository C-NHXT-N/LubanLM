# 附录 B：开发环境

> 2026-10-18环境更新：当前电脑为RTX 5060 Ti，Miniconda在 `C:\Tools\miniconda3`，项目独立Python 3.11环境、GPU计算及50M加载已验证。WSL2 Ubuntu 22.04.5可启动，RKLLM仍待准备，详见 [ENVIRONMENT.md](../ENVIRONMENT.md) 和 [新电脑环境复现](../docs/新电脑环境复现.md)。下文保留旧4060 Ti电脑的安装与验证记录，旧D盘命令、Python版本和GPU设置不作为本轮安装方案。

Windows侧已使用独立Python 3.11环境，安装入口与国内镜像实体直连方法见新电脑环境复现。WSL侧按选定RKLLM版本另建转换环境，不共享Windows二进制包。阅读下列历史步骤时，先理解隔离环境、依赖与验证的作用，不将旧命令当作新电脑安装记录。

本次使用 Windows 11、RTX 4060 Ti 16GB 与 D 盘 Miniconda。先确认现有安装，避免重复下载：

```powershell
Test-Path D:\miniconda3\Scripts\conda.exe
& D:\miniconda3\Scripts\conda.exe --version
nvidia-smi
```

本机已经有 Miniconda，因此没有再次下载安装器。若另一台机器未安装，从 [Miniconda 官方安装说明](https://www.anaconda.com/docs/getting-started/miniconda/install) 获取 Windows x86_64 安装包，下载到 D 盘，安装到独立 D 盘目录。不要把 Linux 安装包当成 Windows 安装器运行。

创建课程环境，使用 conda-forge 作为本次已验证的软件源：

```powershell
$env:CONDA_PKGS_DIRS = 'D:\miniconda3\pkgs'
& D:\miniconda3\Scripts\conda.exe create --prefix D:\miniconda3\envs\lubanlm-gpu --override-channels --channel conda-forge python=3.10 pip -y
```

创建前先确认该目录不存在；已有环境就复用，不覆盖。现有 `lubanlm` CPU 环境也保留。默认 Anaconda 软件源本次要求额外接受服务条款，因此改用上述源，没有自动代替用户接受条款。

从项目根目录选择环境与缓存：

```powershell
. .\scripts\environment.ps1
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r .\configs\requirements-gpu.txt
python -m pip check
python -m scripts.verify_gpu
```

官方提供 Windows CUDA 12.4 的 PyTorch 2.6.0 wheel，见 [安装版本页](https://pytorch.org/get-started/previous-versions/)。依赖和 pip/Hugging Face 缓存都放在 D 盘。环境脚本使用 PATH 选择解释器，不修改 PowerShell 全局执行策略；可用 `python -c "import sys; print(sys.executable)"` 确认当前进程实际使用的路径。

`nvidia-smi` 的 CUDA 数字表示驱动可支持的上限，不是当前 Python 环境已经安装了该版本 Toolkit。`torch.version.cuda` 表示当前 wheel 配套 runtime。完整 CUDA Toolkit 主要用于编译 CUDA 程序，本课程运行预编译 wheel，先通过真实操作检查，不额外安装 Toolkit 或升级已经可用的驱动。

GPU 名字与 UUID 也要确认：

```powershell
nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

本机脚本固定选择 RTX 4060 Ti；换电脑时先按查询结果修改 `CUDA_VISIBLE_DEVICES`。选择后程序内部显示 `cuda:0`，不会因此变成原来的 GTX 1650。CUDA 初始化失败时，先核对解释器、wheel 和驱动，不要直接修改模型结构。

Git 与 VS Code 不是训练依赖。若已经安装，可以运行 `git --version`、`code --version`；VS Code 的 Python 解释器选择 `D:\miniconda3\envs\lubanlm-gpu\python.exe`。本次不会为了这些可选工具再修改系统配置，实际可用状态写入环境报告。

WSL、RKLLM Toolkit 与 RK3576 属于后续平台。当前只记录 `not_run`，第 3.4 节给出 Linux 工具链流程，第 9～11 章需要另外验证。Linux 专用 wheel 不装进 Windows 环境，板端 Runtime 也不能由 PC 的 PyTorch 导入检查代替。

本机有约 16GB 系统 RAM，BPE 临时统计曾发生内存不足；当前采用有界训练片段，完整预训练文章仍保留。系统 RAM 与 GPU 显存承担不同任务，一项宽裕不代表另一项也够。精确依赖、硬件与检查结果见 [Windows 实跑报告](../docs/Windows实跑报告.md)。
