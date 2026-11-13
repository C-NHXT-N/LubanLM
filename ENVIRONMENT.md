# LubanLM 当前环境与历史迁移记录

教材环境以[教材基准v1](docs/教材基准.md)为准，以下为已有安装与迁移记录的汇总。50M加载、双Qwen原模型推理、转换及LubanCat-3短测已通过；历史LoRA最小更新与恢复及134步实验完成，但任务质量未达标。当前统一数据未获训练批准，本次教材修订没有启动正式任务训练。

| 环境 | 已确认 | 尚待准备或验证 |
|---|---|---|
| Windows 11 | RTX5060Ti 16GB、驱动617.42；50M环境与独立Qwen环境可推理，Qwen使用Transformers5.8.0 | 历史LoRA保存/恢复已验证；新数据训练及任务质量待验收 |
| WSL2 | Ubuntu22.04.5、Python3.10.12；SDK1.3.1 CPU环境依赖检查通过；PyTorch2.7.1+cu128兼容实验环境已导出两模型 | GPU环境偏离SDK两项Torch依赖声明；没有新增Linux Conda/CMake/g++ |
| SSH实际设备 | LubanCat-3/RK3576、约8GB内存、剩余45GB存储、Ubuntu22.04、驱动0.9.8、Runtime1.3.1；两模型已完成加载及八题两轮短测 | 2B约5.52 token/s、进程峰值内存1.45 GiB；4B约3.53 token/s、峰值2.55 GiB。短测不代表长期稳定性，也不等同于已验证教程目标鲁班猫4 |

项目环境位于 `C:\Lab\LubanLM-lab\.conda\lubanlm-py311`，已经登记到Conda；Miniconda仍在 `C:\Tools\miniconda3`。base与系统Python保留，base中未安装torch和numpy。使用项目环境的完整解释器路径或先激活该环境，不把依赖装进base。

已安装NumPy 1.26.4、Transformers 4.57.6、Tokenizers 0.22.2、Datasets 4.1.1、PyArrow 25.0.1、Safetensors 0.5.3、PyYAML 6.0.3、lm-eval 0.4.9.2。首次自动选中的PyArrow 26要求NumPy 2，实际导入失败；现固定旧环境已用的25.0.1并重跑通过。`pip check` 与实际模块导入均通过，精确依赖清单随验收报告保存。

Windows PyTorch构建包含 `sm_120`，已在5060 Ti上实际计算。WSL使用独立Python3.10环境，不与Windows环境混用；详细兼容差异和调度器适配验证见实测入口。

WSL的Windows本地代理未镜像问题通过仅用于安装的互操作管道转发解决，上游仍绑定实体WLAN。SDK已完成两个模型的量化与导出；首次Qwen3.5导出因WSL内存不足被终止并留档，增加任务专用临时交换空间后重试成功，交换文件现已关闭并删除。两个模型均已通过Tailscale中继传输并完成板端短问答，原始结果和适用边界见 [本轮报告](docs/cases/原模型实测摘要.md)。

公共路径与GPU检查已取消旧GPU UUID和4060 Ti名称限制，改为当前工作区默认路径及实际CUDA计算检查；显式环境变量优先。正文统一采用workspace_env.ps1；历史配置中的盘符只用于追溯。原生检查点与成熟模型适配器必须各用正确的恢复入口。

Windows Clash系统代理和Meta/TUN保持启用，依赖下载复用实体WLAN绑定与独立DNS。Conda/PyPI使用清华镜像，固定PyTorch来自南京大学镜像；安装转发器仅在子进程运行期间存在。模型小文件来自HF-Mirror、权重来自ModelScope官方Qwen副本，大小及摘要相符；SDK来自Rockchip官方固定版本。未下载新训练语料，校准问答由本地原模型生成。根据用户后续授权，板端Clash另行增加了持久Tailscale DIRECT规则，保留配置备份并验证热加载。

操作入口及边界见 [新电脑环境复现](docs/新电脑环境复现.md)。实际结果见工作区 `reports/environment-setup-20261018-025357/`，首次失败记录保存在 `reports/environment-setup-20261018-024503/`。三个检查点与Tokenizer SHA256前后相同；短生成仍有重复及事实错误，不作为聊天质量通过结论。

## 历史迁移环境原文（旧电脑与早期建议）

以下保留原有文字与参考命令，仅供追溯，不是当前环境状态或本轮执行说明。其中 `project/` 是原迁移包布局，现已归档；当下源码入口为 `LubanLM/`。旧建议中的独立环境方式不能覆盖本轮确定的Python 3.11 Conda方案，当前依赖版本也需重新核验。

# LubanLM 迁移环境记录

## 当前电脑（2026-10-16）

- Windows；Python 3.10.21
- PyTorch 2.6.0+cu124（PyTorch CUDA build 12.4），Transformers 4.57.6，Tokenizers 0.22.2，NumPy 1.26.4，Datasets 4.1.1，Safetensors 0.5.3，PyYAML 6.0.3，PyArrow 25.0.1，tqdm 4.70.1
- NVIDIA Driver 617.42；GPU：RTX 4060 Ti 16 GB（另有 GTX 1650 4 GB）
- 该环境是旧电脑的本地环境说明，归档不含虚拟环境。

## 新电脑建议

RTX 5060 Ti 使用 Blackwell / sm_120。请新建 Python 3.11 环境并安装支持 Blackwell 的 CUDA 13.0 PyTorch wheel；不要复制旧环境或使用本机 CUDA 12.4 wheel。安装前请在 PyTorch 官方安装页面核对当前 Windows/Python wheel 命令与可用版本：

- https://pytorch.org/get-started/locally/
- https://pytorch.org/blog/pytorch-2-12-release-blog/

参考命令（以官方下载页面当前可用 wheel 为准）：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r .\project\configs\requirements-gpu.txt
```

检查安装：

```powershell
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), torch.cuda.get_arch_list()); print('BF16:', torch.cuda.is_bf16_supported())"
```

确认 GPU 可见且架构列表包含 `sm_120` 或对应 Blackwell kernel。若官方 wheel 版本变化，以官方下载页为准；不要为匹配旧机而强行安装 cu124。
