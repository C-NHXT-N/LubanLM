# Qwen双模型RK3576实测入口

本实验使用本地原模型 `Qwen3-4B-Instruct-2507` 与 `Qwen3.5-2B`，只验证文本问答，不涉及LoRA或视觉编码器。原始PC八题与权重来源见 [PC报告](cases/原模型实测摘要.md)。转换、板端回答与性能的最终状态见 [本轮检查目录](cases/原模型实测摘要.md)。文件存在或PC可运行，均不代表NPU部署已通过。

## 实验配置

- 官方SDK固定为1.3.1，Git提交 `f7390530443bf84f0394255a449d7cbe81e69d1c`；获取文件时校验Git blob，记录SHA256。
- 每模型128条校准样本，由该模型的本地BF16实例生成，使用原生模板，排除八道测试题；每条最多96个新Token，本次均未达到上限。校准数据不是训练集或独立能力评测集。
- 目标为RK3576、W4A16、GRQ、双NPU核心、2048上下文。Qwen3.5只导出语言模型分支，不能据此宣称图像能力已部署。
- 实际SSH设备标识为LubanCat-3、RK3576、8GB内存、Ubuntu22.04，驱动0.9.8；板端Runtime为1.3.1，文件哈希与官方SDK一致。原教程的鲁班猫4目标与本次实际设备应分别记录。

## 环境差异必须披露

官方依赖环境位于工作区 `.conda/rkllm-py310`，采用PyTorch2.6.0 CPU版，依赖检查和SDK导入通过。SDK会将CPU上的BF16请求改成FP32，GRQ逐层处理较慢；本轮CPU尝试主动中止，保留原始日志。

针对RTX5060Ti的Blackwell架构，另建WSL环境 `/home/aiyong/.venvs/lubanlm-rkllm-gpu310`，采用PyTorch2.7.1+cu128及torchvision0.22.1+cu128。该环境通过SDK导入、GPU识别和实际CUDA矩阵运算检查，但**偏离SDK声明的torch2.6.0、torchvision0.21.0依赖**。两条声明冲突保留在报告中，不称为官方完整兼容环境；转换是否可用仍以实际构建和板端结果为准。

GPU环境通过 `.pth` 复用CPU环境内的SDK和其他固定依赖，两个环境须一并保留。首次GPU安装的fsspec冲突已通过固定2025.9.0修复，未更改Windows的50M与Qwen环境。

SDK的GRQ还使用了已在PyTorch2.7删除的调度器 `verbose` 参数。首轮GPU构建因此失败并留档。转换入口现只在GPU路径安装兼容层：接受 `verbose=False`，保持原学习率参数，拒绝 `verbose=True`；LinearLR的小张量学习率序列与原实现逐项一致，证据为本轮报告中的 `scheduler-compatibility-test.json`。不修改已安装Torch或SDK源码。

## 当前工作区执行入口

以下脚本针对当前 `C:\Lab\LubanLM-lab` 与WSL Ubuntu-22.04。将来迁移目录时，需调整脚本内的WSL路径，不能原样视为跨平台安装器。

在教程目录PowerShell中设置模块路径后使用：

```powershell
$env:PYTHONPATH=(Get-Location).Path
& C:\Tools\miniconda3\python.exe -B -X utf8 -m scripts.prepare_rk3576_assets
& C:\Tools\miniconda3\python.exe -B -X utf8 -m scripts.setup_rkllm_wsl
& ..\.conda\qwen-py311\python.exe -B -X utf8 -m scripts.prepare_qwen_calibration
& C:\Tools\miniconda3\python.exe -B -X utf8 -m scripts.setup_rkllm_wsl --gpu
wsl -d Ubuntu-22.04 -u root -- python3 /mnt/c/Lab/LubanLM-lab/LubanLM/scripts/rkllm_temporary_swap.py
try {
    & C:\Tools\miniconda3\python.exe -B -X utf8 -m scripts.run_rkllm_conversion --gpu
} finally {
    wsl -d Ubuntu-22.04 -u root -- python3 /mnt/c/Lab/LubanLM-lab/LubanLM/scripts/rkllm_temporary_swap.py --cleanup
}
```

网络安装经临时本地转发器与WSL互操作管道，外部socket绑定实体WLAN；Python依赖来自清华与南大镜像，SDK来自Rockchip官方固定版本。不会把Clash回退出口用于依赖下载。模型与校准生成离线运行；转换拒绝覆盖现有产物，失败结果和日志保存在独立报告目录。

SDK打包有较高的主存峰值，本轮在WSL约15GiB内存外增加32GiB临时交换文件后完成两个导出，随后已关闭并删除交换文件。量化器会在源目录生成grq.qparams，本轮将这一派生缓存移入对应exports目录，原始24个模型文件逐一SHA256校验均未改变。导出成功仍须经过板端哈希、加载和问答验证。

## 板端测试与证据

`prepare_board_tokens.py`直接复用PC测试的原生输入Token，避免板端重复套模板；`prepare_board_bundle.py`打包C++代码、匹配头文件和问题信息。板端使用独立目录 `/home/cat/lubanlm-qwen-20261018`，保留此前50M实验。

C++程序读取模型路径、Token文件、结果文件、生成上限和上下文参数；先预热一题，再独立执行八题两轮，每题清空缓存。记录完整输出、返回码、流式首Token耗时、Runtime性能、RSS和VmHWM。`runtime_finish`只表示Runtime正常结束，不能未经Token核查就等同EOS。两轮小样本只能观察明显异常，不能证明长期内存稳定。

结果应将预热排除，区分进程RSS、峰值RSS和Runtime报告的内存；短问题结果不能外推长上下文。保持系统原有频率和驱动，不执行固定频率脚本或刷机。模型传输后按SHA256核对，再运行测试。

## Tailscale直连规则

根据用户后续授权，板端Clash通过 `configure_board_tailscale_direct.py`增加Linux进程、Tailscale IPv4/IPv6地址及相关域名的DIRECT规则。持久入口为clashctl的 `mixin.yaml` 中 `rules.prefix`，并非Clash Verge的 `prepend/append/delete` 格式。配置先备份、经Mihomo验证，再通过本地API热加载并检查生效规则。

本轮配置后Windows到板端测得44ms的Tailscale直连。DIRECT规则不保证所有网络都能打洞，也不能永久指定或排除某个DERP地区。SSH密码与代理订阅密钥不进入教程或报告。
