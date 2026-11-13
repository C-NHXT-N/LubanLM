# 手动传输两个Qwen模型

当前这两个模型已手动通过Tailscale DERP中继传输，板端SHA256校验通过，并完成短问答测试；详细结果见[板端回答与性能报告](cases/原模型板端问答.md)。本页保留脚本启动与故障恢复说明，供需要重新传输时参考。WSL临时交换空间已关闭并删除；Windows Clash配置没有改动，先前按授权添加的板端DIRECT规则保留。

## 启动

1. 你自行关闭本机Clash，确保两端Tailscale仍在运行，板端保持联网。
2. 打开PowerShell，运行下面的命令。无需激活Conda、无需管理员权限，命令与当前工作目录无关。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Lab\LubanLM-lab\LubanLM\scripts\transfer_qwen_to_board.ps1"
```

脚本先校验本地模型，再检查Tailscale直连，按Qwen3.5-2B、Qwen3-4B顺序传输。两文件合计约5.30GB，实际时间由上传速度决定。没有观察到直连时默认停止，不会自动开始慢速中继传输；如果你愿意接受中继，可手动附加 `-AllowRelay`。

连接检查会显示Tailscale的回应、诊断文字和退出码。`direct connection not established` 表示本次检查未建立直连；若同时显示 `pong from ... via DERP(...)`，说明板端可达但经中继。脚本已修正Windows PowerShell 5.1将这段诊断误当异常、提前退出的问题；仍会在未观察到直连时停止传输。关闭Clash并不能保证一定建立直连，两端网络条件也会影响连接。

发送结束后，OpenSSH会提示输入 `cat@cat` 的密码，输入时不回显。脚本不会保存密码。板端通过现有 `sudo -n tailscale` 接收并核对SHA256，将模型放在本轮独立实验目录中，不启动模型或修改Clash。SSH使用已有主机密钥记录，密钥不匹配时停止。

成功标志为 `TRANSFER_AND_BOARD_SHA256_VERIFIED`，随后会在工作区 `reports/rk3576-qwen-20261018/` 生成 `manual-transfer-时间.json`。最终文件分别位于：

```text
/home/cat/lubanlm-qwen-20261018/models/Qwen3.5-2B/Qwen3.5-2B_w4a16_rk3576_ctx2048.rkllm
/home/cat/lubanlm-qwen-20261018/models/Qwen3-4B-Instruct-2507/Qwen3-4B-Instruct-2507_w4a16_rk3576_ctx2048.rkllm
```

## 先验证或只传一个

只做本地大小与SHA256校验，不联网、不传输：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Lab\LubanLM-lab\LubanLM\scripts\transfer_qwen_to_board.ps1" -VerifyOnly
```

只传Qwen3.5-2B，在启动命令末尾添加：

```powershell
-Model Qwen3.5-2B
```

只传4B则用 `-Model Qwen3-4B-Instruct-2507`。脚本失败时保留本地文件、板端模型与部分传输；可以排除网络问题后重新运行。部分文件的续传由所安装的Tailscale版本处理，不承诺所有中断位置均可续接，完成后始终核对SHA256。板端已有同名目标模型时先验哈希，不覆盖不同内容。

板端Taildrop收件箱统一接收到本轮 `incoming/` 目录，再只处理选中的模型文件；其他待收文件不会被移入模型目录。已有接收文件冲突或缺失会由哈希/文件检查阻止继续。传输完成后可以自行恢复Clash，再通知继续板端测试。
