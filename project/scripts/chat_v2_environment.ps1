# 从项目根目录 dot-source；v1 的语料、账本和权重保持不变。
. "$PSScriptRoot/environment.ps1"
$env:LUBAN_DATA = 'D:\LubanLM-Data\chat-v2'
$env:LUBAN_RUNS = 'D:\LubanLM-Runs\chat-v2'
$env:LUBAN_GPU_BUDGET_MINUTES = '1440'
