$ErrorActionPreference = 'Stop'
$workspace = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..\..')).Path
$archive = Join-Path $workspace 'reports\linux-assistant-cleanup-20261020'
$relativeTargets = @(
    'runs\linux-assistant-unified-qwen35-2b-20261020',
    'runs\linux-assistant-unified-qwen35-2b-mb2-20261020',
    'runs\linux-assistant-unified-qwen35-2b-mb8-20261020',
    'runs\linux-assistant-v1-20261019',
    'exports\linux-assistant-v1-20261019'
)
if (Test-Path -LiteralPath (Join-Path $archive 'cleanup.json')) {
    throw 'Cleanup already recorded; inspect its record instead of repeating deletion.'
}
if (Get-Process -Name python,pythonw -ErrorAction SilentlyContinue) {
    throw 'A Python process exists. Confirm experiment ownership before cleanup.'
}
New-Item -ItemType Directory -Path $archive -Force | Out-Null
$inventory = @()
$resolvedTargets = @()
foreach ($relative in $relativeTargets) {
    $target = [IO.Path]::GetFullPath((Join-Path $workspace $relative))
    if (-not $target.StartsWith($workspace + '\', [StringComparison]::OrdinalIgnoreCase)) { throw "Target escapes workspace: $target" }
    if ($target -ne (Join-Path $workspace $relative)) { throw "Unexpected normalized target: $target" }
    if (-not (Test-Path -LiteralPath $target)) { continue }
    $resolved = (Resolve-Path -LiteralPath $target).Path
    if ($resolved -ne $target) { throw "Unexpected resolved target: $resolved" }
    $entries = @(Get-Item -LiteralPath $target) + @(Get-ChildItem -LiteralPath $target -Recurse -Force)
    if ($entries | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint }) { throw "Reparse point in deletion target: $target" }
    $resolvedTargets += $target
    foreach ($file in ($entries | Where-Object { -not $_.PSIsContainer })) {
        $relFile = $file.FullName.Substring($workspace.Length + 1)
        $checkpoint = $file.FullName -match '\\checkpoint-\d+\\'
        $record = ($file.Extension -in @('.json','.jsonl','.log','.md','.txt')) -and (
            (-not $checkpoint) -or $file.Name -in @('run-state.json','adapter_config.json'))
        $digest = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($record) {
            $copy = Join-Path (Join-Path $archive 'result-records') $relFile
            New-Item -ItemType Directory -Path (Split-Path $copy -Parent) -Force | Out-Null
            Copy-Item -LiteralPath $file.FullName -Destination $copy
            if ((Get-FileHash -LiteralPath $copy -Algorithm SHA256).Hash.ToLowerInvariant() -ne $digest) { throw "Record copy mismatch: $relFile" }
        }
        $inventory += @{ path=$relFile; bytes=$file.Length; sha256=$digest; result_record_preserved=$record }
    }
}
$plan = @{ authorized_by='User requested deletion of prior wrong/failed experiments while retaining result records';
    created_utc=[DateTime]::UtcNow.ToString('o'); exact_targets=$resolvedTargets; files=$inventory;
    bytes_to_remove=($inventory | Measure-Object -Property bytes -Sum).Sum;
    invalid_evaluation_policy='Preserved raw result records are historical evidence only. Leaked scores are invalid; weights were not corrupted by the evaluation bug.';
    protected='Original Qwen models, RK3576 original quantized files, 50M pretrain/SFT, datasets and all existing reports are outside deletion targets.' }
$plan | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $archive 'cleanup-plan.json') -Encoding utf8
@'
# 历史实验清理记录

旧运行目录和失败/撤回候选检查点已按用户要求清理。保留配置、训练日志、错误原因、原始输出与摘要作为结果记录，存放在result-records/。

其中统一数据旧实验的6份输出共402次生成受到目标答案泄漏影响，质量分数无效。保存这些原始记录仅供排错，不代表有效评测结果。评测错误未损坏训练权重，但本次按用户要求删除旧候选，不再续训或选型。其他旧实验未达门槛或已中止，状态以各自历史报告为准。

原始Qwen、已有原模型量化文件与50M教学模型保留。cleanup-plan.json列出删除对象、字节数、SHA256及是否保存结果记录。
'@ | Set-Content -LiteralPath (Join-Path $archive 'README.md') -Encoding utf8
foreach ($target in $resolvedTargets) {
    # Each absolute target was checked against an explicit allowlist above, before recursive deletion.
    Remove-Item -LiteralPath $target -Recurse -Force
    if (Test-Path -LiteralPath $target) { throw "Deletion incomplete: $target" }
}
@{ completed=$true; completed_utc=[DateTime]::UtcNow.ToString('o'); deleted_targets=$resolvedTargets;
   deleted_files=$inventory.Count; result_records_preserved=@($inventory | Where-Object result_record_preserved).Count;
   operational_bytes_removed=$plan.bytes_to_remove; files=$inventory } |
    ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $archive 'cleanup.json') -Encoding utf8
Write-Output "Cleanup completed; record: $(Join-Path $archive 'cleanup.json')"
