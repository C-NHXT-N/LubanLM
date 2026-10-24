$ErrorActionPreference = 'Stop'
$taskDataDir = [IO.Path]::GetFullPath('D:\LubanLM-Data\general-2b-v1')
$taskRunsDir = [IO.Path]::GetFullPath('D:\LubanLM-Runs\general-2b-v1')
$taskInfo = Get-Content -LiteralPath (Join-Path $taskRunsDir 'reports\controller.json') -Raw | ConvertFrom-Json
if ($taskInfo.status -ne 'preparing_cpu_data') {
    throw 'Controller is no longer in CPU preparation; do not interrupt training.'
}
$taskOwner = Get-CimInstance Win32_Process -Filter ('ProcessId=' + $taskInfo.pid)
if ($null -eq $taskOwner -or $taskOwner.CommandLine -notlike '*scripts.run_general_2b*50m-general-pretrain-2b.json*') {
    throw 'Controller ownership did not match.'
}
$taskChildren = @(Get-CimInstance Win32_Process -Filter ('ParentProcessId=' + $taskInfo.pid) | Where-Object { $_.Name -match '^python' })
foreach ($taskChild in $taskChildren) {
    if ($taskChild.CommandLine -notmatch 'scripts\.(prepare_general_2b|keep_awake)') {
        throw 'Unexpected Python child; inspect before stopping.'
    }
}
$taskStamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$taskHandoff = [ordered]@{
    reason = 'Apply measured batch 24 and four prefetched downloads. Preserve raw files, weights and both GPU ledgers.'
    previous_controller = $taskInfo
    previous_progress = (Get-Content -LiteralPath (Join-Path $taskRunsDir 'reports\preparation_progress.json') -Raw | ConvertFrom-Json)
    archive_suffix = $taskStamp
}
$taskHandoff | ConvertTo-Json -Depth 18 | Set-Content -LiteralPath (Join-Path $taskRunsDir 'reports\optimization_handoff.json') -Encoding utf8

# 只停止已确认本实验的Python进程；不操作conhost或其他应用。
Stop-Process -Id $taskInfo.pid -Force
foreach ($taskChild in $taskChildren) {
    Stop-Process -Id $taskChild.ProcessId -Force -ErrorAction SilentlyContinue
    $taskWaitProc = Get-Process -Id $taskChild.ProcessId -ErrorAction SilentlyContinue
    if ($null -ne $taskWaitProc -and -not $taskWaitProc.WaitForExit(10000)) {
        throw 'Child process did not stop.'
    }
}

function Move-OwnedExperimentItem($taskRoot, $taskLeaf, $taskArchive) {
    $taskSrc = [IO.Path]::GetFullPath((Join-Path $taskRoot $taskLeaf))
    $taskDst = [IO.Path]::GetFullPath((Join-Path $taskArchive $taskLeaf))
    $taskPrefix = $taskRoot + [IO.Path]::DirectorySeparatorChar
    if (-not $taskSrc.StartsWith($taskPrefix, [StringComparison]::OrdinalIgnoreCase) -or
        -not $taskDst.StartsWith($taskPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Move target escapes the named experiment directory.'
    }
    if (Test-Path -LiteralPath $taskSrc) {
        Move-Item -LiteralPath $taskSrc -Destination $taskDst
    }
}
$taskDataArchive = Join-Path $taskDataDir ('before-optimization-' + $taskStamp)
$taskRunsArchive = Join-Path $taskRunsDir ('before-optimization-' + $taskStamp)
New-Item -ItemType Directory -Path $taskDataArchive, $taskRunsArchive | Out-Null
foreach ($taskLeaf in @('processed', 'dedup.sqlite', 'dedup.sqlite-journal', 'dedup.sqlite-wal', 'dedup.sqlite-shm')) {
    Move-OwnedExperimentItem $taskDataDir $taskLeaf $taskDataArchive
}
foreach ($taskLeaf in @('controller.lock', 'controller.stdout.log', 'controller.stderr.log')) {
    Move-OwnedExperimentItem $taskRunsDir $taskLeaf $taskRunsArchive
}
$env:PYTHONIOENCODING = 'utf-8'
$taskProc = Start-Process -FilePath 'D:\miniconda3\envs\lubanlm-gpu\python.exe' `
    -ArgumentList @('-u', '-m', 'scripts.run_general_2b', '--plan', 'configs/50m-general-pretrain-2b.json') `
    -WorkingDirectory 'D:\BaiduSyncdisk\知识库\LubanLM' -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $taskRunsDir 'controller.stdout.log') `
    -RedirectStandardError (Join-Path $taskRunsDir 'controller.stderr.log') -PassThru
Write-Output ('Restarted optimized controller PID ' + $taskProc.Id)
Write-Output ('Recoverable partial data archive: ' + $taskDataArchive)
