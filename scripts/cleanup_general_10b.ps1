$ErrorActionPreference = 'Stop'
$taskWorkspace = [IO.Path]::GetFullPath('D:\BaiduSyncdisk\知识库\LubanLM')
$taskRuns = [IO.Path]::GetFullPath('D:\LubanLM-Runs')
$taskData = [IO.Path]::GetFullPath('D:\LubanLM-Data')
$taskReport = Join-Path $taskWorkspace 'docs\run-results\general-10b-20261005'
New-Item -ItemType Directory -Path $taskReport -Force | Out-Null
if (@(Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python' -and $_.CommandLine -match 'training.train|scripts.run_general|scripts.prepare_general' }).Count -gt 0) {
    throw 'A training/data preparation process is active; refuse cleanup.'
}
$taskParent = Join-Path $taskRuns 'general-2b-v1\checkpoints\50m-general-base-2b\last.pt'
$taskParentHash = (Get-FileHash -LiteralPath $taskParent -Algorithm SHA256).Hash
$taskRefs = [Text.StringBuilder]::new()
foreach ($taskScan in @((Join-Path $taskWorkspace 'configs'), (Join-Path $taskWorkspace 'docs'), (Join-Path $taskWorkspace 'scripts'), (Join-Path $taskRuns 'chat-v2\reports'), (Join-Path $taskRuns 'general-2b-v1\reports'))) {
    foreach ($taskFile in Get-ChildItem -LiteralPath $taskScan -File -Recurse) {
        if ($taskFile.Extension -in @('.json','.jsonl','.md','.py','.ps1') -and $taskFile.Length -lt 5MB) {
            [void]$taskRefs.AppendLine([IO.File]::ReadAllText($taskFile.FullName).Replace('\','/').ToLowerInvariant())
        }
    }
}
$taskRefText = $taskRefs.ToString()
$taskCandidates = @()
$taskPreserved = @()
foreach ($taskFile in Get-ChildItem -LiteralPath $taskRuns -File -Recurse -Filter 'step-*.pt') {
    $taskPath = [IO.Path]::GetFullPath($taskFile.FullName)
    if (-not $taskPath.StartsWith($taskRuns + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Checkpoint escapes runs root.' }
    if (($taskFile.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) { throw 'Checkpoint reparse point.' }
    if ($taskRefText.Contains($taskPath.Replace('\','/').ToLowerInvariant()) -or $taskRefText.Contains($taskFile.Name.ToLowerInvariant())) {
        $taskPreserved += $taskPath
    } else { $taskCandidates += [ordered]@{path=$taskPath; bytes=$taskFile.Length; reason='unreferenced intermediate checkpoint'} }
}
foreach ($taskPath in @((Join-Path $taskData 'cache\pip'), (Join-Path $taskData 'general-2b-v1\before-optimization-20261005-020830'))) {
    $taskResolved = [IO.Path]::GetFullPath($taskPath)
    if (-not $taskResolved.StartsWith($taskData + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Cache escapes data root.' }
    if (Test-Path -LiteralPath $taskResolved) {
        $taskItems = @(Get-ChildItem -LiteralPath $taskResolved -Recurse -Force)
        if (@($taskItems | Where-Object { ($_.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0 -or $_.Extension -eq '.pt' }).Count) { throw 'Unexpected cache contents.' }
        $taskSize = ($taskItems | Where-Object {-not $_.PSIsContainer} | Measure-Object Length -Sum).Sum
        $taskCandidates += [ordered]@{path=$taskResolved; bytes=$taskSize; reason='download cache or superseded incomplete CPU preparation'}
    }
}
$taskBefore = (Get-PSDrive -Name D).Free
$taskReceipt = [ordered]@{timestamp=(Get-Date -Format o); before_free_bytes=$taskBefore; parent=$taskParent; parent_sha256=$taskParentHash; preserved_steps=$taskPreserved; deletions=$taskCandidates; status='validated_before_delete'}
$taskReceipt | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $taskReport 'cleanup.json') -Encoding utf8
foreach ($taskItem in $taskCandidates) {
    $taskFinal = [IO.Path]::GetFullPath($taskItem.path)
    if (-not ($taskFinal.StartsWith($taskRuns + '\', [StringComparison]::OrdinalIgnoreCase) -or $taskFinal.StartsWith($taskData + '\', [StringComparison]::OrdinalIgnoreCase))) { throw 'Delete target escapes authorized roots.' }
    Remove-Item -LiteralPath $taskFinal -Recurse -Force
}
if ((Get-FileHash -LiteralPath $taskParent -Algorithm SHA256).Hash -ne $taskParentHash) { throw 'Parent checkpoint changed.' }
$taskReceipt.after_free_bytes = (Get-PSDrive -Name D).Free
$taskReceipt.released_bytes = $taskReceipt.after_free_bytes - $taskBefore
$taskReceipt.status = 'completed'
$taskReceipt | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $taskReport 'cleanup.json') -Encoding utf8
[ordered]@{deleted_steps=($taskCandidates | Where-Object {$_.reason -eq 'unreferenced intermediate checkpoint'}).Count; preserved_steps=$taskPreserved.Count; released_GiB=[Math]::Round($taskReceipt.released_bytes/1GB,2); free_GiB=[Math]::Round($taskReceipt.after_free_bytes/1GB,2)} | ConvertTo-Json
