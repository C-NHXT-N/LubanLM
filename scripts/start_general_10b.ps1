$ErrorActionPreference='Stop'
$taskWorkspace='D:\BaiduSyncdisk\知识库\LubanLM'
$taskRuns='D:\LubanLM-Runs\general-10b-v1'
$taskLock=Join-Path $taskRuns 'controller.lock'
if (Test-Path -LiteralPath $taskLock) {
    $taskOwnerId=[int](Get-Content -LiteralPath $taskLock -Raw)
    $taskOwner=Get-CimInstance Win32_Process -Filter ('ProcessId='+$taskOwnerId)
    if ($taskOwner) { throw 'An existing controller lock has a live owner; do not duplicate it.' }
    Remove-Item -LiteralPath $taskLock
}
$taskGPU=Join-Path $taskRuns 'reports\gpu_budget.lock'
if (Test-Path -LiteralPath $taskGPU) { throw 'GPU verification or training is still active.' }
$env:PYTHONIOENCODING='utf-8'
$taskProc=Start-Process -FilePath 'D:\miniconda3\envs\lubanlm-gpu\python.exe' -ArgumentList @('-B','-u','-m','scripts.run_general_10b','--plan','configs/50m-general-pretrain-10b.json') -WorkingDirectory $taskWorkspace -WindowStyle Hidden -RedirectStandardOutput (Join-Path $taskRuns 'controller.stdout.log') -RedirectStandardError (Join-Path $taskRuns 'controller.stderr.log') -PassThru
Write-Output ('Controller PID '+$taskProc.Id)
