param([Parameter(Mandatory=$true)][string]$Config, [switch]$KeepAlive)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$workspace = Split-Path $projectRoot -Parent
$pythonPath = Join-Path $workspace '.conda\qwen-lora-py311\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) { throw "Training Python environment not found: $pythonPath" }
$configPath = if ([IO.Path]::IsPathRooted($Config)) { $Config } else { Join-Path $projectRoot $Config }
$configPath = (Resolve-Path -LiteralPath $configPath).Path
$settings = Get-Content -LiteralPath $configPath -Raw -Encoding UTF8 | ConvertFrom-Json
$reportPath = Join-Path $workspace $settings.report
New-Item -ItemType Directory -Force -Path $reportPath | Out-Null
if (Test-Path -LiteralPath (Join-Path $reportPath 'launch.json')) { throw 'Already launched; inspect status.json. Do not duplicate GPU jobs.' }
$preflightArguments = @('-X','utf8','-m','scripts.linux_assistant.retrain','--config',$configPath,'--preflight-only')
Push-Location -LiteralPath $projectRoot
try {
    $preflightOutput = & $pythonPath @preflightArguments 2>&1
} finally {
    Pop-Location
}
if ($LASTEXITCODE -ne 0) {
    $preflightOutput | Write-Output
    throw "Training preflight failed with exit code $LASTEXITCODE; no background worker was started."
}
$preflightOutput | Write-Output
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
$stdout = Join-Path $reportPath "worker-$stamp.out.log"
$stderr = Join-Path $reportPath "worker-$stamp.err.log"
$arguments = @('-X','utf8','-u','-m','scripts.linux_assistant.retrain','--config',('"' + $configPath + '"'))
$worker = Start-Process -FilePath $pythonPath -ArgumentList $arguments -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
@{ pid=$worker.Id; started_utc=[DateTime]::UtcNow.ToString('o'); config=$configPath; stdout=$stdout; stderr=$stderr } |
    ConvertTo-Json | Set-Content -LiteralPath (Join-Path $reportPath 'launch.json') -Encoding utf8
Write-Output "Started background reviewed-data pipeline. PID: $($worker.Id). Report: $reportPath"
if ($KeepAlive) {
    # Retain the tool-host parent so its job container does not kill detached workers.
    $worker.WaitForExit()
    exit $worker.ExitCode
}
