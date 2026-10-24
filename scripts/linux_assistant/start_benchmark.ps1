param([string]$Config = 'configs/linux-assistant-qwen35-2b-pipeline-check-20261023.json', [switch]$KeepAlive)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$workspace = Split-Path $projectRoot -Parent
$pythonPath = Join-Path $workspace '.conda\qwen-lora-py311\python.exe'
$configPath = if ([IO.Path]::IsPathRooted($Config)) { $Config } else { Join-Path $projectRoot $Config }
$configPath = (Resolve-Path -LiteralPath $configPath).Path
$settings = Get-Content -LiteralPath $configPath -Raw -Encoding UTF8 | ConvertFrom-Json
$reportPath = [IO.Path]::GetFullPath((Join-Path $workspace $settings.report))
if (-not $reportPath.StartsWith($workspace + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Report path must stay inside workspace.' }
New-Item -ItemType Directory -Force -Path $reportPath | Out-Null
if (Test-Path -LiteralPath (Join-Path $reportPath 'launch.json')) { throw 'Already launched; inspect the recorded result.' }
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
$stdout = Join-Path $reportPath "worker-$stamp.out.log"
$stderr = Join-Path $reportPath "worker-$stamp.err.log"
$arguments = @('-X','utf8','-u','-m','scripts.linux_assistant.benchmark_training_pipeline','--config',('"' + $configPath + '"'))
$worker = Start-Process -FilePath $pythonPath -ArgumentList $arguments -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
@{pid=$worker.Id; started_utc=[DateTime]::UtcNow.ToString('o'); config=$configPath;
  stdout=$stdout; stderr=$stderr; mode='synthetic_performance_validation'; model_saved=$false} |
    ConvertTo-Json | Set-Content -LiteralPath (Join-Path $reportPath 'launch.json') -Encoding utf8
Write-Output "Started unattended performance validation. PID: $($worker.Id). Result: $reportPath"
if ($KeepAlive) {
    $worker.WaitForExit()
    exit $worker.ExitCode
}
