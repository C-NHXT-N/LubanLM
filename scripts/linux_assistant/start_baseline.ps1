param([string]$Config = 'configs/linux-assistant-qwen35-baseline-quality.json', [switch]$KeepAlive)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$workspace = Split-Path $projectRoot -Parent
$pythonPath = Join-Path $workspace '.conda\qwen-lora-py311\python.exe'
$configPath = if ([IO.Path]::IsPathRooted($Config)) { $Config } else { Join-Path $projectRoot $Config }
$configPath = (Resolve-Path -LiteralPath $configPath).Path
$settings = Get-Content -LiteralPath $configPath -Raw -Encoding UTF8 | ConvertFrom-Json
$baselineReport = if ($settings.baseline_report) { $settings.baseline_report } else { $settings.report }
$reportPath = Join-Path $workspace $baselineReport
New-Item -ItemType Directory -Force -Path $reportPath | Out-Null
$statusPath = Join-Path $reportPath 'status.json'
if (Test-Path -LiteralPath $statusPath) {
    $state = Get-Content -LiteralPath $statusPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($state.state -eq 'completed') {
        Write-Output "Already completed. Report: $(Join-Path $reportPath 'REPORT.md')"
        exit 0
    }
    if ($state.state -eq 'running' -and (Get-Process -Id $state.pid -ErrorAction SilentlyContinue)) {
        Write-Output "Already running. PID: $($state.pid). Status: $statusPath"
        exit 0
    }
}
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
$stdout = Join-Path $reportPath "worker-$stamp.out.log"
$stderr = Join-Path $reportPath "worker-$stamp.err.log"
$arguments = @('-u', '-m', 'scripts.linux_assistant.baseline', '--config', ('"' + $configPath + '"'))
$worker = Start-Process -FilePath $pythonPath -ArgumentList $arguments -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
@{ pid=$worker.Id; started_utc=[DateTime]::UtcNow.ToString('o'); config=$configPath; stdout=$stdout; stderr=$stderr; status=$statusPath } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $reportPath 'launch.json') -Encoding utf8
Write-Output "Started background baseline. PID: $($worker.Id). Status: $statusPath"
if ($KeepAlive) {
    # Tool-host job containers may terminate detached children when their parent exits.
    $worker.WaitForExit()
    exit $worker.ExitCode
}
