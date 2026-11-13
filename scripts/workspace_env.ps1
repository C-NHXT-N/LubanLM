# Dot-source this file before importing the Python training modules.
$workspaceRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not $env:LUBAN_DATA) {
    $env:LUBAN_DATA = Join-Path $workspaceRoot 'datasets\general-10b-teaching-v2'
}
if (-not $env:LUBAN_RUNS) {
    $env:LUBAN_RUNS = Join-Path $workspaceRoot 'runs'
}
if (-not $env:CUDA_VISIBLE_DEVICES) { $env:CUDA_VISIBLE_DEVICES = '0' }
$tutorialRoot = Split-Path -Parent $PSScriptRoot
$pythonPaths = @($env:PYTHONPATH -split [IO.Path]::PathSeparator | Where-Object { $_ })
if ($pythonPaths -notcontains $tutorialRoot) {
    $env:PYTHONPATH = (@($tutorialRoot) + $pythonPaths) -join [IO.Path]::PathSeparator
}
if (-not $env:HF_HOME) { $env:HF_HOME = Join-Path $workspaceRoot '.cache\huggingface' }
if (-not $env:HF_DATASETS_CACHE) { $env:HF_DATASETS_CACHE = Join-Path $workspaceRoot '.cache\datasets' }
if (-not $env:HF_ENDPOINT) { $env:HF_ENDPOINT = 'https://hf-mirror.com' }
if (-not $env:LUBAN_DOWNLOAD_INTERFACE) { $env:LUBAN_DOWNLOAD_INTERFACE = 'WLAN' }
# HF_ENDPOINT alone does not bypass TUN; downloads must use PhysicalMirror.
Write-Output "Tutorial: $tutorialRoot"
Write-Output "Data: $env:LUBAN_DATA"
Write-Output "Runs: $env:LUBAN_RUNS"
