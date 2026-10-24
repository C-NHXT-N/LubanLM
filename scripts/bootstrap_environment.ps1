param([string]$Interface = 'WLAN')
$ErrorActionPreference = 'Stop'
$bootstrapPython = 'C:\Tools\miniconda3\python.exe'
if (-not (Test-Path -LiteralPath $bootstrapPython)) { throw 'Existing Miniconda interpreter not found.' }
$tutorialRoot = Split-Path -Parent $PSScriptRoot
$savedPythonPath = $env:PYTHONPATH
try {
    $env:PYTHONPATH = $tutorialRoot
    & $bootstrapPython -B -X utf8 -m scripts.bootstrap_environment --interface $Interface
    if ($LASTEXITCODE -ne 0) { throw "Environment setup failed with exit code $LASTEXITCODE" }
} finally { $env:PYTHONPATH = $savedPythonPath }
