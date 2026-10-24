param([string]$DestinationDirectory, [string]$PythonExecutable)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.IO.Compression.FileSystem
$sourceRoot = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
if (-not $DestinationDirectory) {
    $DestinationDirectory = Join-Path (Split-Path -Parent $sourceRoot) 'archives\shares'
}
$destinationRoot = [IO.Path]::GetFullPath($DestinationDirectory)
if ($destinationRoot -eq $sourceRoot -or $destinationRoot.StartsWith($sourceRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Place share packages outside the tutorial source directory.'
}
New-Item -ItemType Directory -Path $destinationRoot -Force | Out-Null
if (-not $PythonExecutable) {
    foreach ($candidate in @(
        (Join-Path (Split-Path -Parent $sourceRoot) '.conda\lubanlm-py311\python.exe'),
        (Join-Path (Split-Path -Parent $sourceRoot) '.conda\reader-py311\python.exe')
    )) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) { $PythonExecutable = $candidate; break }
    }
    if (-not $PythonExecutable) {
        $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
        if ($pythonCommand) { $PythonExecutable = $pythonCommand.Source }
    }
}
if (-not $PythonExecutable) { throw 'Specify a Python 3.11 interpreter with -PythonExecutable.' }
$name = 'LubanLM-tutorial-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
$zipPath = Join-Path $destinationRoot ($name + '.zip')
if (Test-Path -LiteralPath $zipPath) { throw 'Share package already exists.' }
$manifestPath = Join-Path $destinationRoot ($name + '.manifest.json')
$checker = Join-Path $sourceRoot 'scripts\check_release.py'
& $PythonExecutable -B -X utf8 $checker --root $sourceRoot --output $manifestPath
if ($LASTEXITCODE -ne 0) { throw "Release checks failed. See $manifestPath" }
$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding utf8 | ConvertFrom-Json
if (-not $manifest.passed -or -not $manifest.files.Count) { throw 'No verified release files.' }
$oversized = @($manifest.files | Where-Object bytes -GT 100MB)
if ($oversized.Count) { throw "Review oversized tutorial files: $($oversized.path -join ', ')" }
$zip = [IO.Compression.ZipFile]::Open($zipPath, [IO.Compression.ZipArchiveMode]::Create)
try {
    foreach ($file in $manifest.files) {
        $filePath = Join-Path $sourceRoot $file.path
        $entryName = 'LubanLM/' + $file.path
        [IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip, $filePath, $entryName, [IO.Compression.CompressionLevel]::Optimal) | Out-Null
    }
} finally { $zip.Dispose() }
& $PythonExecutable -B -X utf8 $checker --root $sourceRoot --output $manifestPath --zip $zipPath
if ($LASTEXITCODE -ne 0) { throw 'ZIP verification failed; do not publish this archive.' }
$hash = (Get-FileHash -LiteralPath $zipPath -Algorithm SHA256).Hash.ToLowerInvariant()
[IO.File]::WriteAllText($zipPath + '.sha256', "$hash  $name.zip`n", [Text.UTF8Encoding]::new($false))
Write-Output "Packaged $($manifest.files.Count) verified files: $zipPath"
Write-Output "SHA256: $hash"
Write-Output "Manifest: $manifestPath"
