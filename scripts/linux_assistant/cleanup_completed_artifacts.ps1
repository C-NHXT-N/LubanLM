param([switch]$Apply)
$ErrorActionPreference = 'Stop'
$workspace = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..\..')).Path
$report = Join-Path $workspace 'reports\linux-assistant-cleanup-20261023'
$planPath = Join-Path $report 'cleanup-plan.json'
$completionPath = Join-Path $report 'cleanup.json'
New-Item -ItemType Directory -Force -Path $report | Out-Null

function Get-CheckedPath([string]$Relative) {
    $absolute = [IO.Path]::GetFullPath((Join-Path $workspace $Relative))
    if (-not $absolute.StartsWith($workspace + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw "Path escapes the workspace: $absolute"
    }
    $cursor = $absolute
    while ($cursor.Length -ge $workspace.Length) {
        if (Test-Path -LiteralPath $cursor) {
            $item = Get-Item -LiteralPath $cursor -Force
            if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Reparse point: $cursor" }
            if ((Resolve-Path -LiteralPath $cursor).Path -ne $cursor) { throw "Unexpected resolved path: $cursor" }
        }
        if ($cursor -eq $workspace) { break }
        $cursor = Split-Path $cursor -Parent
    }
    return $absolute
}

$oldRuns = @(
    'runs\linux-assistant-qwen35-2b-reviewed-20261020',
    'runs\linux-assistant-qwen35-2b-exploration-20261022'
)
$cacheRoots = @('.cache\pip', '.cache\pip-linux', '.cache\bootstrap')
function Test-DeletionAllowed([string]$Relative) {
    foreach ($cache in $cacheRoots) {
        if ($Relative.StartsWith($cache + '\', [StringComparison]::OrdinalIgnoreCase)) { return $true }
    }
    foreach ($run in $oldRuns) {
        $pattern = '^' + [regex]::Escape($run) + '\\(?:lr-1e-4|smoke)\\checkpoint-\d+\\(?:adapter_model\.safetensors|training-state\.pt|tokenizer\.json)$'
        if ($Relative -match $pattern) { return $true }
    }
    return $false
}

if (Test-Path -LiteralPath $completionPath) {
    Get-Content -Raw -LiteralPath $completionPath | ConvertFrom-Json |
        Select-Object completed,deleted_files,operational_bytes_removed,retained_files_verified
    exit 0
}

$lockPath = Join-Path $workspace 'runs\linux-assistant-gpu.lock'
$lockStream = [IO.File]::Open($lockPath, [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::ReadWrite)
$locked = $false
try {
    if ($lockStream.Length -eq 0) { $lockStream.WriteByte(48); $lockStream.Flush() }
    $lockStream.Lock(0, 1)
    $locked = $true
    if (-not (Test-Path -LiteralPath $planPath)) {
        $scoresPath = Join-Path $workspace 'reports\linux-assistant-checkpoint-calibration-20261022\calibrated-score-summary.json'
        $scores = Get-Content -LiteralPath $scoresPath -Raw | ConvertFrom-Json
        if ($scores.status -ne 'rowwise_scored' -or $scores.ledger_rows -ne 360) {
            throw 'Checkpoint diagnostic scoring must finish before deleting historical adapters.'
        }
        $ledger = Join-Path $workspace 'reports\linux-assistant-checkpoint-calibration-20261022\calibrated-score-ledger.jsonl'
        if ((Get-FileHash -LiteralPath $ledger -Algorithm SHA256).Hash.ToLowerInvariant() -ne $scores.ledger_sha256) {
            throw 'Checkpoint diagnostic score ledger changed.'
        }
        $files = @()
        $retained = @()
        $baseTokenizerHash = (Get-FileHash -LiteralPath (Join-Path $workspace 'models\Qwen\Qwen3.5-2B\tokenizer.json') -Algorithm SHA256).Hash.ToLowerInvariant()
        foreach ($relativeRoot in @($oldRuns) + @($cacheRoots)) {
            $root = Get-CheckedPath $relativeRoot
            if (-not (Test-Path -LiteralPath $root)) { continue }
            foreach ($file in Get-ChildItem -LiteralPath $root -Recurse -Force -File) {
                $relative = $file.FullName.Substring($workspace.Length + 1)
                $checked = Get-CheckedPath $relative
                $digest = (Get-FileHash -LiteralPath $checked -Algorithm SHA256).Hash.ToLowerInvariant()
                $row = @{path=$relative; bytes=$file.Length; sha256=$digest}
                if (Test-DeletionAllowed $relative) {
                    if ($file.Name -eq 'tokenizer.json' -and $relative.StartsWith('runs\') -and $digest -ne $baseTokenizerHash) {
                        # Transformers may serialize a modified tokenizer. Preserve one
                        # exact copy of each unique digest as diagnostic evidence.
                        $tokenizerRecord = Join-Path $report ('retained-tokenizers\' + $digest + '.json')
                        if (-not (Test-Path -LiteralPath $tokenizerRecord)) {
                            New-Item -ItemType Directory -Force -Path (Split-Path $tokenizerRecord -Parent) | Out-Null
                            Copy-Item -LiteralPath $checked -Destination $tokenizerRecord
                            if ((Get-FileHash -LiteralPath $tokenizerRecord -Algorithm SHA256).Hash.ToLowerInvariant() -ne $digest) {
                                throw "Tokenizer evidence copy mismatch: $relative"
                            }
                            $retained += @{path=$tokenizerRecord.Substring($workspace.Length+1); bytes=$file.Length; sha256=$digest}
                        }
                        $row.retained_evidence = $tokenizerRecord.Substring($workspace.Length+1)
                    }
                    $row.reason = if ($relative.StartsWith('.cache\')) {'Rebuildable package/download cache'} else {'Completed historical experiment; comparison and raw answers retained'}
                    $files += $row
                } else { $retained += $row }
            }
        }
        # Pin the diagnostic output too; only binary checkpoint/cache files are deleted.
        foreach ($relativeRoot in @('reports\linux-assistant-checkpoint-calibration-20261022','runs\linux-assistant-checkpoint-calibration-20261022')) {
            foreach ($file in Get-ChildItem -LiteralPath (Get-CheckedPath $relativeRoot) -Recurse -File -Force) {
                $retained += @{path=$file.FullName.Substring($workspace.Length+1); bytes=$file.Length; sha256=(Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()}
            }
        }
        $plan = @{created_utc=[DateTime]::UtcNow.ToString('o'); authorized_by='User requested removal of unused prior artifacts, preserving results';
            workspace=$workspace; files=$files; retained_files=$retained;
            bytes_to_remove=($files | Measure-Object -Property bytes -Sum).Sum;
            protected='All models/, exports/, datasets/, provenance/, archives/, environments and 50M runs are outside this exact file allowlist.'}
        $plan | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $planPath -Encoding utf8
    }
    $plan = Get-Content -Raw -LiteralPath $planPath | ConvertFrom-Json
    if ($plan.workspace -ne $workspace) { throw 'Cleanup workspace identity mismatch.' }
    # Revalidate every file before the first deletion. No computed recursive delete is used.
    foreach ($row in $plan.files) {
        if (-not (Test-DeletionAllowed $row.path)) { throw "Target outside exact allowlist: $($row.path)" }
        $target = Get-CheckedPath $row.path
        if (-not (Test-Path -LiteralPath $target -PathType Leaf)) { throw "Planned target is missing: $target" }
        if ((Get-Item -LiteralPath $target).Length -ne $row.bytes -or
            (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant() -ne $row.sha256) {
            throw "Planned target changed: $target"
        }
    }
    if (-not $Apply) {
        Write-Output "Plan saved: $planPath; files=$($plan.files.Count); GiB=$([math]::Round($plan.bytes_to_remove/1GB,3))"
        exit 0
    }
    foreach ($row in $plan.files) {
        $target = Get-CheckedPath $row.path
        Remove-Item -LiteralPath $target -Force
        if (Test-Path -LiteralPath $target) { throw "Deletion failed: $target" }
    }
    foreach ($row in $plan.retained_files) {
        $target = Get-CheckedPath $row.path
        if ((Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant() -ne $row.sha256) {
            throw "Retained result record changed: $target"
        }
    }
    @{completed=$true; completed_utc=[DateTime]::UtcNow.ToString('o'); deleted_files=$plan.files.Count;
      operational_bytes_removed=$plan.bytes_to_remove; deleted_targets=@($plan.files | ForEach-Object {Get-CheckedPath $_.path});
      retained_files_verified=$plan.retained_files.Count; plan_sha256=(Get-FileHash -LiteralPath $planPath -Algorithm SHA256).Hash.ToLowerInvariant()} |
        ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $completionPath -Encoding utf8
    Write-Output "Cleanup completed: $completionPath; reclaimed GiB=$([math]::Round($plan.bytes_to_remove/1GB,3))"
} finally {
    if ($locked) { $lockStream.Unlock(0, 1) }
    $lockStream.Dispose()
}
