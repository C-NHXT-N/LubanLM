[CmdletBinding()]
param(
    [ValidateSet('Both','Qwen3.5-2B','Qwen3-4B-Instruct-2507')]
    [string]$Model = 'Both',
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9.-]*$')]
    [string]$Board = 'cat',
    [ValidatePattern('^[a-z_][a-z0-9_-]*$')]
    [string]$User = 'cat',
    [switch]$VerifyOnly,
    [switch]$AllowRelay
)

# Does not change Clash, TUN, routes, Tailscale preferences, or start inference.
# Close Clash yourself; keep Tailscale running on both computers.
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Get-BoardConnection {
    param([string]$Executable, [string]$Target)
    # Capture stderr as text. Windows PowerShell 5.1 otherwise turns native
    # stderr into a terminating error before we can inspect the pong lines.
    if ($Target -notmatch '^[A-Za-z0-9][A-Za-z0-9.-]*$') { throw 'Invalid board hostname.' }
    $startInfo = New-Object System.Diagnostics.ProcessStartInfo
    $startInfo.FileName = $Executable
    $startInfo.Arguments = 'ping --c 3 --timeout 3s ' + $Target
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $startInfo
    try {
        [void]$process.Start()
        $stdout = $process.StandardOutput.ReadToEndAsync()
        $stderr = $process.StandardError.ReadToEndAsync()
        if (!$process.WaitForExit(20000)) {
            $process.Kill()
            $process.WaitForExit()
            throw 'Tailscale connectivity check timed out after 20 seconds; transfer was not started.'
        }
        $lines = @(($stdout.GetAwaiter().GetResult() + "`n" + $stderr.GetAwaiter().GetResult()) -split '\r?\n' | Where-Object { $_.Trim() })
        $pongs = @($lines | Where-Object { $_ -match '^pong from ' })
        $direct = @($pongs | Where-Object {
            $_ -match ' via ' -and $_ -notmatch ' via DERP\(' -and $_ -notmatch ' via peer-relay'
        }).Count -gt 0
        [pscustomobject]@{ Lines=$lines; Reachable=($pongs.Count -gt 0); Direct=$direct; ExitCode=$process.ExitCode }
    } finally {
        $process.Dispose()
    }
}

$workspace = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$reportRoot = Join-Path $workspace 'reports\rk3576-qwen-20261018'
$remoteRoot = '/home/cat/lubanlm-qwen-20261018'
$models = @('Qwen3.5-2B','Qwen3-4B-Instruct-2507')
if ($Model -ne 'Both') { $models = @($Model) }

try {
    $jobs = @()
    foreach ($name in $models) {
        $record = Join-Path $reportRoot ('conversion-' + $name + '.json')
        if (!(Test-Path -LiteralPath $record)) { throw "Conversion record missing: $record" }
        $info = Get-Content -LiteralPath $record -Raw -Encoding UTF8 | ConvertFrom-Json
        if (!$info.completed) { throw "Conversion is not complete: $name" }
        $fileName = $name + '_w4a16_rk3576_ctx2048.rkllm'
        $folder = Join-Path (Join-Path $workspace 'exports\rk3576') $name
        $file = Join-Path $folder $fileName
        if (!(Test-Path -LiteralPath $file)) { throw "Model file missing: $file" }
        if ((Get-Item -LiteralPath $file).Length -ne [long]$info.bytes) { throw "Size mismatch: $file" }
        Write-Host "Checking SHA256: $name"
        $hash = (Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($hash -ne $info.sha256) { throw "SHA256 mismatch: $file" }
        $jobs += [pscustomobject]@{ Name=$name; File=$file; FileName=$fileName; Hash=$hash; Bytes=[long]$info.bytes }
        Write-Host ('  OK: {0:N2} GB, {1}' -f ($info.bytes / 1e9), $hash)
    }
    if ($VerifyOnly) {
        Write-Host 'Local verification complete. No network connection or transfer was started.'
        return
    }

    $tailscaleCommand = Get-Command tailscale.exe -ErrorAction SilentlyContinue
    $tailscale = 'C:\Program Files\Tailscale\tailscale.exe'
    if ($tailscaleCommand) { $tailscale = $tailscaleCommand.Source }
    if (!(Test-Path -LiteralPath $tailscale)) { throw 'Tailscale CLI not found. Keep Tailscale installed and running.' }
    $ssh = (Get-Command ssh.exe -ErrorAction Stop).Source
    Write-Host 'Checking board connectivity. Clash must be closed by you; this script does not control it.'
    $connection = Get-BoardConnection -Executable $tailscale -Target $Board
    $connection.Lines | ForEach-Object { Write-Host $_ }
    Write-Host ('Tailscale ping exit code: {0}' -f $connection.ExitCode)
    if (!$connection.Reachable) { throw 'No reply from the board. Check Tailscale and board power/network.' }
    $direct = $connection.Direct
    if (!$direct -and !$AllowRelay) {
        throw 'Direct connection was not observed; transfer was not started. Check Clash/network, or explicitly use -AllowRelay to accept a potentially slow relay.'
    }
    if (!$direct) { Write-Warning 'Relay transfer explicitly allowed. Large files may take a long time.' }

    foreach ($job in $jobs) {
        Write-Host ('Sending {0} ({1:N2} GB) to {2}...' -f $job.Name, ($job.Bytes / 1e9), $Board)
        & $tailscale file cp --verbose --update-interval 1s $job.File ($Board + ':')
        if ($LASTEXITCODE -ne 0) { throw "Transfer failed: $($job.Name). Files were retained; rerun after fixing connectivity." }
    }

    # Only known public model filenames and hashes are inserted into this script.
    # Password authentication is handled interactively by OpenSSH; no password is saved.
    $remoteLines = @(
        'set -eu',
        ('mkdir -p ' + $remoteRoot + '/incoming'),
        ('cd ' + $remoteRoot),
        'sudo -n tailscale file get --conflict=skip incoming || receive_status=$?',
        'mkdir -p models'
    )
    foreach ($job in $jobs) {
        $target = 'models/' + $job.Name + '/' + $job.FileName
        $incoming = 'incoming/' + $job.FileName
        $remoteLines += ('mkdir -p models/' + $job.Name)
        $remoteLines += ('if test -f ' + $target + '; then')
        $remoteLines += ('  printf ''%s  %s\n'' ''' + $job.Hash + ''' ''' + $target + ''' | sha256sum -c -')
        $remoteLines += 'else'
        $remoteLines += ('  test -f ' + $incoming)
        $remoteLines += ('  printf ''%s  %s\n'' ''' + $job.Hash + ''' ''' + $incoming + ''' | sha256sum -c -')
        $remoteLines += ('  mv -n -- ' + $incoming + ' ' + $target)
        $remoteLines += 'fi'
        $remoteLines += ('printf ''%s  %s\n'' ''' + $job.Hash + ''' ''' + $target + ''' | sha256sum -c -')
    }
    $remoteLines += 'echo TRANSFER_AND_BOARD_SHA256_VERIFIED'
    $remoteText = ($remoteLines -join "`n") + "`n"
    $encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($remoteText))
    $remoteCommand = "printf %s '$encoded' | base64 -d | bash"
    Write-Host 'Files sent. Enter the board SSH password when OpenSSH asks; input is not displayed.'
    & $ssh -o StrictHostKeyChecking=yes -o ConnectTimeout=15 ($User + '@' + $Board) $remoteCommand
    if ($LASTEXITCODE -ne 0) {
        throw 'Board receipt or SHA256 check failed. Files were retained. Check the board output; inference was not started.'
    }
    $result = [ordered]@{
        completed = $true
        verified_on_board = $true
        board = $Board
        remote_root = $remoteRoot
        finished_at = (Get-Date).ToString('o')
        models = $jobs
        inference_started = $false
        clash_modified = $false
    }
    $resultPath = Join-Path $reportRoot ('manual-transfer-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '.json')
    $result | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $resultPath -Encoding UTF8
    Write-Host "Transfer and board SHA256 verification complete. Report: $resultPath"
    Write-Host 'No board inference was started. You may restore Clash yourself when ready.'
} catch {
    Write-Error -ErrorAction Continue $_
    Write-Host 'Stopped. Existing models and partial transfers were retained; Clash was not changed.'
    exit 1
}
