param(
    [ValidateSet('Base','Ai','Cpu')][string]$Mode = 'Ai',
    [switch]$CheckOnly
)
$ErrorActionPreference = 'Stop'
$studioRoot = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot)).TrimEnd('\','/')
$venvPath = [IO.Path]::GetFullPath((Join-Path $studioRoot '.venv'))
$backupPath = [IO.Path]::GetFullPath((Join-Path $studioRoot ('.venv.backup-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff'))))

# Resolve and check both move targets before any filesystem mutation.
$resolvedRoot = (Resolve-Path -LiteralPath $studioRoot).ProviderPath.TrimEnd('\','/')
if (!$resolvedRoot.Equals($studioRoot, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'The project root could not be resolved consistently.'
}
foreach ($candidate in @($venvPath, $backupPath)) {
    $candidateParent = [IO.Path]::GetDirectoryName($candidate)
    if (!$candidateParent.Equals($resolvedRoot, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Environment paths must be direct children of this project.'
    }
}
if (Test-Path -LiteralPath $venvPath) {
    $existingVenv = Get-Item -LiteralPath $venvPath -Force
    if (!$existingVenv.PSIsContainer -or ($existingVenv.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        throw 'The existing .venv must be a normal project directory, not a link or junction.'
    }
    $resolvedVenv = (Resolve-Path -LiteralPath $venvPath).ProviderPath
    if (!$resolvedVenv.Equals($venvPath, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'The existing environment resolves outside the expected location.'
    }
}
if (Test-Path -LiteralPath $backupPath) { throw 'The timestamped backup already exists. Please retry.' }

# Check Python before stopping the server or moving the working environment.
if (!(Get-Command py -ErrorAction SilentlyContinue)) {
    throw 'Install Python 3.10 with the Windows Python launcher first.'
}
& py -3.10 -c 'import sys; print(sys.executable); sys.exit(0 if sys.version_info[:2] == (3, 10) else 1)'
if ($LASTEXITCODE -ne 0) { throw 'Python 3.10 is unavailable. The current environment has not been changed.' }
Write-Host "Project: $studioRoot"
Write-Host "Rebuild mode: $Mode"
Write-Host "Previous environment backup: $backupPath"
if ($CheckOnly) {
    Write-Host 'Check completed. No server was stopped, and no environment was moved or installed.'
    return
}

# Run stop.ps1 in its own shell because that script may exit for an idle server.
$shellExecutable = (Get-Process -Id $PID).Path
& $shellExecutable -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'stop.ps1')
if ($LASTEXITCODE -ne 0) { throw 'Could not safely stop the recorded server. No environment was moved.' }
if (Test-Path -LiteralPath $venvPath) {
    Move-Item -LiteralPath $venvPath -Destination $backupPath
    Write-Host "Previous environment retained at: $backupPath"
}
try {
    & $shellExecutable -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'install.ps1') -Mode $Mode
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
} catch {
    Write-Host 'Models, LoRAs, weights, settings and results were not moved or removed.'
    Write-Host "The previous environment, if present, remains at: $backupPath"
    Write-Host 'Retry the normal environment installer to resume the new partial installation.'
    throw
}
Write-Host 'Environment rebuilt. Run the normal platform launcher.'
Write-Host 'After verifying the new environment, you may remove the old backup manually to reclaim disk space.'
