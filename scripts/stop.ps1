$ErrorActionPreference = 'Stop'
$studioRoot = Split-Path -Parent $PSScriptRoot
$recordPath = Join-Path $studioRoot 'logs/server.json'
if (!(Test-Path -LiteralPath $recordPath)) { Write-Host 'No recorded server.'; exit 0 }
$record = Get-Content -LiteralPath $recordPath -Raw | ConvertFrom-Json
$process = Get-CimInstance Win32_Process -Filter "ProcessId = $($record.pid)" -ErrorAction SilentlyContinue
if (!$process) { Write-Host 'Server is already stopped.'; exit 0 }
$expected = (Join-Path $studioRoot '.venv/Scripts/python.exe').Replace('/','\')
if ($process.ExecutablePath -ne $expected -or $process.CommandLine -notmatch 'uvicorn app.main:app') {
    throw 'The recorded PID now belongs to another process. It was not stopped.'
}
# A Windows venv launcher has a base-interpreter child. Stop that child first,
# otherwise killing only the launcher can leave the HTTP server alive.
$children = Get-CimInstance Win32_Process -Filter "ParentProcessId = $($record.pid)" | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match 'uvicorn app.main:app' }
foreach ($child in $children) { Stop-Process -Id $child.ProcessId -ErrorAction SilentlyContinue }
Stop-Process -Id $record.pid -ErrorAction SilentlyContinue
Write-Host 'Landslide Studio stopped. Running tasks will be marked interrupted on next launch.'
