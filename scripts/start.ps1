param([switch]$NoBrowser, [int]$Port = 8765)
$ErrorActionPreference = 'Stop'
$studioRoot = Split-Path -Parent $PSScriptRoot
$studioPython = Join-Path $studioRoot '.venv/Scripts/python.exe'
if (!(Test-Path -LiteralPath $studioPython)) { throw 'Run the 01_*.cmd base environment installer first.' }
New-Item -ItemType Directory -Force -Path (Join-Path $studioRoot 'logs') | Out-Null
$studioUrl = "http://127.0.0.1:$Port"
# A fresh entry URL prevents an old cached index from hiding updated controls.
$studioPageUrl = "$studioUrl/?launch=$([DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds())"
$existing = $null
try { $existing = Invoke-RestMethod "$studioUrl/api/system" -TimeoutSec 3 } catch {}
if ($existing) {
    if ($existing.root -ne $studioRoot) { throw "Port $Port belongs to another application. Use -Port with a different port." }
    Write-Host "Landslide Studio is already running at $studioUrl"
    if (!$NoBrowser) { Start-Process $studioPageUrl }
    exit 0
}
$server = Start-Process -FilePath $studioPython -WorkingDirectory $studioRoot -ArgumentList @('-m','uvicorn','app.main:app','--host','127.0.0.1','--port',"$Port") -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $studioRoot 'logs/server.log') -RedirectStandardError (Join-Path $studioRoot 'logs/server-error.log')
@{pid=$server.Id; port=$Port; root=$studioRoot} | ConvertTo-Json | Set-Content (Join-Path $studioRoot 'logs/server.json') -Encoding UTF8
$ready = $false
for ($attempt=0; $attempt -lt 30; $attempt++) {
    Start-Sleep -Milliseconds 500
    if ($server.HasExited) { break }
    try {
        $health = Invoke-RestMethod "$studioUrl/api/system" -TimeoutSec 2
        if ($health.root -eq $studioRoot) { $ready=$true; break }
    } catch {}
}
if (!$ready) { throw "Startup failed. See $studioRoot/logs/server-error.log" }
Write-Host "Landslide Studio: $studioUrl"
if (!$NoBrowser) { Start-Process $studioPageUrl }
