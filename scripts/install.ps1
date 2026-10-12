param([ValidateSet('Base','Ai','Cpu')][string]$Mode='Base')
$ErrorActionPreference='Stop'
$studioRoot=Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $studioRoot
New-Item -ItemType Directory -Force -Path '.cache/pip','.cache/tmp' | Out-Null
$env:TEMP=Join-Path $studioRoot '.cache/tmp'
$env:TMP=$env:TEMP
$env:PIP_CACHE_DIR=Join-Path $studioRoot '.cache/pip'
$studioPython=Join-Path $studioRoot '.venv/Scripts/python.exe'
if (!(Test-Path -LiteralPath $studioPython)) {
    & py -3.10 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Install Python 3.10 first, then retry.' }
}
function Pip-Install([string[]]$Arguments) {
    & $studioPython -m pip install --timeout 120 --retries 3 @Arguments
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed. Already installed packages are retained. Retry when the network is available.' }
}
& $studioPython -c 'import pip; exit(0 if int(pip.__version__.partition(chr(46))[0]) >= 25 else 1)'
if ($LASTEXITCODE -ne 0) { Pip-Install @('--upgrade','pip') }
Pip-Install @('-r','requirements.txt')
if ($Mode -eq 'Ai') {
    Pip-Install @('--resume-retries','8','torch==2.7.1+cu126','torchvision==0.22.1+cu126','--index-url','https://download.pytorch.org/whl/cu126')
    Pip-Install @('-r','requirements-ai.txt')
} elseif ($Mode -eq 'Cpu') {
    Pip-Install @('--resume-retries','8','torch==2.7.1','torchvision==0.22.1','--index-url','https://pypi.org/simple')
    Pip-Install @('-r','requirements-ai.txt')
}
& $studioPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency check failed.' }
Write-Host 'Installation completed. Restart the platform to use newly installed modules.'
