param(
    [int]$Port = 8765,
    [switch]$SkipSetup
)

$ErrorActionPreference = "Stop"
$DemoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $DemoRoot ".venv\Scripts\python.exe"

if (-not $SkipSetup -and -not (Test-Path -LiteralPath $VenvPython)) {
    & (Join-Path $DemoRoot "setup.ps1")
}

if (-not (Test-Path -LiteralPath $VenvPython)) {
    throw "Python environment not found. Run setup.ps1 first."
}

Push-Location $DemoRoot
try {
    & $VenvPython "server.py" --host 127.0.0.1 --port $Port
}
finally {
    Pop-Location
}

