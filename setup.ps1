param(
    [switch]$SkipDataset,
    [switch]$SkipTraining,
    [switch]$DownloadDataset
)

$ErrorActionPreference = "Stop"
$DemoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvRoot = Join-Path $DemoRoot ".venv"
$VenvPython = Join-Path $VenvRoot "Scripts\python.exe"

function Assert-NativeSuccess {
    param([string]$Step)
    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed with exit code $LASTEXITCODE."
    }
}

if ($SkipDataset -and $DownloadDataset) {
    throw "-SkipDataset and -DownloadDataset cannot be used together."
}

if (-not (Test-Path -LiteralPath $VenvPython)) {
    $PythonCommand = Get-Command python -ErrorAction SilentlyContinue
    $PyLauncher = Get-Command py -ErrorAction SilentlyContinue
    $PythonExecutable = $null
    $PythonArguments = @()

    if ($PythonCommand) {
        & $PythonCommand.Source -c "import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] < (3, 13) else 1)" 2>$null
        if ($LASTEXITCODE -eq 0) {
            $PythonExecutable = $PythonCommand.Source
        }
    }

    if (-not $PythonExecutable -and $PyLauncher) {
        foreach ($Version in @("3.12", "3.11", "3.10")) {
            & $PyLauncher.Source "-$Version" -c "import sys" 2>$null
            if ($LASTEXITCODE -eq 0) {
                $PythonExecutable = $PyLauncher.Source
                $PythonArguments = @("-$Version")
                break
            }
        }
    }

    if (-not $PythonExecutable) {
        throw "Python 3.10--3.12 was not found. Install Python, then rerun setup.ps1."
    }

    & $PythonExecutable @PythonArguments -m venv $VenvRoot
    Assert-NativeSuccess "Creating the Python environment"
}

& $VenvPython -m pip install --upgrade pip
Assert-NativeSuccess "Upgrading pip"
& $VenvPython -m pip install -r (Join-Path $DemoRoot "requirements.txt")
Assert-NativeSuccess "Installing runtime dependencies"
& $VenvPython -m pip install -e $DemoRoot
Assert-NativeSuccess "Installing DenoiseAPT"

if (-not $SkipDataset) {
    if ($DownloadDataset) {
        & $VenvPython (Join-Path $DemoRoot "scripts\download_data.py")
        Assert-NativeSuccess "Downloading the optional TSB-AD guided case"
    }
    & $VenvPython (Join-Path $DemoRoot "scripts\prepare_demo_case.py") --ensure-demo-case
    Assert-NativeSuccess "Preparing demo cases"
}

# Kept as an accepted legacy switch. The frozen demonstration runtime is packaged and
# setup never retrains or replaces its frozen checkpoints.
if ($SkipTraining) {
    Write-Verbose "-SkipTraining is retained for compatibility; no setup training is required."
}

Write-Host "DenoiseAPT setup is complete. Run .\run_demo.ps1"
