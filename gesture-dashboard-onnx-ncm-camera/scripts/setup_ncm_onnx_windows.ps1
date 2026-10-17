$ErrorActionPreference = "Stop"
$ProjectDirectory = Split-Path -Parent $PSScriptRoot
$VirtualEnvironment = Join-Path $ProjectDirectory ".venv"
$PythonExecutable = Join-Path $VirtualEnvironment "Scripts\python.exe"
$env:NODE_OPTIONS = "--max-old-space-size=512 --max-semi-space-size=4"
$env:OPENBLAS_NUM_THREADS = if ($env:OPENBLAS_NUM_THREADS) { $env:OPENBLAS_NUM_THREADS } else { "1" }
$env:OMP_NUM_THREADS = if ($env:OMP_NUM_THREADS) { $env:OMP_NUM_THREADS } else { "1" }

Write-Host "Preparing the independent Gesture Control Lab ONNX + NCM package..." -ForegroundColor Cyan

if (-not [Environment]::Is64BitOperatingSystem) {
    throw "A 64-bit Windows installation is required."
}

if (-not (Test-Path -LiteralPath $PythonExecutable)) {
    $PythonCandidates = @()
    foreach ($CommandName in @("python", "python3")) {
        $PythonCommand = Get-Command $CommandName -ErrorAction SilentlyContinue
        if ($null -ne $PythonCommand) {
            $PythonCandidates += $PythonCommand.Source
        }
    }

    foreach ($PythonRoot in @(
        (Join-Path $env:LOCALAPPDATA "Programs\Python"),
        $env:ProgramFiles
    )) {
        if (Test-Path -LiteralPath $PythonRoot) {
            $PythonCandidates += Get-ChildItem -LiteralPath $PythonRoot -Filter python.exe -Recurse -Depth 2 -ErrorAction SilentlyContinue |
                Select-Object -ExpandProperty FullName
        }
    }

    $BasePython = $null
    foreach ($Candidate in ($PythonCandidates | Select-Object -Unique)) {
        try {
            & $Candidate -c "import sys; assert sys.version_info[:2] in {(3, 11), (3, 12)} and sys.maxsize > 2**32" 2>$null
            if ($LASTEXITCODE -eq 0) {
                $BasePython = $Candidate
                break
            }
        } catch {
            # Ignore Microsoft Store aliases and incompatible installations.
        }
    }

    if ($null -ne $BasePython) {
        & $BasePython -m venv $VirtualEnvironment
    } else {
        $PyLauncherCandidates = @()
        $PyLauncher = Get-Command py -ErrorAction SilentlyContinue
        if ($null -ne $PyLauncher) {
            $PyLauncherCandidates += $PyLauncher.Source
        }
        $PyLauncherCandidates += Join-Path $env:LOCALAPPDATA "Programs\Python\Launcher\py.exe"
        $PyLauncherCandidates += Join-Path $env:WINDIR "py.exe"

        $EnvironmentCreated = $false
        foreach ($PyLauncherExecutable in ($PyLauncherCandidates | Select-Object -Unique)) {
            if (-not (Test-Path -LiteralPath $PyLauncherExecutable)) {
                continue
            }
            foreach ($Version in @("3.12", "3.11")) {
                try {
                    & $PyLauncherExecutable "-$Version" -c "import sys; assert sys.version_info[:2] in {(3, 11), (3, 12)} and sys.maxsize > 2**32" 2>$null
                    if ($LASTEXITCODE -eq 0) {
                        & $PyLauncherExecutable "-$Version" -m venv $VirtualEnvironment
                        $EnvironmentCreated = $true
                        break
                    }
                } catch {
                    # Continue to the next supported launcher/version pair.
                }
            }
            if ($EnvironmentCreated) {
                break
            }
        }

        if (-not $EnvironmentCreated) {
            throw "Install 64-bit Python 3.11 or 3.12, then rerun setup."
        }
    }

    if (-not (Test-Path -LiteralPath $PythonExecutable)) {
        throw "Python was found, but the project virtual environment could not be created."
    }
}

& $PythonExecutable -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "Could not upgrade pip. Check the internet connection, then rerun setup."
}
& $PythonExecutable -m pip install -r (Join-Path $ProjectDirectory "requirements.lock.txt")
if ($LASTEXITCODE -ne 0) {
    throw "Could not install Python dependencies. Check the internet connection and pip output above."
}
& $PythonExecutable (Join-Path $PSScriptRoot "check_onnx_runtime.py")
if ($LASTEXITCODE -ne 0) {
    Write-Host "Install the Microsoft Visual C++ 2015-2022 x64 Redistributable:" -ForegroundColor Yellow
    Write-Host "https://aka.ms/vc14/vc_redist.x64.exe"
    throw "ONNX Runtime native dependency check failed."
}

if (-not (Test-Path -LiteralPath (Join-Path $ProjectDirectory "models\hand_landmarker.task"))) {
    & $PythonExecutable (Join-Path $PSScriptRoot "download_hand_landmarker.py")
}

Push-Location $ProjectDirectory
try {
    & $PythonExecutable (Join-Path $PSScriptRoot "download_face_detector.py")
    if ($LASTEXITCODE -ne 0) { throw "Could not install the face guard model." }
    $env:CI = "true"
    $PnpmCommand = Get-Command pnpm -ErrorAction SilentlyContinue
    if ($null -ne $PnpmCommand) {
        & $PnpmCommand.Source install --frozen-lockfile
    } else {
        $CorepackCommand = Get-Command corepack -ErrorAction Stop
        & $CorepackCommand.Source pnpm install --frozen-lockfile
    }
} finally {
    Pop-Location
}

& $PythonExecutable (Join-Path $PSScriptRoot "build_artifact_manifest.py")
& $PythonExecutable (Join-Path $PSScriptRoot "onnx_preflight.py")
if ($LASTEXITCODE -ne 0) {
    throw "ONNX deployment preflight failed."
}

Write-Host "ONNX + NCM setup complete." -ForegroundColor Green
Write-Host "Start it with start_ncm_onnx_dashboard.bat"
Write-Host "Dashboard: http://127.0.0.1:3200"
Write-Host "API docs:  http://127.0.0.1:8200/docs"
