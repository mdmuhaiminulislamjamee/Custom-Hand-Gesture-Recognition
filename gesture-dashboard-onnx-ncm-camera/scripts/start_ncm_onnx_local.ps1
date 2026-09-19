param(
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$ProjectDirectory = Split-Path -Parent $PSScriptRoot
$RuntimeDirectory = Join-Path $ProjectDirectory ".runtime"
$PythonExecutable = Join-Path $ProjectDirectory ".venv\Scripts\python.exe"
$VinextScript = Join-Path $ProjectDirectory "node_modules\vinext\dist\cli.js"

& (Join-Path $PSScriptRoot "stop_ncm_onnx_local.ps1")

if (-not (Test-Path -LiteralPath $PythonExecutable)) {
    throw "NCM ONNX Python environment missing. Run setup_ncm_onnx_dashboard.bat first."
}
if (-not (Test-Path -LiteralPath $VinextScript)) {
    throw "NCM ONNX frontend environment missing. Run setup_ncm_onnx_dashboard.bat first."
}

New-Item -ItemType Directory -Force -Path $RuntimeDirectory | Out-Null
$MatplotlibCache = Join-Path $RuntimeDirectory "matplotlib"
New-Item -ItemType Directory -Force -Path $MatplotlibCache | Out-Null
$env:MPLCONFIGDIR = $MatplotlibCache
$env:NODE_OPTIONS = "--max-old-space-size=512 --max-semi-space-size=4"
$env:OPENBLAS_NUM_THREADS = if ($env:OPENBLAS_NUM_THREADS) { $env:OPENBLAS_NUM_THREADS } else { "1" }
$env:OMP_NUM_THREADS = if ($env:OMP_NUM_THREADS) { $env:OMP_NUM_THREADS } else { "1" }
$env:NEXT_PUBLIC_GESTURE_API_URL = "http://127.0.0.1:8200"
$env:NCM_HOST_IP = if ($env:NCM_HOST_IP) { $env:NCM_HOST_IP } else { "192.168.50.1" }
$env:NCM_DEVICE_IP = if ($env:NCM_DEVICE_IP) { $env:NCM_DEVICE_IP } else { "192.168.50.2" }
$env:NCM_TCP_PORT = if ($env:NCM_TCP_PORT) { $env:NCM_TCP_PORT } else { "5000" }
$env:NCM_DISCOVERY_PORT = if ($env:NCM_DISCOVERY_PORT) { $env:NCM_DISCOVERY_PORT } else { "5001" }
$env:NCM_DISCOVERY_PAYLOAD = if ($env:NCM_DISCOVERY_PAYLOAD) { $env:NCM_DISCOVERY_PAYLOAD } else { "JL-CAMERA-DISCOVER" }
$env:GESTURE_ENVIRONMENT = "production"
$env:GESTURE_REQUIRE_ARTIFACT_MANIFEST = "true"
$env:GESTURE_ALLOW_FORCE_LEARNING = "false"
$env:GESTURE_ALLOW_ARTIFACT_RELOAD = "false"
$BackendOutput = Join-Path $RuntimeDirectory "backend.stdout.log"
$BackendError = Join-Path $RuntimeDirectory "backend.stderr.log"
$FrontendOutput = Join-Path $RuntimeDirectory "frontend.stdout.log"
$FrontendError = Join-Path $RuntimeDirectory "frontend.stderr.log"
$NodeCommand = Get-Command node -ErrorAction Stop

Write-Host "Running ONNX + NCM deployment preflight..." -ForegroundColor Cyan
& $PythonExecutable (Join-Path $PSScriptRoot "onnx_preflight.py")
if ($LASTEXITCODE -ne 0) {
    throw "ONNX preflight failed. Review the output above."
}

Write-Host "Building the independent ONNX + NCM dashboard..." -ForegroundColor Cyan
& $NodeCommand.Source $VinextScript "build"
if ($LASTEXITCODE -ne 0) {
    throw "Frontend build failed. Review the output above."
}

$Backend = Start-Process -FilePath $PythonExecutable `
    -ArgumentList "-m", "uvicorn", "backend.app:app", "--host", "127.0.0.1", "--port", "8200" `
    -WorkingDirectory $ProjectDirectory -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput $BackendOutput -RedirectStandardError $BackendError

$QuotedVinextScript = '"' + $VinextScript + '"'
$Frontend = Start-Process -FilePath $NodeCommand.Source `
    -ArgumentList $QuotedVinextScript, "start", "--hostname", "127.0.0.1", "--port", "3200" `
    -WorkingDirectory $ProjectDirectory -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput $FrontendOutput -RedirectStandardError $FrontendError

Set-Content -LiteralPath (Join-Path $RuntimeDirectory "backend.pid") -Value $Backend.Id
Set-Content -LiteralPath (Join-Path $RuntimeDirectory "frontend.pid") -Value $Frontend.Id

$BackendReady = $false
$FrontendReady = $false
for ($Attempt = 0; $Attempt -lt 60; $Attempt++) {
    Start-Sleep -Milliseconds 500
    try { $BackendReady = (Invoke-WebRequest -UseBasicParsing "http://127.0.0.1:8200/readyz" -TimeoutSec 1).StatusCode -eq 200 } catch { $BackendReady = $false }
    try { $FrontendReady = (Invoke-WebRequest -UseBasicParsing "http://127.0.0.1:3200" -TimeoutSec 1).StatusCode -eq 200 } catch { $FrontendReady = $false }
    if ($BackendReady -and $FrontendReady) { break }
}

if (-not ($BackendReady -and $FrontendReady)) {
    Write-Host "One of the independent ONNX + NCM servers did not become ready." -ForegroundColor Red
    Write-Host "Review logs in $RuntimeDirectory"
    & (Join-Path $PSScriptRoot "stop_ncm_onnx_local.ps1")
    exit 1
}

Write-Host "Gesture Control Lab ONNX + NCM is running." -ForegroundColor Green
Write-Host "Dashboard: http://127.0.0.1:3200"
Write-Host "API docs:  http://127.0.0.1:8200/docs"
Write-Host "Board link: $env:NCM_HOST_IP -> $env:NCM_DEVICE_IP`:$env:NCM_TCP_PORT"
Write-Host "This dashboard supports both the PC webcam and USB-NCM board camera."
if (-not $NoBrowser) {
    Start-Process "http://127.0.0.1:3200"
}
