@echo off
setlocal
cd /d "%~dp0"
set OPENBLAS_NUM_THREADS=1

echo ========================================================
echo   Starting 10-Command ONNX Gesture Recognition...
echo ========================================================
echo.
echo   [1] Run with PC Webcam (Default)
echo   [2] Run with USB-NCM Board Camera (192.168.50.2:5000)
echo.
set /p choice="Choose camera source [1 or 2, default: 1]: "

if "%choice%"=="2" (
    echo.
    echo Connecting to USB-NCM Camera...
    .\.venv\Scripts\python.exe scripts\run_live_camera.py --source ncm
) else (
    echo.
    echo Opening PC Webcam...
    .\.venv\Scripts\python.exe scripts\run_live_camera.py --source webcam
)

pause
