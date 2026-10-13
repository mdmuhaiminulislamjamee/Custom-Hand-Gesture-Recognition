@echo off
setlocal
cd /d "%~dp0"
set OPENBLAS_NUM_THREADS=1

echo ========================================================
echo   Starting 12-Command ONNX Gesture Recognition...
echo   Automatic two-hand selection is enabled.
echo ========================================================
echo.
echo   Hold a clear gesture. If both hands gesture, bring one forward.
echo   Gray hands are observed; the selected hand is highlighted.
echo   Dorsal below the waist is no gesture when the body is visible.
echo   A clear listed gesture can qualify beside the face.
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
