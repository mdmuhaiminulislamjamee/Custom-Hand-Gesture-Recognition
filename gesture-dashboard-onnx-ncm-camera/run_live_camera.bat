@echo off
setlocal
cd /d "%~dp0"
set OPENBLAS_NUM_THREADS=1

echo ========================================================
echo   Starting 11-Command ONNX Gesture Recognition...
echo   Automatic two-hand selection is enabled.
echo ========================================================
echo.
echo   Hold a clear gesture. If both hands gesture, bring one forward.
echo   Gray hands are observed; the selected hand is highlighted.
echo   Hand-only detection; body position and clothing are not filters.
echo   Dorsal is a hard negative: No Gesture, with no action.
echo   Gestures can qualify in front of the face; no face filter.
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
