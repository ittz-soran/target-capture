@echo off
REM Makes Target Capture use an NVIDIA graphics card (GeForce GTX / RTX). Run once, after setup_windows.bat.
REM Needs the NVIDIA driver (nvidia.com/drivers). Downloads about 3 GB.
REM The Settings tab runs a copy of this file with "restart <app folder>": it waits for the web page to
REM close, stops live monitoring (Windows cannot replace files in use) and opens the page again at the end.
if "%~2"=="" (cd /d "%~dp0") else (cd /d "%~2")
if "%~1"=="restart" (
  timeout /t 5 /nobreak >nul
  echo Stopping Target Capture and live monitoring...
  type nul > live.stop
  powershell -NoProfile -Command "Start-Sleep 3; Get-CimInstance Win32_Process | Where-Object { ($_.Name -match '^python|^streamlit' -and $_.CommandLine -match 'targetcapture|streamlit') -or ($_.Name -match '^ffmpeg' -and $_.ExecutablePath -match 'imageio_ffmpeg') } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
  timeout /t 2 /nobreak >nul
)
if not exist .venv\Scripts\python.exe (echo Run setup_windows.bat first. & pause & exit /b 1)
where nvidia-smi >nul 2>nul || (echo No NVIDIA driver found. Install the latest driver from https://www.nvidia.com/Download/index.aspx and run this again. & pause & exit /b 1)
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
call .venv\Scripts\activate.bat
echo.
echo Installing the GPU version of PyTorch (large download, please wait)...
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
if errorlevel 1 (echo PyTorch GPU install failed, see the error above. & pause & exit /b 1)
call :onnx_gpu
if errorlevel 1 (echo ONNX Runtime GPU install failed, see the error above. & pause & exit /b 1)
echo gpu> .venv\gpu.txt
echo.
python -c "from targetcapture.gpu import describe; print(describe())"
echo.
echo Done. Start it with run_windows.bat; the left side of the page shows "GPU: <your card>".
if "%~1"=="restart" (start "" run_windows.bat & exit /b 0)
pause
exit /b 0

:onnx_gpu
REM onnxruntime and onnxruntime-gpu share one folder, so remove both and put the GPU one back.
REM Below 1.27: newer builds need CUDA 13, which the PyTorch cu126 install does not ship (and older GTX cards cannot run).
pip uninstall -y onnxruntime onnxruntime-gpu
pip install --force-reinstall --no-deps "onnxruntime-gpu<1.27"
pip install "onnxruntime-gpu<1.27"
exit /b
