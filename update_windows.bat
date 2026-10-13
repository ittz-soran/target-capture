@echo off
REM Updates Target Capture to the latest version from GitHub, keeping your videos,
REM index (targetcapture.db), results and .venv. Needs Git for Windows (git-scm.com).
REM Settings tab: runs a copy of this file with "restart <app folder>" (the update replaces this file),
REM waits for the web page to close, and starts it again at the end.
if "%~2"=="" (cd /d "%~dp0") else (cd /d "%~2")
if "%~1"=="restart" timeout /t 5 /nobreak >nul
REM Stop this app and live monitoring first (it runs hidden): Windows cannot replace files they use
echo Stopping Target Capture and live monitoring...
type nul > live.stop
powershell -NoProfile -Command "Start-Sleep 3; Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python|^streamlit' -and $_.CommandLine -match 'targetcapture|streamlit' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
timeout /t 2 /nobreak >nul
set REPO=https://github.com/ittz-soran/target-capture.git
where git >nul 2>nul || (echo Git is not installed. Get it from https://git-scm.com/download/win & pause & exit /b 1)
if not exist .git (
  echo First update: linking this folder to GitHub...
  git init -q -b main
  git remote add origin %REPO%
)
git fetch -q origin main || (echo Could not reach GitHub. Check your internet or GitHub login. & pause & exit /b 1)
git reset -q --hard origin/main
echo Code updated. Installing any new libraries...
if not exist .venv\Scripts\python.exe (
  echo Python environment not set up yet in this folder, running setup...
  call setup_windows.bat
  exit /b
)
call .venv\Scripts\activate.bat
pip install -q -r requirements.txt
if errorlevel 1 (echo Installing libraries failed, see the error above. & pause & exit /b 1)
if exist .venv\gpu.txt (
  REM requirements.txt brings back the CPU-only onnxruntime; swap the GPU one in again
  pip uninstall -q -y onnxruntime onnxruntime-gpu
  pip install -q --force-reinstall --no-deps "onnxruntime-gpu<1.27"
  pip install -q "onnxruntime-gpu<1.27"
)
echo.
echo Update done. Start it with run_windows.bat
if "%~1"=="restart" (start "" run_windows.bat & exit /b 0)
pause
