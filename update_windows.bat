@echo off
REM Updates Target Capture to the latest version from GitHub, keeping your videos,
REM index (targetcapture.db), results and .venv. Needs Git for Windows (git-scm.com).
cd /d "%~dp0"
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
if not exist .venv (py -3.11 -m venv .venv || python -m venv .venv)
call .venv\Scripts\activate.bat
pip install -q -r requirements.txt
echo.
echo Update done. Start it with run_windows.bat
pause
