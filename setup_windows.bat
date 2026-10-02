@echo off
REM One-time setup on Windows. Finds (or installs) Python 3.11. Also needs "Microsoft C++ Build Tools"
REM (Desktop development with C++), which InsightFace needs to install.
cd /d "%~dp0"
call find_python.bat
if errorlevel 1 (pause & exit /b 1)
if exist .venv if not exist .venv\Scripts\python.exe rmdir /s /q .venv
if not exist .venv\Scripts\python.exe %PY% -m venv .venv
if not exist .venv\Scripts\python.exe (echo Could not create .venv with %PY% & pause & exit /b 1)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
  echo.
  echo Install failed. If the error mentions insightface or "Microsoft Visual C++ 14.0",
  echo install Microsoft C++ Build Tools and run this file again.
  pause
  exit /b 1
)
echo.
echo Setup done. Double-click run_windows.bat to open Target Capture.
pause
