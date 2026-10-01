@echo off
REM One-time setup on Windows. Needs Python 3.11 from python.org ("Add python.exe to PATH" ticked)
REM and "Microsoft C++ Build Tools" (Desktop development with C++), which InsightFace needs to install.
cd /d "%~dp0"
py -3.11 -m venv .venv || python -m venv .venv
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
