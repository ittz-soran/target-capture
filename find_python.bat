@echo off
REM Used by setup_windows.bat and update_windows.bat. Finds a Python 3.10 - 3.13 and puts the
REM command in PY. If there is none, installs Python 3.11 with winget (built into Windows 10/11).
set PY=
for %%v in (3.11 3.12 3.10 3.13) do (
  if not defined PY py -%%v -c "import sys" >nul 2>nul && set "PY=py -%%v"
)
if defined PY exit /b 0
python -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,13) else 1)" >nul 2>nul && set "PY=python"
if defined PY exit /b 0
for %%v in (311 312 310 313) do (
  if not defined PY if exist "%LOCALAPPDATA%\Programs\Python\Python%%v\python.exe" set PY="%LOCALAPPDATA%\Programs\Python\Python%%v\python.exe"
)
if defined PY exit /b 0

echo No Python 3.10 - 3.13 found on this PC. Installing Python 3.11 ...
where winget >nul 2>nul || goto manual
winget install -e --id Python.Python.3.11 --scope user --accept-package-agreements --accept-source-agreements
if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" set PY="%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
if defined PY exit /b 0
py -3.11 -c "import sys" >nul 2>nul && set "PY=py -3.11"
if defined PY exit /b 0

:manual
echo.
echo Could not install Python automatically. Please install Python 3.11 from
echo   https://www.python.org/downloads/release/python-3119/   (Windows installer 64-bit)
echo tick "Add python.exe to PATH", then run this file again.
exit /b 1
