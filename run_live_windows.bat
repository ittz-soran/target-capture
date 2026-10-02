@echo off
REM Records the live cameras added in the "Live cameras" tab and indexes them as they record.
REM Keep this window open; close it (or press Ctrl+C) to stop.
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python -m targetcapture live
pause
