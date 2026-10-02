@echo off
REM Opens the Target Capture web page in your browser.
cd /d "%~dp0"
call .venv\Scripts\activate.bat
REM Largest upload allowed, from the Settings tab (applies when the page starts)
set MB=20000
for /f %%m in ('python -m targetcapture.settings max_upload_mb 2^>nul') do set MB=%%m
streamlit run app.py --server.maxUploadSize %MB%
