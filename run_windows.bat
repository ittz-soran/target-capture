@echo off
REM Opens the Target Capture web page in your browser.
cd /d "%~dp0"
call .venv\Scripts\activate.bat
streamlit run app.py
