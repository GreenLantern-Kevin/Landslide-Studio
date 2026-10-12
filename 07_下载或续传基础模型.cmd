@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" "scripts\download_base.py" all
pause
