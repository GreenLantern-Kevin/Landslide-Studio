@echo off
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\rebuild_environment.ps1" -Mode Ai
pause
