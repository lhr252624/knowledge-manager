@echo off
rem Free ports 8000 (backend) and 3000 (frontend)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\free-ports.ps1"
echo.
pause
