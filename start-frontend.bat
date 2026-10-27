@echo off
cd /d "%~dp0frontend"

echo ============================================================
echo  Starting frontend  ^-^>  http://localhost:3000
echo ============================================================
echo.

rem Redirect esbuild temp files into the project (avoids sandbox temp issues)
if not exist ".tmp" mkdir ".tmp"
set TMP=%~dp0frontend\.tmp
set TEMP=%~dp0frontend\.tmp

call npm run dev

echo.
echo Frontend stopped.
pause
