@echo off
cd /d "%~dp0backend"

rem Avoid tensorflow/oneDNN noise and console mojibake
set TF_CPP_MIN_LOG_LEVEL=3
set PYTHONIOENCODING=utf-8

echo ============================================================
echo  Starting backend  ^-^>  http://127.0.0.1:8000/api/v1/docs
echo ============================================================
echo.

rem --reload-dir app: 只监听 app/，改 tests/ 下的脚本不会触发后端热重启
.\venv\Scripts\python.exe -m uvicorn app.main:app --reload --reload-dir app --host 127.0.0.1 --port 8000

echo.
echo Backend stopped.
pause
