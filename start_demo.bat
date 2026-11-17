@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment not found.
    echo Run these commands first:
    echo   python -m venv .venv
    echo   .venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

echo Starting BookMind at http://127.0.0.1:8000/demo/

".venv\Scripts\python.exe" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)" >nul 2>&1
if %errorlevel%==0 (
    echo BookMind API is already running - opening the demo.
) else (
    start "BookMind API" ".venv\Scripts\python.exe" -m uvicorn api.main:app --host 127.0.0.1 --port 8000
    timeout /t 4 /nobreak >nul
)

start "" "http://127.0.0.1:8000/demo/"
echo Demo opened. Close the "BookMind API" window when you are finished.
