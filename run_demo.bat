@echo off
:: Force the script to run from its own directory
cd /d "%~dp0"

title Suspicious Activity Detection - Hybrid AI
echo [INFO] Starting Hybrid AI Surveillance System...

:: Check if the activation script exists
if not exist .venv\Scripts\activate.bat goto NO_VENV

echo [INFO] Activating Virtual Environment...
call .venv\Scripts\activate.bat

echo [INFO] Launching UI...
python src\main.py

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Application crashed with exit code %ERRORLEVEL%
    pause
)
exit /b

:NO_VENV
echo [ERROR] Virtual environment (.venv) not found!
echo [INFO] Please ensure the .venv folder is in this directory.
pause
exit /b
