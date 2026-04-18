@echo off
echo ======================================================
echo   Suspicious Activity Detection - One-Click Launcher
echo ======================================================
echo.

if not exist ".venv" (
    echo [ERROR] Virtual environment not found! 
    echo Please run setup first.
    pause
    exit /b
)

echo [INFO] Activating environment and launching app...
.\.venv\Scripts\python.exe src\main.py
echo.
echo [INFO] Session ended.
pause
