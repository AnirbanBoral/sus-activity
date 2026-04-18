@echo off
:: Force the script to run from its own directory
cd /d "%~dp0"

title Hybrid AI Surveillance - Auto-Setup & Launch
echo ===================================================
echo   Hybrid AI Surveillance System - Professional
echo ===================================================

:: 1. Check for Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH!
    echo Please install Python 3.9+ from python.org and try again.
    pause
    exit /b
)

:: 2. Check for Virtual Environment
if exist .venv\Scripts\activate.bat goto ACTIVATE

echo [INFO] Virtual environment (.venv) not found.
echo [INFO] Starting automatic environment setup...

echo [INFO] Creating virtual environment...
python -m venv .venv
if %errorlevel% neq 0 goto VENV_ERROR

echo [INFO] Upgrading pip...
.venv\Scripts\python.exe -m pip install --upgrade pip

echo [INFO] Installing required libraries (this may take a minute)...
.venv\Scripts\python.exe -m pip install -r requirements.txt
if %errorlevel% neq 0 goto INSTALL_ERROR

echo [INFO] Setup complete!

:ACTIVATE
echo [INFO] Activating Virtual Environment...
call .venv\Scripts\activate.bat

echo [INFO] Launching AI Surveillance Hub...
python src\main.py

if %ERRORLEVEL% NEQ 0 goto APP_ERROR
exit /b

:VENV_ERROR
echo [ERROR] Failed to create virtual environment!
pause
exit /b

:INSTALL_ERROR
echo [ERROR] Failed to install requirements!
pause
exit /b

:APP_ERROR
echo [ERROR] Application crashed with exit code %ERRORLEVEL%
pause
exit /b
