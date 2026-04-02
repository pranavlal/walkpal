@echo off
REM WalkingPal Windows Launcher
REM Runs the windowless entry point (.pyw) using the local virtual environment.

set SCRIPT_DIR=%~dp0
set VENV_PYTHON=%SCRIPT_DIR%.venv\Scripts\pythonw.exe

if not exist "%VENV_PYTHON%" (
    echo Error: Virtual environment not found. Please run WalkingPal_Setup.ps1 first.
    pause
    exit /b 1
)

start "" "%VENV_PYTHON%" "%SCRIPT_DIR%WalkingPal.pyw" %*
