@echo off
echo =============================================
echo  Cats vs Dogs MLOps - Windows Setup
echo =============================================

:: Always run from the cats-dogs-mlops directory
cd /d "%~dp0.."
echo Working directory: %CD%

set "PY_CMD=python"
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    py -3 --version >nul 2>&1
    if %ERRORLEVEL% NEQ 0 (
        echo ERROR: Python not found. Install Python 3.9+ from https://python.org
        pause & exit /b 1
    )
    set "PY_CMD=py -3"
)

echo Installing dependencies...
%PY_CMD% -m pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 ( echo pip install failed & pause & exit /b 1 )
echo.
echo Setup complete!
echo.
echo Quick start:
echo   1. %PY_CMD% src\models\train.py --dry-run
echo   2. docker build -t cats-dogs-api:latest .
echo   3. docker compose up -d
echo   4. %PY_CMD% scripts\smoke_test.py
echo.
pause
