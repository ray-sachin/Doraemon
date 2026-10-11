@echo off
setlocal EnableDelayedExpansion
title Doraemon Certificate Scanner

:: Ensure current working directory is the script directory
cd /d "%~dp0"

echo ===================================================================
echo                     DORAEMON CERTIFICATE SCANNER
echo             Offline Multi-Layer Visual Extraction Engine
echo                     Developed by Sachin Kumar Ray
echo ===================================================================
echo.

:: ---------------------------------------------------------------------
:: 1. Check Python Environment
:: ---------------------------------------------------------------------
echo [1/3] Checking Python environment...
set "PY_CMD="

python --version >nul 2>&1
if %ERRORLEVEL% equ 0 (
    set "PY_CMD=python"
    goto :python_found
)

py -3 --version >nul 2>&1
if %ERRORLEVEL% equ 0 (
    set "PY_CMD=py -3"
    goto :python_found
)

:: If Python is not found
echo.
echo [NOTICE] Python is not detected on this system.
echo Doraemon requires Python 3.10+ to run.
echo.
set /p "INSTALL_PY=Do you want to automatically install Python 3.11 now? [Y/N]: "
if /i "%INSTALL_PY%"=="Y" goto :install_python
goto :python_missing_exit

:install_python
where winget >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo Installing Python 3.11 using Windows Package Manager...
    winget install -e --id Python.Python.3.11 --accept-package-agreements --accept-source-agreements
) else (
    echo Downloading official Python installer...
    powershell -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe' -OutFile '%TEMP%\python_installer.exe'"
    echo Running installer silently...
    "%TEMP%\python_installer.exe" /quiet InstallAllUsers=0 PrependPath=1 Include_pip=1
    del "%TEMP%\python_installer.exe" >nul 2>&1
)
echo.
echo Python installation complete. Please run start.bat again so PATH is refreshed.
pause
exit /b 0

:python_missing_exit
echo.
echo Python is required to run Doraemon. You can install it from https://python.org
pause
exit /b 1

:python_found
for /f "tokens=*" %%v in ('%PY_CMD% --version') do echo   - Found %%v

:: ---------------------------------------------------------------------
:: 2. Check Python Dependencies
:: ---------------------------------------------------------------------
echo.
echo [2/3] Verifying required libraries...
%PY_CMD% -c "import cv2, easyocr, requests, rapidfuzz, pypdfium2" >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo   - Core dependencies are installed and ready.
    goto :ollama_check
)

:: Check if virtual environment exists with dependencies
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
    python -c "import cv2, easyocr, requests, rapidfuzz, pypdfium2" >nul 2>&1
    if !ERRORLEVEL! equ 0 (
        set "PY_CMD=python"
        echo   - Virtual environment libraries ready.
        goto :ollama_check
    )
)

echo.
echo [NOTICE] Some required libraries are missing.
set /p "INSTALL_PKGS=Do you want to install dependencies from requirements.txt now? [Y/N]: "
if /i not "%INSTALL_PKGS%"=="Y" (
    echo [NOTICE] Skipping package installation. Some features may not work.
    goto :ollama_check
)

echo Upgrading pip...
%PY_CMD% -m pip install --upgrade pip
echo Installing libraries from requirements.txt...
%PY_CMD% -m pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo.
    echo [WARNING] Some dependencies failed to install.
)

:ollama_check
:: ---------------------------------------------------------------------
:: 3. Verify Ollama and Local Vision Models
:: ---------------------------------------------------------------------
echo.
echo [3/3] Checking Ollama local AI runtime...
where ollama >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo   - Ollama is not installed. Classical OCR and QR extraction will run normally.
    goto :launch_scanner
)

echo   - Ollama detected.
:: Start background service if not responding
curl -s http://localhost:11434/api/tags >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo   - Starting local Ollama background service...
    start /B "" ollama serve >nul 2>&1
    timeout /t 2 /nobreak >nul
)

:: Check Primary Model: qwen2.5vl:3b
ollama list 2>nul | findstr /i "qwen2.5vl" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo   - Primary model qwen2.5vl:3b is not yet downloaded.
    set /p "PULL_QWEN=Do you want to download qwen2.5vl:3b [approx 3.2 GB]? [Y/N]: "
    if /i "!PULL_QWEN!"=="Y" (
        echo Downloading qwen2.5vl:3b...
        ollama pull qwen2.5vl:3b
    )
) else (
    echo   - Primary vision model qwen2.5vl:3b is ready.
)

:: Check Secondary Model: minicpm-v
ollama list 2>nul | findstr /i "minicpm-v" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo   - Secondary model minicpm-v is not yet downloaded.
    set /p "PULL_MINI=Do you want to download minicpm-v [approx 5.5 GB]? [Y/N]: "
    if /i "!PULL_MINI!"=="Y" (
        echo Downloading minicpm-v...
        ollama pull minicpm-v
    )
) else (
    echo   - Secondary vision model minicpm-v is ready.
)

:launch_scanner
:: ---------------------------------------------------------------------
:: 4. Launch Doraemon Scanner
:: ---------------------------------------------------------------------
echo.
echo ===================================================================
echo Environment verification complete. Launching Doraemon...
echo ===================================================================

%PY_CMD% scan.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo An unexpected error occurred while running scan.py.
    pause
)
