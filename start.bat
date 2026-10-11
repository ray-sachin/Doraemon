@echo off
setlocal enabledelayedexpansion
title Doraemon Certificate Scanner
chcp 65001 >nul
cls

echo ===================================================================
echo                     DORAEMON CERTIFICATE SCANNER
echo             Offline Multi-Layer Visual Extraction Engine
echo                     Developed by Sachin Kumar Ray
echo ===================================================================
echo.

:: ---------------------------------------------------------------------
:: 1. Check Python Installation
:: ---------------------------------------------------------------------
echo [1/4] Checking Python environment...
set "PY_CMD="

python --version >nul 2>&1
if %ERRORLEVEL% equ 0 (
    set "PY_CMD=python"
) else (
    py -3 --version >nul 2>&1
    if !ERRORLEVEL! equ 0 (
        set "PY_CMD=py -3"
    )
)

if "%PY_CMD%"=="" (
    echo.
    echo [NOTICE] Python is not installed on this system.
    echo Doraemon requires Python 3.10+ to run OCR and visual extraction.
    echo.
    set /p "INSTALL_PY=Do you want to automatically install Python 3.11 now? (Y/N): "
    if /i "!INSTALL_PY!"=="Y" (
        where winget >nul 2>&1
        if !ERRORLEVEL! equ 0 (
            echo Installing Python 3.11 via Windows Package Manager (winget)...
            winget install -e --id Python.Python.3.11 --accept-package-agreements --accept-source-agreements
        ) else (
            echo Downloading Python 3.11 installer from official Python repository...
            powershell -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe' -OutFile '%TEMP%\python_installer.exe'"
            echo Running Python installer silently...
            "%TEMP%\python_installer.exe" /quiet InstallAllUsers=0 PrependPath=1 Include_pip=1
            del "%TEMP%\python_installer.exe" >nul 2>&1
        )
        echo.
        echo Python installation complete! Please close and reopen this window,
        echo or run start.bat again so your system PATH updates.
        pause
        exit /b 0
    ) else (
        echo Cannot continue without Python. Please install Python from https://python.org
        pause
        exit /b 1
    )
)

for /f "tokens=*" %%v in ('%PY_CMD% --version') do echo   - Found: %%v

:: ---------------------------------------------------------------------
:: 2. Setup Virtual Environment
:: ---------------------------------------------------------------------
echo.
echo [2/4] Verifying isolated virtual environment (.venv)...
if not exist ".venv\Scripts\activate.bat" (
    echo   - Creating new virtual environment in .venv...
    %PY_CMD% -m venv .venv
    if !ERRORLEVEL! neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
)

call .venv\Scripts\activate.bat
echo   - Virtual environment active.

:: ---------------------------------------------------------------------
:: 3. Check Python Dependencies
:: ---------------------------------------------------------------------
echo.
echo [3/4] Verifying required libraries (OpenCV, EasyOCR, PyTorch)...
python -c "import cv2, easyocr, requests, rapidfuzz, pypdfium2" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo.
    echo [NOTICE] Required libraries are not yet installed in .venv.
    set /p "INSTALL_PKGS=Do you want to install dependencies from requirements.txt now? (Y/N): "
    if /i "!INSTALL_PKGS!"=="Y" (
        echo Upgrading pip...
        python -m pip install --upgrade pip
        echo Installing dependencies (this may take a few minutes on first run)...
        pip install -r requirements.txt
        if !ERRORLEVEL! neq 0 (
            echo [ERROR] Failed to install some dependencies.
            pause
            exit /b 1
        )
    ) else (
        echo [NOTICE] Skipping dependency installation. Some features may not work.
    )
) else (
    echo   - Core dependencies are installed and ready.
)

:: ---------------------------------------------------------------------
:: 4. Check Ollama & Local Vision Models (Optional for Deep VLM)
:: ---------------------------------------------------------------------
echo.
echo [4/4] Checking Ollama local AI runtime...
where ollama >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo   - Ollama is not installed on this system.
    echo   - Classical OCR + QR extraction will work normally.
    echo   - For deep Vision-Language reasoning (Qwen2.5-VL), install Ollama from https://ollama.com
) else (
    echo   - Ollama command detected.
    :: Ensure Ollama service is running
    curl -s http://localhost:11434/api/tags >nul 2>&1
    if !ERRORLEVEL! neq 0 (
        echo   - Starting local Ollama background service...
        start /B "" ollama serve >nul 2>&1
        timeout /t 2 /nobreak >nul
    )
    
    :: Check if qwen2.5vl:3b is available
    ollama list 2>nul | findstr /i "qwen2.5vl" >nul 2>&1
    if !ERRORLEVEL! neq 0 (
        echo   - Primary model 'qwen2.5vl:3b' is not yet downloaded.
        set /p "PULL_QWEN=Do you want to download qwen2.5vl:3b (~3.2 GB)? (Y/N): "
        if /i "!PULL_QWEN!"=="Y" (
            echo Downloading qwen2.5vl:3b via Ollama...
            ollama pull qwen2.5vl:3b
        )
    ) else (
        echo   - Primary vision model (qwen2.5vl:3b) is downloaded and ready.
    )

    :: Check if minicpm-v is available
    ollama list 2>nul | findstr /i "minicpm-v" >nul 2>&1
    if !ERRORLEVEL! neq 0 (
        echo   - Secondary model 'minicpm-v' is not yet downloaded.
        set /p "PULL_MINI=Do you want to download minicpm-v (~5.5 GB) for dual-VLM consensus? (Y/N): "
        if /i "!PULL_MINI!"=="Y" (
            echo Downloading minicpm-v via Ollama...
            ollama pull minicpm-v
        )
    ) else (
        echo   - Secondary vision model (minicpm-v) is downloaded and ready.
    )
)

:: ---------------------------------------------------------------------
:: 5. Launch Doraemon Interactive Scanner
:: ---------------------------------------------------------------------
echo.
echo ===================================================================
echo Environment verification complete. Launching Doraemon...
echo ===================================================================

python scan.py

pause
