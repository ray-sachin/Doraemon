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
echo [1/2] Checking Python environment...
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
echo [2/2] Verifying required libraries...
%PY_CMD% -c "import cv2, easyocr, rapidfuzz, pypdfium2" >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo   - Core dependencies are installed and ready.
    goto :launch_scanner
)

:: Check if virtual environment exists with dependencies
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
    python -c "import cv2, easyocr, rapidfuzz, pypdfium2" >nul 2>&1
    if !ERRORLEVEL! equ 0 (
        set "PY_CMD=python"
        echo   - Virtual environment libraries ready.
        goto :launch_scanner
    )
)

echo.
echo [NOTICE] Some required libraries are missing.
set /p "INSTALL_PKGS=Do you want to install dependencies from requirements.txt now? [Y/N]: "
if /i not "%INSTALL_PKGS%"=="Y" (
    echo [NOTICE] Skipping package installation. Some features may not work.
    goto :launch_scanner
)

echo Upgrading pip...
%PY_CMD% -m pip install --upgrade pip
echo Installing libraries from requirements.txt...
%PY_CMD% -m pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo.
    echo [WARNING] Some dependencies failed to install.
)

:launch_scanner
:: ---------------------------------------------------------------------
:: 3. Launch Doraemon Scanner
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
