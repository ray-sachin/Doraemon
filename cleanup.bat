@echo off
setlocal EnableDelayedExpansion
title Doraemon Cleanup and Uninstaller

:: Ensure script directory is current working directory
cd /d "%~dp0"

echo ===================================================================
echo                   DORAEMON CLEANUP ^& UNINSTALLER
echo                     Developed by Sachin Kumar Ray
echo ===================================================================
echo.
echo This tool helps you remove temporary files, clean Python environments,
echo or free up disk space by removing downloaded AI vision models.
echo.
echo Select an option:
echo   [1] Quick Clean        - Delete .venv, output files, and Python cache
echo   [2] Reclaim Disk Space - Remove downloaded Ollama AI models [~9 GB] + Quick Clean
echo   [3] Full Uninstall     - Remove files, delete models, and uninstall Ollama
echo   [4] Cancel and Exit
echo.

set /p "CHOICE=Enter choice [1-4, default: 4]: "
if "%CHOICE%"=="1" goto :quick_clean
if "%CHOICE%"=="2" goto :reclaim_models
if "%CHOICE%"=="3" goto :full_uninstall
goto :exit_clean

:: ---------------------------------------------------------------------
:: 1. Quick Clean
:: ---------------------------------------------------------------------
:quick_clean
echo.
echo ===================================================================
echo Running Quick Clean...
echo ===================================================================

if exist ".venv" (
    echo   - Removing virtual environment [.venv]...
    rmdir /s /q ".venv"
) else (
    echo   - No .venv folder found.
)

if exist "output" (
    echo   - Removing generated outputs [output/]...
    rmdir /s /q "output"
) else (
    echo   - No output folder found.
)

echo   - Cleaning Python __pycache__ and bytecode...
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d" 2>nul
del /s /q *.pyc >nul 2>&1

if exist "%TEMP%\OllamaSetup.exe" del "%TEMP%\OllamaSetup.exe" >nul 2>&1
if exist "%TEMP%\python_installer.exe" del "%TEMP%\python_installer.exe" >nul 2>&1

echo.
echo [DONE] Quick clean finished successfully!
goto :finish

:: ---------------------------------------------------------------------
:: 2. Reclaim Disk Space (Remove Models)
:: ---------------------------------------------------------------------
:reclaim_models
echo.
echo ===================================================================
echo Removing Downloaded AI Models [Reclaiming Disk Space]
echo ===================================================================

where ollama >nul 2>&1
if %ERRORLEVEL% neq 0 (
    if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" (
        set "PATH=%LOCALAPPDATA%\Programs\Ollama;%PATH%"
    )
)

where ollama >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [NOTICE] Ollama is not installed or not in PATH. No models to remove.
    goto :quick_clean
)

echo.
set /p "CONFIRM_RM=Are you sure you want to delete qwen2.5vl:3b and minicpm-v from Ollama? [Y/N]: "
if /i not "%CONFIRM_RM%"=="Y" (
    echo Skipping model deletion.
    goto :quick_clean
)

echo.
echo Removing model 'qwen2.5vl:3b'...
ollama rm qwen2.5vl:3b 2>nul

echo Removing model 'minicpm-v'...
ollama rm minicpm-v 2>nul

echo.
echo Current installed Ollama models:
ollama list 2>nul

goto :quick_clean

:: ---------------------------------------------------------------------
:: 3. Full Uninstall
:: ---------------------------------------------------------------------
:full_uninstall
echo.
echo ===================================================================
echo Full System Uninstallation
echo ===================================================================

:: First delete models
where ollama >nul 2>&1
if %ERRORLEVEL% neq 0 (
    if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" (
        set "PATH=%LOCALAPPDATA%\Programs\Ollama;%PATH%"
    )
)

where ollama >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo Removing downloaded Ollama models...
    ollama rm qwen2.5vl:3b 2>nul
    ollama rm minicpm-v 2>nul
)

:: Run Quick Clean
if exist ".venv" rmdir /s /q ".venv"
if exist "output" rmdir /s /q "output"
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d" 2>nul
del /s /q *.pyc >nul 2>&1

:: Offer to uninstall Ollama
echo.
set /p "UNINSTALL_OL=Do you want to completely uninstall the Ollama application? [Y/N]: "
if /i "%UNINSTALL_OL%"=="Y" (
    if exist "%LOCALAPPDATA%\Programs\Ollama\unins000.exe" (
        echo Launching Ollama uninstaller...
        start /wait "" "%LOCALAPPDATA%\Programs\Ollama\unins000.exe"
    ) else (
        where winget >nul 2>&1
        if !ERRORLEVEL! equ 0 (
            echo Uninstalling Ollama via winget...
            winget uninstall Ollama.Ollama
        )
    )

    :: Clean up remaining .ollama directory in user profile
    if exist "%USERPROFILE%\.ollama" (
        echo Cleaning Ollama cached data in user profile...
        rmdir /s /q "%USERPROFILE%\.ollama" 2>nul
    )
)

echo.
echo [DONE] Full uninstallation and cleanup completed!
goto :finish

:exit_clean
echo.
echo Cleanup canceled. No files were modified.

:finish
echo.
pause
