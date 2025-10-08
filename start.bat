@echo off
:: =========================================================================
:: 🚀 SEPSIS PREDICTION SYSTEM - ROOT LAUNCHER
:: =========================================================================
:: Quick access launcher from root directory
:: Redirects to organized launcher system
:: =========================================================================

title Sepsis Prediction System - Root Launcher

color 0A

cls
echo.
echo =========================================================================
echo                🏥 SEPSIS PREDICTION SYSTEM
echo                        Root Quick Launcher
echo =========================================================================
echo.
echo 🔥 Main Model: ensemble_learning_pipeline.ipynb (This Directory)
echo 📁 Organized Structure: All support files properly organized
echo.
echo =========================================================================
echo.

echo 🚀 LAUNCHER OPTIONS:
echo.
echo 1. 🔥 Open Main Ensemble Model (Current Directory)
echo 2. 🌐 Launch Complete System (Web Dashboard + Backend)
echo 3. 🖥️ Full Backend System (Professional Interface)
echo 4. ⚡ Quick Launch Menu
echo 5. 🖥️ Create Desktop Shortcuts
echo 6. 📋 View Documentation
echo 7. 🚪 Exit
echo.
set /p choice="👉 Select option (1-7): "

if "%choice%"=="1" goto MAIN_MODEL
if "%choice%"=="2" goto COMPLETE_SYSTEM
if "%choice%"=="3" goto FULL_BACKEND
if "%choice%"=="4" goto QUICK_LAUNCH
if "%choice%"=="5" goto SHORTCUTS
if "%choice%"=="6" goto DOCS
if "%choice%"=="7" goto EXIT
echo ❌ Invalid choice. Please select 1-7.
pause
goto START

:MAIN_MODEL
echo.
echo 🔥 Opening Main Ensemble Model...
if exist "ensemble_learning_pipeline.ipynb" (
    start ensemble_learning_pipeline.ipynb
    echo ✅ Main model opened successfully
) else (
    echo ❌ Main model file not found!
)
pause
exit

:COMPLETE_SYSTEM
echo.
echo 🌐 Launching Complete System (Web Dashboard + Backend)...
if exist "launch_system.bat" (
    call "launch_system.bat"
) else (
    echo ❌ Complete system launcher not found!
    echo 🔧 Using alternative launcher...
    call "launchers\main.bat"
)
exit

:FULL_BACKEND
echo.
echo 🖥️ Launching Full Backend System...
call "launchers\main.bat"
exit

:QUICK_LAUNCH
echo.
echo ⚡ Opening Quick Launch Menu...
call "launchers\quick_launch.bat"
exit

:SHORTCUTS
echo.
echo 🖥️ Creating Desktop Shortcuts...
call "launchers\create_shortcuts.bat"
pause
exit

:DOCS
echo.
echo 📋 Opening Documentation...
if exist "docs\README.md" start "docs\README.md"
if exist "docs\PROJECT_STRUCTURE.md" start "docs\PROJECT_STRUCTURE.md"
pause
exit

:EXIT
echo.
echo 👋 Thank you for using Sepsis Prediction System!
echo 🏥 Saving lives with AI - Zero missed sepsis cases!
pause
exit