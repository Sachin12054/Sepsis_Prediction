@echo off
title Sepsis Prediction System - Quick Launcher
color 0A

cls
echo =========================================================================
echo                 SEPSIS PREDICTION SYSTEM
echo                     Quick Launcher
echo =========================================================================
echo.
echo Choose an option:
echo.
echo 1. Launch Main System (Full Dashboard)
echo 2. Launch Enhanced Dashboard  
echo 3. Launch ICU-Compatible Dashboard
echo 4. Create/Update Models
echo 5. Run Tests
echo 6. Open Web Interface Only
echo 7. Open Main Jupyter Notebook
echo 8. Advanced Scripts Menu
echo 9. System Diagnostics
echo 0. Exit
echo.
echo =========================================================================

set /p choice="Enter your choice (0-9): "

if "%choice%"=="1" goto MAIN_SYSTEM
if "%choice%"=="2" goto ENHANCED_DASHBOARD
if "%choice%"=="3" goto ICU_DASHBOARD
if "%choice%"=="4" goto CREATE_MODELS
if "%choice%"=="5" goto RUN_TESTS
if "%choice%"=="6" goto WEB_ONLY
if "%choice%"=="7" goto JUPYTER_NOTEBOOK
if "%choice%"=="8" goto ADVANCED_SCRIPTS
if "%choice%"=="9" goto DIAGNOSTICS
if "%choice%"=="0" goto EXIT

echo Invalid choice. Please select 0-9.
pause
goto START

:MAIN_SYSTEM
cls
echo Starting Main Dashboard System...
echo.
cd /d "%~dp0"
echo [1/3] Creating models if needed...
python src\models\create_model.py
echo.
echo [2/3] Starting backend server...
start /MIN cmd /c "python src\dashboard\dashboard_server.py"
echo.
echo [3/3] Opening web dashboard...
timeout /t 3 /nobreak >nul
start "" "web\sepsis_dashboard_live.html"
echo.
echo ✅ System launched successfully!
echo 🌐 Dashboard: http://localhost:5000
pause
goto START

:ENHANCED_DASHBOARD
cls
echo Starting Enhanced Dashboard...
cd /d "%~dp0"
start /MIN cmd /c "python src\dashboard\dashboard_server_enhanced.py"
timeout /t 3 /nobreak >nul
start "" "web\enhanced_dashboard.html"
echo ✅ Enhanced Dashboard launched!
pause
goto START

:ICU_DASHBOARD
cls
echo Starting ICU-Compatible Dashboard...
cd /d "%~dp0"
start /MIN cmd /c "python src\dashboard\dashboard_server_icu_fixed.py"
timeout /t 3 /nobreak >nul
echo ✅ ICU Dashboard launched at http://localhost:5000
pause
goto START

:CREATE_MODELS
cls
echo Creating/Updating Models...
cd /d "%~dp0"
python src\models\create_model.py
python src\models\create_real_models.py
echo ✅ Models updated!
pause
goto START

:RUN_TESTS
cls
echo Running System Tests...
cd /d "%~dp0"
python tests\test_dashboard.py
python tests\test_integration.py
echo ✅ Tests completed!
pause
goto START

:WEB_ONLY
cls
echo Opening Web Interface...
start "" "web\sepsis_dashboard_live.html"
echo ✅ Web interface opened!
pause
goto START

:JUPYTER_NOTEBOOK
cls
echo Opening Main Jupyter Notebook...
if exist "ensemble_learning_pipeline.ipynb" (
    start ensemble_learning_pipeline.ipynb
    echo ✅ Main model notebook opened!
) else (
    echo ❌ Main notebook file not found!
)
pause
goto START

:ADVANCED_SCRIPTS
cls
echo Opening Advanced Scripts Menu...
if exist "scripts\main.bat" (
    call "scripts\main.bat"
) else (
    echo ❌ Advanced scripts not found!
)
pause
goto START

:DIAGNOSTICS
cls
echo System Diagnostics...
cd /d "%~dp0"
python src\utils\check_server.py
echo.
python --version
echo.
echo Project structure check...
if exist "src\dashboard\dashboard_server.py" (
    echo ✅ Dashboard server found
) else (
    echo ❌ Dashboard server missing
)
echo ✅ Diagnostics completed!
pause
goto START

:EXIT
echo.
echo Thank you for using Sepsis Prediction System!
echo.
exit