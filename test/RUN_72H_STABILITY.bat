@echo off
setlocal enabledelayedexpansion

REM ============================================================
REM   RUN_72H_STABILITY.bat
REM   72-Hour Stability Test - Go + Python
REM   Please run RUN_ALL_STANDARD_TESTS.bat first to confirm
REM   the 1-hour pre-check passes before running this script.
REM   Estimated duration: ~144 hours (Go 72h + Python 72h)
REM ============================================================

cd /d "%~dp0"

for /f "delims=" %%a in ('powershell -Command "Get-Date -Format 'yyyyMMdd_HHmmss'"') do set TS=%%a
set LOG=%~dp0run_72h_stability_%TS%.log

echo [%date% %time%] START >> "%LOG%"
echo ============================================================
echo   72-Hour Stability Test
echo   Log: run_72h_stability_%TS%.log
echo ============================================================
echo.
echo [WARNING] Please ensure:
echo   1. Power plan set to High Performance, disable hibernation
echo      and screen auto-off
echo   2. Disable Windows Update automatic restart
echo   3. Disable antivirus real-time scanning
echo   4. Ensure sufficient disk space
echo.

REM ============================================================
REM  Part 1: Go 72h Stability Test
REM ============================================================
echo.
echo ============================================================
echo   Part 1: Go 72h Stability Test
echo   Start: %date% %time%
echo ============================================================
echo.
echo [%date% %time%] Go 72h start >> "%LOG%"

cd /d "%~dp0benchmark"

REM Build
set BUILD_ERR_FILE=%TEMP%\go_build_72h_%RANDOM%.tmp
go build -o go_72h_stability.exe go_72h_stability.go >nul 2>"%BUILD_ERR_FILE%"
if not !errorlevel!==0 goto :go72h_build_fail
del "%BUILD_ERR_FILE%" 2>nul

echo   Running go_72h_stability.exe 72 ...
echo [%date% %time%] Running go_72h_stability.exe 72 >> "%LOG%"

go_72h_stability.exe 72 >> "%LOG%" 2>&1
if !errorlevel!==0 (echo [OK] Go 72h test completed! && echo [OK] Go 72h test completed! >> "%LOG%" && set GO72H_FAIL=0) else (echo [FAIL] Go 72h test failed! && echo [FAIL] Go 72h test failed! >> "%LOG%" && set GO72H_FAIL=1)
goto :go72h_done
:go72h_build_fail
echo [FATAL] go_72h_stability build failed!
echo [FATAL] go_72h_stability build failed! >> "%LOG%"
type "%BUILD_ERR_FILE%" >> "%LOG%" 2>nul
del "%BUILD_ERR_FILE%" 2>nul
cd /d "%~dp0"
pause & exit /b 1
:go72h_done

echo Go 72h done: %date% %time%
echo [%date% %time%] Go 72h end >> "%LOG%"

del go_72h_stability.exe 2>nul

cd /d "%~dp0"

REM ============================================================
REM  Part 2: Python 72h Stability Test
REM ============================================================
echo.
echo ============================================================
echo   Part 2: Python 72h Stability Test
echo   Start: %date% %time%
echo ============================================================
echo.
echo [%date% %time%] Python 72h start >> "%LOG%"

python --version >nul 2>&1
if !errorlevel!==0 goto :check_onnx
echo [FATAL] Python not found, skipping Python 72h test!
echo [FATAL] Python not found >> "%LOG%"
goto :summary

:check_onnx
python -c "import onnxruntime" >nul 2>&1
if !errorlevel!==0 goto :run_py72h
echo [FATAL] onnxruntime not installed, skipping Python 72h test!
echo [FATAL] onnxruntime not installed >> "%LOG%"
goto :summary

:run_py72h
echo   Running python_long_stability_72h.py 72 ...
echo [%date% %time%] Running python python_long_stability_72h.py 72 >> "%LOG%"

python "%~dp0python\python_long_stability_72h.py" 72 >> "%LOG%" 2>&1
if !errorlevel!==0 (echo [OK] Python 72h test completed! && echo [OK] Python 72h test completed! >> "%LOG%" && set PY72H_FAIL=0) else (echo [FAIL] Python 72h test failed! && echo [FAIL] Python 72h test failed! >> "%LOG%" && set PY72H_FAIL=1)

echo Python 72h done: %date% %time%
echo [%date% %time%] Python 72h end >> "%LOG%"

:summary

echo.
echo ============================================================
echo   72h Stability Test Completed
echo ============================================================
echo   End time: %date% %time%
echo   Log: %LOG%
echo ============================================================

echo [%date% %time%] END >> "%LOG%"

pause
exit /b 0
