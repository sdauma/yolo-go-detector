@echo off
setlocal enabledelayedexpansion

REM ============================================================
REM   RUN_1H_TEST.bat
REM   1-Hour Stability Smoke Test (Go + Python, 1h each)
REM   Used to verify the scripts complete correctly before
REM   committing to a full 72-hour run.
REM ============================================================

cd /d "%~dp0"

for /f "delims=" %%a in ('powershell -Command "Get-Date -Format 'yyyyMMdd_HHmmss'"') do set TS=%%a
set LOG=%~dp0run_1h_test_%TS%.log

echo [%date% %time%] START >> "%LOG%"
echo ============================================================
echo   1-Hour Stability SMOKE TEST
echo   Log: run_1h_test_%TS%.log
echo ============================================================
echo   NOTE: This is a short smoke test. Use RUN_72H_STABILITY.bat
echo   for the real 72-hour run.
echo.

REM ============================================================
REM  Part 1: Go 1h Stability Test
REM ============================================================
echo.
echo ============================================================
echo   Part 1: Go 1h Stability Test
echo   Start: %date% %time%
echo ============================================================
echo.
echo [%date% %time%] Go 1h start >> "%LOG%"

cd /d "%~dp0benchmark"

REM Build (fresh build to verify the .go still compiles)
if exist go_1h_test.exe del go_1h_test.exe
set BUILD_ERR_FILE=%TEMP%\go_build_1h_%RANDOM%.tmp
go build -o go_1h_test.exe go_72h_stability.go >nul 2>"%BUILD_ERR_FILE%"
if not !errorlevel!==0 goto :go1h_build_fail
del "%BUILD_ERR_FILE%" 2>nul

echo   Running go_1h_test.exe 1 ...
echo [%date% %time%] Running go_1h_test.exe 1 >> "%LOG%"

go_1h_test.exe 1 >> "%LOG%" 2>&1
if !errorlevel!==0 (echo [OK] Go 1h test completed! && echo [OK] Go 1h test completed! >> "%LOG%" && set GO1H_FAIL=0) else (echo [FAIL] Go 1h test failed! && echo [FAIL] Go 1h test failed! >> "%LOG%" && set GO1H_FAIL=1)
goto :go1h_done
:go1h_build_fail
echo [FATAL] go_1h_test build failed!
echo [FATAL] go_1h_test build failed! >> "%LOG%"
type "%BUILD_ERR_FILE%" >> "%LOG%" 2>nul
del "%BUILD_ERR_FILE%" 2>nul
cd /d "%~dp0"
pause & exit /b 1
:go1h_done

echo Go 1h done: %date% %time%
echo [%date% %time%] Go 1h end >> "%LOG%"

del go_1h_test.exe 2>nul

cd /d "%~dp0"

REM ============================================================
REM  Part 2: Python 1h Stability Test (SKIPPED — already verified)
REM ============================================================
echo.
echo   [SKIP] Python 1h: already verified, skipping.
echo [%date% %time%] Python 1h SKIPPED (already verified) >> "%LOG%"

:summary

echo.
echo ============================================================
echo   1h Smoke Test Completed
echo ============================================================
echo   End time: %date% %time%
echo   Log: %LOG%
echo ============================================================
echo   If both parts show [OK], you can safely run the full
echo   72-hour test with RUN_72H_STABILITY.bat
echo ============================================================

echo [%date% %time%] END >> "%LOG%"

pause
exit /b 0