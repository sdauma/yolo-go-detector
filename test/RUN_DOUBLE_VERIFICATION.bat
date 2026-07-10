@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ============================================
echo   Double Verification Test Suite
echo   Round 10 + Round 11 with Auto-Backup
echo ============================================
echo.

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

set RESULTS_DIR=..\results
set LOG_DIR=%SCRIPT_DIR%

for /f "delims=" %%a in ('powershell -Command "Get-Date -Format 'yyyyMMdd_HHmmss'"') do set TS=%%a
set DOUBLE_LOG=%LOG_DIR%\RUN_DOUBLE_VERIFICATION_%TS%.txt

echo [%date% %time%] START >> "%DOUBLE_LOG%"

REM ============================================
REM  Round 10: Full Test Suite
REM ============================================
echo ============================================
echo   [Round 10] Starting full test suite (4-5 hours)
echo ============================================
echo [%date% %time%] [Round 10] START >> "%DOUBLE_LOG%"

cd /d "%SCRIPT_DIR%"
call "RUN_ALL_STANDARD_TESTS.bat"

echo [%date% %time%] [Round 10] END >> "%DOUBLE_LOG%"

REM ============================================
REM  Backup Round 10 Results
REM ============================================
echo.
echo ============================================
echo   [Backup] Saving Round 10 results...
echo ============================================

if exist "%RESULTS_DIR%\go_arena_ablation_result.txt" (
    copy "%RESULTS_DIR%\go_arena_ablation_result.txt" "%RESULTS_DIR%\go_arena_ablation_result_run10.txt" >nul
    echo   [OK] go_arena_ablation_result.txt -> go_arena_ablation_result_run10.txt
    echo [%date% %time%] [Backup] go_arena_ablation_result_run10.txt saved >> "%DOUBLE_LOG%"
) else (
    echo   [WARN] go_arena_ablation_result.txt not found
    echo [%date% %time%] [Backup] WARNING: go_arena_ablation_result.txt not found >> "%DOUBLE_LOG%"
)

if exist "%RESULTS_DIR%\go_architecture_comparison.txt" (
    copy "%RESULTS_DIR%\go_architecture_comparison.txt" "%RESULTS_DIR%\go_architecture_comparison_run10.txt" >nul
    echo   [OK] go_architecture_comparison.txt -> go_architecture_comparison_run10.txt
    echo [%date% %time%] [Backup] go_architecture_comparison_run10.txt saved >> "%DOUBLE_LOG%"
) else (
    echo   [WARN] go_architecture_comparison.txt not found
    echo [%date% %time%] [Backup] WARNING: go_architecture_comparison.txt not found >> "%DOUBLE_LOG%"
)

if exist "%RESULTS_DIR%\python_arena_ablation_result.txt" (
    copy "%RESULTS_DIR%\python_arena_ablation_result.txt" "%RESULTS_DIR%\python_arena_ablation_result_run10.txt" >nul
    echo   [OK] python_arena_ablation_result.txt -> python_arena_ablation_result_run10.txt
    echo [%date% %time%] [Backup] python_arena_ablation_result_run10.txt saved >> "%DOUBLE_LOG%"
) else (
    echo   [WARN] python_arena_ablation_result.txt not found
    echo [%date% %time%] [Backup] WARNING: python_arena_ablation_result.txt not found >> "%DOUBLE_LOG%"
)

REM ============================================
REM  Round 11: Verification Run (Only 3 tests)
REM ============================================
echo.
echo ============================================
echo   [Round 11] Starting verification run (30-60 minutes)
echo   Note: Only go_architecture_benchmark, go_arena_ablation,
echo         and python_arena_ablation will run (others skipped)
echo ============================================
echo [%date% %time%] [Round 11] START >> "%DOUBLE_LOG%"

cd /d "%SCRIPT_DIR%"
call "RUN_ALL_STANDARD_TESTS.bat"

echo [%date% %time%] [Round 11] END >> "%DOUBLE_LOG%"

REM ============================================
REM  Backup Round 11 Results
REM ============================================
echo.
echo ============================================
echo   [Backup] Saving Round 11 results...
echo ============================================

if exist "%RESULTS_DIR%\go_arena_ablation_result.txt" (
    copy "%RESULTS_DIR%\go_arena_ablation_result.txt" "%RESULTS_DIR%\go_arena_ablation_result_run11.txt" >nul
    echo   [OK] go_arena_ablation_result.txt -> go_arena_ablation_result_run11.txt
    echo [%date% %time%] [Backup] go_arena_ablation_result_run11.txt saved >> "%DOUBLE_LOG%"
) else (
    echo   [WARN] go_arena_ablation_result.txt not found
    echo [%date% %time%] [Backup] WARNING: go_arena_ablation_result.txt not found >> "%DOUBLE_LOG%"
)

if exist "%RESULTS_DIR%\go_architecture_comparison.txt" (
    copy "%RESULTS_DIR%\go_architecture_comparison.txt" "%RESULTS_DIR%\go_architecture_comparison_run11.txt" >nul
    echo   [OK] go_architecture_comparison.txt -> go_architecture_comparison_run11.txt
    echo [%date% %time%] [Backup] go_architecture_comparison_run11.txt saved >> "%DOUBLE_LOG%"
) else (
    echo   [WARN] go_architecture_comparison.txt not found
    echo [%date% %time%] [Backup] WARNING: go_architecture_comparison.txt not found >> "%DOUBLE_LOG%"
)

if exist "%RESULTS_DIR%\python_arena_ablation_result.txt" (
    copy "%RESULTS_DIR%\python_arena_ablation_result.txt" "%RESULTS_DIR%\python_arena_ablation_result_run11.txt" >nul
    echo   [OK] python_arena_ablation_result.txt -> python_arena_ablation_result_run11.txt
    echo [%date% %time%] [Backup] python_arena_ablation_result_run11.txt saved >> "%DOUBLE_LOG%"
) else (
    echo   [WARN] python_arena_ablation_result.txt not found
    echo [%date% %time%] [Backup] WARNING: python_arena_ablation_result.txt not found >> "%DOUBLE_LOG%"
)

REM ============================================
REM  Compare Results
REM ============================================
echo.
echo ============================================
echo   [Compare] Running comparison...
echo ============================================
echo [%date% %time%] [Compare] START >> "%DOUBLE_LOG%"

cd /d "%SCRIPT_DIR%"
python compare_runs.py "%RESULTS_DIR%"

echo [%date% %time%] [Compare] END >> "%DOUBLE_LOG%"

REM ============================================
REM  Summary
REM ============================================
echo.
echo ============================================
echo   Double Verification Completed!
echo ============================================
echo.
echo   Round 10 results saved as:
echo     - go_arena_ablation_result_run10.txt
echo     - go_architecture_comparison_run10.txt
echo     - python_arena_ablation_result_run10.txt
echo.
echo   Round 11 results saved as:
echo     - go_arena_ablation_result_run11.txt
echo     - go_architecture_comparison_run11.txt
echo     - python_arena_ablation_result_run11.txt
echo.
echo   Log file: %DOUBLE_LOG%
echo ============================================

echo [%date% %time%] END >> "%DOUBLE_LOG%"
pause
