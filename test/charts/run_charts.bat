@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul

REM ============================================================
REM   run_charts.bat ??Generate all paper charts
REM   Prerequisite: test result files exist in ../results/
REM ============================================================

cd /d "%~dp0"

set PASS=0
set FAIL=0

echo ============================================================
echo   Paper Chart Generation
echo ============================================================
echo.

REM --- Fig1: Session Pool Architecture ---
echo [1/9] Fig1: Session Pool Architecture...
python "%~dp0generate_session_pool_arch.py"
if errorlevel 1 (
    echo   [WARN] Fig1 failed
    set /a FAIL+=1
) else (
    echo   [OK] Fig1 done
    set /a PASS+=1
)

REM --- Fig2-Fig7: Main charts ---
echo [2/9] Fig2-Fig7: Main charts...
python "%~dp0generate_all_charts.py"
if errorlevel 1 (
    echo   [WARN] generate_all_charts failed
    set /a FAIL+=1
) else (
    echo   [OK] Fig2-Fig7 done
    set /a PASS+=1
)

REM --- Journal supplementary charts ---
echo [3/9] Journal supplementary charts...
python "%~dp0generate_journal_charts.py"
if errorlevel 1 (
    echo   [WARN] journal_charts failed
    set /a FAIL+=1
) else (
    echo   [OK] Journal supplementary charts done
    set /a PASS+=1
)

REM --- Reinforced experiment charts ---
echo [4/9] Reinforced experiment charts...
python "%~dp0generate_reinforced_charts.py"
if errorlevel 1 (
    echo   [WARN] reinforced_charts failed
    set /a FAIL+=1
) else (
    echo   [OK] Reinforced experiment charts done
    set /a PASS+=1
)

REM --- Model size comparison ---
echo [5/9] Model size comparison...
python "%~dp0generate_model_size_comparison.py"
if errorlevel 1 (
    echo   [WARN] model_size_comparison failed
    set /a FAIL+=1
) else (
    echo   [OK] Model size comparison done
    set /a PASS+=1
)

REM --- Memory scalability ---
echo [6/9] Memory scalability...
python "%~dp0generate_memory_scalability.py"
if errorlevel 1 (
    echo   [WARN] memory_scalability failed
    set /a FAIL+=1
) else (
    echo   [OK] Memory scalability done
    set /a PASS+=1
)

REM --- Latency boxplot ---
echo [7/9] Latency boxplot...
python "%~dp0generate_latency_boxplot.py"
if errorlevel 1 (
    echo   [WARN] latency_boxplot failed
    set /a FAIL+=1
) else (
    echo   [OK] Latency boxplot done
    set /a PASS+=1
)

REM --- RSS curve ---
echo [8/9] RSS curve...
python "%~dp0plot_rss_curve.py"
if errorlevel 1 (
    echo   [WARN] rss_curve failed
    set /a FAIL+=1
) else (
    echo   [OK] RSS curve done
    set /a PASS+=1
)

REM --- Reference flowchart ---
echo [9/9] Reference flowchart...
python "%~dp0generate_reference_flowchart.py"
if errorlevel 1 (
    echo   [WARN] reference_flowchart failed
    set /a FAIL+=1
) else (
    echo   [OK] Reference flowchart done
    set /a PASS+=1
)

echo.
echo ============================================================
echo   Chart Generation Complete
echo   Passed: %PASS%
echo   Failed: %FAIL%
echo ============================================================

if %FAIL% gtr 0 (
    echo.
    echo [WARNING] %FAIL% chart^(s^) failed. Check if the corresponding test result files exist in results/
)

exit /b 0

