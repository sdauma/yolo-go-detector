@echo off
setlocal enabledelayedexpansion

echo ========================================
echo Go vs Python Performance Test Suite
echo ========================================
echo.

REM Set project root directory (batch file is in project root)
set PROJECT_ROOT=%~dp0
set BENCHMARK_DIR=%PROJECT_ROOT%test\benchmark
set PYTHON_DIR=%PROJECT_ROOT%test\python
set CHARTS_DIR=%PROJECT_ROOT%test\charts
set RESULTS_DIR=%PROJECT_ROOT%results

echo Project Root: %PROJECT_ROOT%
echo.

REM ========================================
REM Part 1: Go Test Programs (5)
REM ========================================
echo ========================================
echo Part 1: Go Test Programs
echo ========================================
echo.

echo [1/10] Running Go Baseline Test...
cd /d %BENCHMARK_DIR%
go run go_baseline_minimal.go
if %errorlevel% neq 0 (
    echo ERROR: Go Baseline Test failed!
    pause
    exit /b 1
)
echo DONE: Go Baseline Test
echo.

echo [2/10] Running Go Thread Config Test...
cd /d %BENCHMARK_DIR%
go run thread_config_benchmark.go
if %errorlevel% neq 0 (
    echo ERROR: Go Thread Config Test failed!
    pause
    exit /b 1
)
echo DONE: Go Thread Config Test
echo.

echo [3/10] Running Go Cold Start Test...
cd /d %BENCHMARK_DIR%
go run cold_start_benchmark.go
if %errorlevel% neq 0 (
    echo ERROR: Go Cold Start Test failed!
    pause
    exit /b 1
)
echo DONE: Go Cold Start Test
echo.

echo [4/10] Running Go Long Stability Test...
cd /d %BENCHMARK_DIR%
go run go_long_stability.go
if %errorlevel% neq 0 (
    echo ERROR: Go Long Stability Test failed!
    pause
    exit /b 1
)
echo DONE: Go Long Stability Test
echo.

echo [5/10] Running Go AdvancedSession Supplementary Test...
cd /d %BENCHMARK_DIR%
go run go_advanced_session_supplementary.go
if %errorlevel% neq 0 (
    echo ERROR: Go AdvancedSession Supplementary Test failed!
    pause
    exit /b 1
)
echo DONE: Go AdvancedSession Supplementary Test
echo.

REM ========================================
REM Part 2: Python Test Programs (5)
REM ========================================
echo ========================================
echo Part 2: Python Test Programs
echo ========================================
echo.

echo [6/10] Running Python Baseline Test...
cd /d %PYTHON_DIR%
python python_baseline.py
if %errorlevel% neq 0 (
    echo ERROR: Python Baseline Test failed!
    pause
    exit /b 1
)
echo DONE: Python Baseline Test
echo.

echo [7/10] Running Python Thread Config Test...
cd /d %PYTHON_DIR%
python python_thread_config_benchmark.py
if %errorlevel% neq 0 (
    echo ERROR: Python Thread Config Test failed!
    pause
    exit /b 1
)
echo DONE: Python Thread Config Test
echo.

echo [8/10] Running Python Cold Start Test...
cd /d %PYTHON_DIR%
python python_cold_start_benchmark.py
if %errorlevel% neq 0 (
    echo ERROR: Python Cold Start Test failed!
    pause
    exit /b 1
)
echo DONE: Python Cold Start Test
echo.

echo [9/10] Running Python Long Stability Test...
cd /d %PYTHON_DIR%
python python_long_stability.py
if %errorlevel% neq 0 (
    echo ERROR: Python Long Stability Test failed!
    pause
    exit /b 1
)
echo DONE: Python Long Stability Test
echo.

echo [10/10] Running Python Baseline Supplementary Test...
cd /d %PYTHON_DIR%
python python_baseline_supplementary.py
if %errorlevel% neq 0 (
    echo ERROR: Python Baseline Supplementary Test failed!
    pause
    exit /b 1
)
echo DONE: Python Baseline Supplementary Test
echo.

REM ========================================
REM Part 3: Chart Generation Scripts (4)
REM ========================================
echo ========================================
echo Part 3: Chart Generation Scripts
echo ========================================
echo.

echo [1/4] Generating Cold Start and Thread Config Charts (PDF)...
cd /d %CHARTS_DIR%
python generate_cold_start_and_thread_charts.py
if %errorlevel% neq 0 (
    echo WARNING: Cold Start and Thread Config Charts generation failed!
) else (
    echo DONE: Cold Start and Thread Config Charts (PDF)
)
echo.

echo [2/4] Generating Latency Boxplot (PDF)...
cd /d %CHARTS_DIR%
python generate_latency_boxplot.py
if %errorlevel% neq 0 (
    echo WARNING: Latency Boxplot generation failed!
) else (
    echo DONE: Latency Boxplot (PDF)
)
echo.

echo [3/4] Generating Long Stability Memory Curve (PDF)...
cd /d %CHARTS_DIR%
python plot_rss_curve.py
if %errorlevel% neq 0 (
    echo WARNING: Long Stability Memory Curve generation failed!
) else (
    echo DONE: Long Stability Memory Curve (PDF)
)
echo.

echo [4/4] Generating All PNG Format Charts...
cd /d %CHARTS_DIR%
python generate_charts_png.py
if %errorlevel% neq 0 (
    echo WARNING: PNG Format Charts generation failed!
) else (
    echo DONE: PNG Format Charts
)
echo.

REM ========================================
REM Test Complete
REM ========================================
echo ========================================
echo All Tests and Charts Generation Complete!
echo ========================================
echo.
echo Test Results saved in: %RESULTS_DIR%
echo.
echo Test Result Files:
echo   - Go: go_baseline_result.txt, go_thread_*.txt, go_cold_start_result.txt, go_long_stability_result.txt, go_advanced_session_supplementary.txt
echo   - Python: python_baseline_result.txt, python_thread_*.txt, python_cold_start_result.txt, python_long_stability_result.txt, python_baseline_supplementary.txt
echo   - Comprehensive: go_thread_config_comprehensive.txt, python_thread_config_comprehensive.txt
echo.
echo Chart Files:
echo   - PDF: latency_boxplot.pdf, cold_start_comparison.pdf, thread_config_comparison.pdf, rss_curve.pdf
echo   - PNG: cold_start_factor.png, cold_start_vs_stable.png, thread_config_*.png
echo.
echo ========================================
echo Test Suite Execution Complete
echo ========================================
pause
