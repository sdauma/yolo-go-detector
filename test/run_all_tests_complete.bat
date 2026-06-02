@echo off
setlocal enabledelayedexpansion

echo ========================================
echo Go vs Python Performance Test Suite
echo ========================================
echo.

REM Set project root directory (batch file is in test directory)
set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..
set BENCHMARK_DIR=%SCRIPT_DIR%benchmark
set PYTHON_DIR=%SCRIPT_DIR%python
set CHARTS_DIR=%SCRIPT_DIR%charts
set RESULTS_DIR=%PROJECT_ROOT%results

echo Script Directory: %SCRIPT_DIR%
echo Project Root: %PROJECT_ROOT%
echo.

REM 初始化计数器
REM TOTAL_TESTS=46 = 41篇标准化测试(24Go+17Py) + 1数据分析(data_analysis.py) + 4图表生成
REM data_analysis.py是统计显著性分析(读结果跑t-test)，不算论文标准化测试
set TOTAL_TESTS=46
set CURRENT_TEST=0
set SUCCESS_COUNT=0
set FAIL_COUNT=0
set FAILED_TESTS=

REM ========================================
REM Part 1: Go Baseline Tests (Core)
REM ========================================
echo ========================================
echo Part 1: Go Baseline Tests (Core)
echo ========================================
echo.

set /a CURRENT_TEST+=1
echo [%CURRENT_TEST%/%TOTAL_TESTS%] Running Go Baseline Test...
cd /d %BENCHMARK_DIR%
go run go_baseline_minimal.go
if %errorlevel% neq 0 (
    echo ERROR: Go Baseline Test failed!
    set /a FAIL_COUNT+=1
    set FAILED_TESTS=!FAILED_TESTS! Go_Baseline
) else (
    set /a SUCCESS_COUNT+=1
)
echo DONE: Go Baseline Test
echo.

set /a CURRENT_TEST+=1
echo [%CURRENT_TEST%/%TOTAL_TESTS%] Running Go Pure Inference Benchmark...
cd /d %BENCHMARK_DIR%
go run go_pure_inference_benchmark.go
if %errorlevel% neq 0 (
    echo ERROR: Go Pure Inference Benchmark failed!
    set /a FAIL_COUNT+=1
    set FAILED_TESTS=!FAILED_TESTS! Go_Pure_Inference
)
echo DONE: Go Pure Inference Benchmark
echo.

set /a CURRENT_TEST+=1
echo [%CURRENT_TEST%/%TOTAL_TESTS%] Running Go AdvancedSession Supplementary Test...
cd /d %BENCHMARK_DIR%
go run go_advanced_session_supplementary.go
if %errorlevel% neq 0 (
    echo ERROR: Go AdvancedSession Supplementary Test failed!
    set /a FAIL_COUNT+=1
    set FAILED_TESTS=!FAILED_TESTS! Go_AdvancedSession
)
echo DONE: Go AdvancedSession Supplementary Test
echo.

REM ========================================
REM Part 2: Go Reinforced Tests (YOLO11x)
REM ========================================
echo ========================================
echo Part 2: Go Reinforced Tests (YOLO11x)
echo ========================================
echo.

set /a CURRENT_TEST+=1
echo [%CURRENT_TEST%/%TOTAL_TESTS%] Running Go Reinforced Benchmark (YOLO11x)...
cd /d %BENCHMARK_DIR%
go run go_reinforced_benchmark.go
if %errorlevel% neq 0 (
    echo ERROR: Go Reinforced Benchmark failed!
    set /a FAIL_COUNT+=1
    set FAILED_TESTS=!FAILED_TESTS! Go_Reinforced_YOLO11x
)
echo DONE: Go Reinforced Benchmark (YOLO11x)
echo.

REM ========================================
REM Part 3: Go Reinforced Tests (YOLO11n)
REM ========================================
echo ========================================
echo Part 3: Go Reinforced Tests (YOLO11n)
echo ========================================
echo.

set /a CURRENT_TEST+=1
echo [%CURRENT_TEST%/%TOTAL_TESTS%] Running Go Reinforced Benchmark Small (YOLO11n)...
cd /d %BENCHMARK_DIR%
go run go_reinforced_benchmark_small.go
if %errorlevel% neq 0 (
    echo ERROR: Go Reinforced Benchmark Small failed!
    set /a FAIL_COUNT+=1
    set FAILED_TESTS=!FAILED_TESTS! Go_Reinforced_Small
)
echo DONE: Go Reinforced Benchmark Small (YOLO11n)
echo.

set /a CURRENT_TEST+=1
echo [%CURRENT_TEST%/%TOTAL_TESTS%] Running Go Reinforced YOLO11n Test...
cd /d %BENCHMARK_DIR%
go run go_reinforced_yolo11n.go
if %errorlevel% neq 0 (
    echo ERROR: Go Reinforced YOLO11n Test failed!
    set /a FAIL_COUNT+=1
    set FAILED_TESTS=!FAILED_TESTS! Go_Reinforced_YOLO11n
)
echo DONE: Go Reinforced YOLO11n Test
echo.

REM ========================================
REM Part 3a: Go Architecture Benchmark
REM ========================================
echo ========================================
echo Part 3a: Go Architecture Benchmark
echo ========================================
echo.

set /a CURRENT_TEST+=1
echo [%CURRENT_TEST%/%TOTAL_TESTS%] Running Go Architecture Benchmark...
cd /d %BENCHMARK_DIR%
go run go_architecture_benchmark.go
if %errorlevel% neq 0 (
    echo ERROR: Go Architecture Benchmark failed!
    set /a FAIL_COUNT+=1
    set FAILED_TESTS=!FAILED_TESTS! Go_Architecture_Benchmark
)
echo DONE: Go Architecture Benchmark
echo.

set /a CURRENT_TEST+=1
echo [%CURRENT_TEST%/%TOTAL_TESTS%] Running Go Architecture Quick Test...
cd /d %BENCHMARK_DIR%
go run go_architecture_quick_test.go
if %errorlevel% neq 0 (
    echo ERROR: Go Architecture Quick Test failed!
    set /a FAIL_COUNT+=1
    set FAILED_TESTS=!FAILED_TESTS! Go_Architecture_Quick
)
echo DONE: Go Architecture Quick Test
echo.

REM ========================================
REM Part 4: Go Thread Config Tests
REM ========================================
echo ========================================
echo Part 4: Go Thread Config Tests
echo ========================================
echo.

set /a CURRENT_TEST+=1
echo [%CURRENT_TEST%/%TOTAL_TESTS%] Running Go Thread Config Benchmark...
cd /d %BENCHMARK_DIR%
go run thread_config_benchmark.go
if %errorlevel% neq 0 (
    echo ERROR: Go Thread Config Benchmark failed!
    set /a FAIL_COUNT+=1
    set FAILED_TESTS=!FAILED_TESTS! Go_Thread_Config
)
echo DONE: Go Thread Config Benchmark
echo.

REM ========================================
REM Part 4a: Go Batch Inference Test
REM ========================================
echo ========================================
echo Part 4a: Go Batch Inference Test
echo ========================================
echo.

set /a CURRENT_TEST+=1
echo [%CURRENT_TEST%/%TOTAL_TESTS%] Running Go Batch Inference Test...
cd /d %BENCHMARK_DIR%
go run go_batch_inference.go
if %errorlevel% neq 0 (
    echo ERROR: Go Batch Inference Test failed!
    set /a FAIL_COUNT+=1
    set FAILED_TESTS=!FAILED_TESTS! Go_Batch_Inference
)
echo DONE: Go Batch Inference Test
echo.

REM ========================================
REM Part 5: Go Cold Start Tests
REM ========================================
echo ========================================
echo Part 5: Go Cold Start Tests
echo ========================================
echo.

set /a CURRENT_TEST+=1
echo [%CURRENT_TEST%/%TOTAL_TESTS%] Running Go Cold Start Benchmark...
cd /d %BENCHMARK_DIR%
go run cold_start_benchmark.go
if %errorlevel% neq 0 (
    echo ERROR: Go Cold Start Benchmark failed!
    set /a FAIL_COUNT+=1
    set FAILED_TESTS=!FAILED_TESTS! Go_Cold_Start
)
echo DONE: Go Cold Start Benchmark
echo.

set /a CURRENT_TEST+=1
echo [%CURRENT_TEST%/%TOTAL_TESTS%] Running Go Cold Start Decomposition...
cd /d %BENCHMARK_DIR%
go run go_cold_start_decomposition.go
if %errorlevel% neq 0 (
    echo ERROR: Go Cold Start Decomposition failed!
    set /a FAIL_COUNT+=1
    set FAILED_TESTS=!FAILED_TESTS! Go_Cold_Start_Decomp
)
echo DONE: Go Cold Start Decomposition
echo.

REM ========================================
REM Part 6: Go Memory Tests
REM ========================================
echo ========================================
echo Part 6: Go Memory Tests
echo ========================================
echo.

set /a CURRENT_TEST+=1
echo [%CURRENT_TEST%/%TOTAL_TESTS%] Running Go Memory Standardization...
cd /d %BENCHMARK_DIR%
go run go_memory_standardization.go
if %errorlevel% neq 0 (
    echo ERROR: Go Memory Standardization failed!
    set /a FAIL_COUNT+=1
    set FAILED_TESTS=!FAILED_TESTS! Go_Memory_Standard
)
echo DONE: Go Memory Standardization
echo.

set /a CURRENT_TEST+=1
echo [%CURRENT_TEST%/%TOTAL_TESTS%] Running Go Memory Copy Overhead Test...
cd /d %BENCHMARK_DIR%
go run go_memory_copy_overhead.go
if %errorlevel% neq 0 (
    echo ERROR: Go Memory Copy Overhead Test failed!
    set /a FAIL_COUNT+=1
    set FAILED_TESTS=!FAILED_TESTS! Go_Memory_Copy
)
echo DONE: Go Memory Copy Overhead Test
echo.

REM ========================================
REM Part 7: Go Long Stability Tests
REM ========================================
echo ========================================
echo Part 7: Go Long Stability Tests
echo ========================================
echo.

set /a CURRENT_TEST+=1
echo [%CURRENT_TEST%/%TOTAL_TESTS%] Running Go Long Stability Test...
cd /d %BENCHMARK_DIR%
go run go_long_stability.go
if %errorlevel% neq 0 (
    echo ERROR: Go Long Stability Test failed!
    set /a FAIL_COUNT+=1
    set FAILED_TESTS=!FAILED_TESTS! Go_Long_Stability
)
echo DONE: Go Long Stability Test
echo.

REM ========================================
REM Part 7a: Go Long Stability Enhanced Test
REM ========================================
echo ========================================
echo Part 7a: Go Long Stability Enhanced Test
echo ========================================
echo.

set /a CURRENT_TEST+=1
echo [%CURRENT_TEST%/%TOTAL_TESTS%] Running Go Long Stability Enhanced Test...
cd /d %BENCHMARK_DIR%
go run go_long_stability_enhanced.go
if %errorlevel% neq 0 (
    echo ERROR: Go Long Stability Enhanced Test failed!
    set /a FAIL_COUNT+=1
    set FAILED_TESTS=!FAILED_TESTS! Go_Long_Stability_Enhanced
)
echo DONE: Go Long Stability Enhanced Test
echo.

REM ========================================
REM Part 8: Go Session Tests
REM ========================================
echo ========================================
echo Part 8: Go Session Tests
echo ========================================
echo.

set /a CURRENT_TEST+=1
echo [%CURRENT_TEST%/%TOTAL_TESTS%] Running Go Session Creation Benchmark...
cd /d %BENCHMARK_DIR%
go run go_session_creation_benchmark.go
if %errorlevel% neq 0 (
    echo ERROR: Go Session Creation Benchmark failed!
    set /a FAIL_COUNT+=1
    set FAILED_TESTS=!FAILED_TESTS! Go_Session_Creation
)
echo DONE: Go Session Creation Benchmark
echo.

REM ========================================
REM Part 9: Go Output Consistency Test
REM ========================================
echo ========================================
echo Part 9: Go Output Consistency Test
echo ========================================
echo.

set /a CURRENT_TEST+=1
echo [%CURRENT_TEST%/%TOTAL_TESTS%] Running Go Output Consistency Test...
cd /d %BENCHMARK_DIR%
go run go_output_consistency.go
if %errorlevel% neq 0 (
    echo ERROR: Go Output Consistency Test failed!
    set /a FAIL_COUNT+=1
    set FAILED_TESTS=!FAILED_TESTS! Go_Output_Consistency
)
echo DONE: Go Output Consistency Test
echo.

REM ========================================
REM Part 10: Go Concurrent Stress Test
REM ========================================
echo ========================================
echo Part 10: Go Concurrent Stress Test
echo ========================================
echo.

set /a CURRENT_TEST+=1
echo [%CURRENT_TEST%/%TOTAL_TESTS%] Running Go Concurrent Stress Test...
cd /d %BENCHMARK_DIR%
go run go_concurrent_stress_fixed.go
if %errorlevel% neq 0 (
    echo ERROR: Go Concurrent Stress Test failed!
    set /a FAIL_COUNT+=1
    set FAILED_TESTS=!FAILED_TESTS! Go_Concurrent_Stress
)
echo DONE: Go Concurrent Stress Test
echo.

REM ========================================
REM Part 10a: Go Concurrent Architecture Comparison
REM ========================================
echo ========================================
echo Part 10a: Go Concurrent Architecture Comparison
echo ========================================
echo.

set /a CURRENT_TEST+=1
echo [%CURRENT_TEST%/%TOTAL_TESTS%] Running Go Concurrent Architecture Comparison...
cd /d %BENCHMARK_DIR%
go run go_concurrent_architecture_comparison.go
if %errorlevel% neq 0 (
    echo ERROR: Go Concurrent Architecture Comparison failed!
    set /a FAIL_COUNT+=1
    set FAILED_TESTS=!FAILED_TESTS! Go_Concurrent_Arch_Comparison
)
echo DONE: Go Concurrent Architecture Comparison
echo.

REM ========================================
REM Part 11: Go Performance Diagnostic
REM ========================================
echo ========================================
echo Part 11: Go Performance Diagnostic
echo ========================================
echo.

set /a CURRENT_TEST+=1
echo [%CURRENT_TEST%/%TOTAL_TESTS%] Running Go Performance Diagnostic...
cd /d %BENCHMARK_DIR%
go run go_performance_diagnostic.go
if %errorlevel% neq 0 (
    echo ERROR: Go Performance Diagnostic failed!
    set /a FAIL_COUNT+=1
    set FAILED_TESTS=!FAILED_TESTS! Go_Performance_Diagnostic
)
echo DONE: Go Performance Diagnostic
echo.

REM ========================================
REM Part 11a: Go CPU Monitoring Test
REM ========================================
echo ========================================
echo Part 11a: Go CPU Monitoring Test
echo ========================================
echo.

set /a CURRENT_TEST+=1
echo [%CURRENT_TEST%/%TOTAL_TESTS%] Running Go CPU Monitoring...
cd /d %BENCHMARK_DIR%
go run go_cpu_monitoring.go
if %errorlevel% neq 0 (
    echo ERROR: Go CPU Monitoring failed!
    set /a FAIL_COUNT+=1
    set FAILED_TESTS=!FAILED_TESTS! Go_CPU_Monitoring
)
echo DONE: Go CPU Monitoring
echo.

REM ========================================
REM Part 11b: Go Memory Breakdown Test
REM ========================================
echo ========================================
echo Part 11b: Go Memory Breakdown Test
echo ========================================
echo.

set /a CURRENT_TEST+=1
echo [%CURRENT_TEST%/%TOTAL_TESTS%] Running Go Memory Breakdown...
cd /d %BENCHMARK_DIR%
go run go_memory_breakdown.go
if %errorlevel% neq 0 (
    echo ERROR: Go Memory Breakdown failed!
    set /a FAIL_COUNT+=1
    set FAILED_TESTS=!FAILED_TESTS! Go_Memory_Breakdown
)
echo DONE: Go Memory Breakdown
echo.

REM ========================================
REM Part 11c: Go Warmup Effect Test
REM ========================================
echo ========================================
echo Part 11c: Go Warmup Effect Test
echo ========================================
echo.

set /a CURRENT_TEST+=1
echo [%CURRENT_TEST%/%TOTAL_TESTS%] Running Go Warmup Effect...
cd /d %BENCHMARK_DIR%
go run go_warmup_effect.go
if %errorlevel% neq 0 (
    echo ERROR: Go Warmup Effect failed!
    set /a FAIL_COUNT+=1
    set FAILED_TESTS=!FAILED_TESTS! Go_Warmup_Effect
)
echo DONE: Go Warmup Effect
echo.


REM ========================================
REM Part 12: Python Baseline Tests (Core)
REM ========================================
echo ========================================
echo Part 12: Python Baseline Tests (Core)
echo ========================================
echo.

REM 检查 Python 环境
echo Checking Python environment...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found! Please install Python 3.8+.
    set /a FAIL_COUNT+=1
    set FAILED_TESTS=!FAILED_TESTS! Python_Environment_Check
    goto SKIP_PYTHON_TESTS
)

python -c "import onnxruntime" >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: onnxruntime not installed! Please run: pip install onnxruntime
    set /a FAIL_COUNT+=1
    set FAILED_TESTS=!FAILED_TESTS! ONNXRuntime_Check
    goto SKIP_PYTHON_TESTS
)
echo Python environment OK.
echo.

set /a CURRENT_TEST+=1
echo [%CURRENT_TEST%/%TOTAL_TESTS%] Running Python Baseline Test...
cd /d %PYTHON_DIR%
python python_baseline.py
if %errorlevel% neq 0 (
    echo ERROR: Python Baseline Test failed!
    set /a FAIL_COUNT+=1
    set FAILED_TESTS=!FAILED_TESTS! Python_Baseline
)
echo DONE: Python Baseline Test
echo.

set /a CURRENT_TEST+=1
echo [%CURRENT_TEST%/%TOTAL_TESTS%] Running Python Pure Inference Benchmark...
cd /d %PYTHON_DIR%
python python_pure_inference_benchmark.py
if %errorlevel% neq 0 (
    echo ERROR: Python Pure Inference Benchmark failed!
    set /a FAIL_COUNT+=1
    set FAILED_TESTS=!FAILED_TESTS! Python_Pure_Inference
)
echo DONE: Python Pure Inference Benchmark
echo.

set /a CURRENT_TEST+=1
echo [%CURRENT_TEST%/%TOTAL_TESTS%] Running Python Baseline Supplementary Test...
cd /d %PYTHON_DIR%
python python_baseline_supplementary.py
if %errorlevel% neq 0 (
    echo ERROR: Python Baseline Supplementary Test failed!
    set /a FAIL_COUNT+=1
    set FAILED_TESTS=!FAILED_TESTS! Python_AdvancedSession
)
echo DONE: Python Baseline Supplementary Test
echo.

REM ========================================
REM Part 13: Python Reinforced Tests (YOLO11x)
REM ========================================
echo ========================================
echo Part 13: Python Reinforced Tests (YOLO11x)
echo ========================================
echo.

set /a CURRENT_TEST+=1
echo [%CURRENT_TEST%/%TOTAL_TESTS%] Running Python Reinforced Benchmark (YOLO11x)...
cd /d %PYTHON_DIR%
python python_reinforced_benchmark.py
if %errorlevel% neq 0 (
    echo ERROR: Python Reinforced Benchmark failed!
    set /a FAIL_COUNT+=1
    set FAILED_TESTS=!FAILED_TESTS! Python_Reinforced_YOLO11x
)
echo DONE: Python Reinforced Benchmark (YOLO11x)
echo.

REM ========================================
REM Part 14: Python Reinforced Tests (YOLO11n)
REM ========================================
echo ========================================
echo Part 14: Python Reinforced Tests (YOLO11n)
echo ========================================
echo.

set /a CURRENT_TEST+=1
echo [%CURRENT_TEST%/%TOTAL_TESTS%] Running Python Reinforced Benchmark Small (YOLO11n)...
cd /d %PYTHON_DIR%
python python_reinforced_benchmark_small.py
if %errorlevel% neq 0 (
    echo ERROR: Python Reinforced Benchmark Small failed!
    set /a FAIL_COUNT+=1
    set FAILED_TESTS=!FAILED_TESTS! Python_Reinforced_Small
)
echo DONE: Python Reinforced Benchmark Small (YOLO11n)
echo.

set /a CURRENT_TEST+=1
echo [%CURRENT_TEST%/%TOTAL_TESTS%] Running Python Reinforced YOLO11n Test...
cd /d %PYTHON_DIR%
python python_reinforced_yolo11n.py
if %errorlevel% neq 0 (
    echo ERROR: Python Reinforced YOLO11n Test failed!
    set /a FAIL_COUNT+=1
    set FAILED_TESTS=!FAILED_TESTS! Python_Reinforced_YOLO11n
)
echo DONE: Python Reinforced YOLO11n Test
echo.

REM ========================================
REM Part 14a: Python Architecture Benchmark
REM ========================================
echo ========================================
echo Part 14a: Python Architecture Benchmark
echo ========================================
echo.

set /a CURRENT_TEST+=1
echo [%CURRENT_TEST%/%TOTAL_TESTS%] Running Python Architecture Benchmark...
cd /d %PYTHON_DIR%
python python_architecture_benchmark.py
if %errorlevel% neq 0 (
    echo ERROR: Python Architecture Benchmark failed!
    set /a FAIL_COUNT+=1
    set FAILED_TESTS=!FAILED_TESTS! Python_Architecture_Benchmark
)
echo DONE: Python Architecture Benchmark
echo.

REM ========================================
REM Part 14b: Python CPU Monitoring
REM ========================================
echo ========================================
echo Part 14b: Python CPU Monitoring
echo ========================================
echo.

set /a CURRENT_TEST+=1
echo [%CURRENT_TEST%/%TOTAL_TESTS%] Running Python CPU Monitoring...
cd /d %PYTHON_DIR%
python python_cpu_monitoring.py
if %errorlevel% neq 0 (
    echo ERROR: Python CPU Monitoring failed!
    set /a FAIL_COUNT+=1
    set FAILED_TESTS=!FAILED_TESTS! Python_CPU_Monitoring
)
echo DONE: Python CPU Monitoring
echo.

REM ========================================
REM Part 15: Python Thread Config Tests
REM ========================================
echo ========================================
echo Part 15: Python Thread Config Tests
echo ========================================
echo.

set /a CURRENT_TEST+=1
echo [%CURRENT_TEST%/%TOTAL_TESTS%] Running Python Thread Config Benchmark...
cd /d %PYTHON_DIR%
python python_thread_config_benchmark.py
if %errorlevel% neq 0 (
    echo ERROR: Python Thread Config Benchmark failed!
    set /a FAIL_COUNT+=1
    set FAILED_TESTS=!FAILED_TESTS! Python_Thread_Config
)
echo DONE: Python Thread Config Benchmark
echo.

REM ========================================
REM Part 16: Python Cold Start Tests
REM ========================================
echo ========================================
echo Part 16: Python Cold Start Tests
echo ========================================
echo.

set /a CURRENT_TEST+=1
echo [%CURRENT_TEST%/%TOTAL_TESTS%] Running Python Cold Start Benchmark...
cd /d %PYTHON_DIR%
python python_cold_start_benchmark.py
if %errorlevel% neq 0 (
    echo ERROR: Python Cold Start Benchmark failed!
    set /a FAIL_COUNT+=1
    set FAILED_TESTS=!FAILED_TESTS! Python_Cold_Start
)
echo DONE: Python Cold Start Benchmark
echo.

set /a CURRENT_TEST+=1
echo [%CURRENT_TEST%/%TOTAL_TESTS%] Running Python Cold Start Decomposition...
cd /d %PYTHON_DIR%
python python_cold_start_decomposition.py
if %errorlevel% neq 0 (
    echo ERROR: Python Cold Start Decomposition failed!
    set /a FAIL_COUNT+=1
    set FAILED_TESTS=!FAILED_TESTS! Python_Cold_Start_Decomp
)
echo DONE: Python Cold Start Decomposition
echo.

REM ========================================
REM Part 17: Python Memory Tests
REM ========================================
echo ========================================
echo Part 17: Python Memory Tests
echo ========================================
echo.

set /a CURRENT_TEST+=1
echo [%CURRENT_TEST%/%TOTAL_TESTS%] Running Python Memory Standardization...
cd /d %PYTHON_DIR%
python python_memory_standardization.py
if %errorlevel% neq 0 (
    echo ERROR: Python Memory Standardization failed!
    set /a FAIL_COUNT+=1
    set FAILED_TESTS=!FAILED_TESTS! Python_Memory_Standard
)
echo DONE: Python Memory Standardization
echo.

set /a CURRENT_TEST+=1
echo [%CURRENT_TEST%/%TOTAL_TESTS%] Running Python Memory Copy Overhead Test...
cd /d %PYTHON_DIR%
python python_memory_copy_overhead.py
if %errorlevel% neq 0 (
    echo ERROR: Python Memory Copy Overhead Test failed!
    set /a FAIL_COUNT+=1
    set FAILED_TESTS=!FAILED_TESTS! Python_Memory_Copy
)
echo DONE: Python Memory Copy Overhead Test
echo.

REM ========================================
REM Part 18: Python Long Stability Tests
REM ========================================
echo ========================================
echo Part 18: Python Long Stability Tests
echo ========================================
echo.

set /a CURRENT_TEST+=1
echo [%CURRENT_TEST%/%TOTAL_TESTS%] Running Python Long Stability Test...
cd /d %PYTHON_DIR%
python python_long_stability.py
if %errorlevel% neq 0 (
    echo ERROR: Python Long Stability Test failed!
    set /a FAIL_COUNT+=1
    set FAILED_TESTS=!FAILED_TESTS! Python_Long_Stability
)
echo DONE: Python Long Stability Test
echo.

REM ========================================
REM Part 19: Python Session Tests
REM ========================================
echo ========================================
echo Part 19: Python Session Tests
echo ========================================
echo.

set /a CURRENT_TEST+=1
echo [%CURRENT_TEST%/%TOTAL_TESTS%] Running Python Session Creation Benchmark...
cd /d %PYTHON_DIR%
python python_session_creation_benchmark.py
if %errorlevel% neq 0 (
    echo ERROR: Python Session Creation Benchmark failed!
    set /a FAIL_COUNT+=1
    set FAILED_TESTS=!FAILED_TESTS! Python_Session_Creation
)
echo DONE: Python Session Creation Benchmark
echo.

REM ========================================
REM Part 20: Python Output Consistency Test
REM ========================================
echo ========================================
echo Part 20: Python Output Consistency Test
echo ========================================
echo.

set /a CURRENT_TEST+=1
echo [%CURRENT_TEST%/%TOTAL_TESTS%] Running Python Output Consistency Test...
cd /d %PYTHON_DIR%
python python_output_consistency.py
if %errorlevel% neq 0 (
    echo ERROR: Python Output Consistency Test failed!
    set /a FAIL_COUNT+=1
    set FAILED_TESTS=!FAILED_TESTS! Python_Output_Consistency
)
echo DONE: Python Output Consistency Test
echo.

REM ========================================
REM Part 21: Python Concurrent Stress Test
REM ========================================
echo ========================================
echo Part 21: Python Concurrent Stress Test
echo ========================================
echo.

set /a CURRENT_TEST+=1
echo [%CURRENT_TEST%/%TOTAL_TESTS%] Running Python Concurrent Stress Test...
cd /d %PYTHON_DIR%
python python_concurrent_stress_test_fixed.py
if %errorlevel% neq 0 (
    echo ERROR: Python Concurrent Stress Test failed!
    set /a FAIL_COUNT+=1
    set FAILED_TESTS=!FAILED_TESTS! Python_Concurrent_Stress
)
echo DONE: Python Concurrent Stress Test
echo.

REM ========================================
REM Part 22: Python Data Analysis
REM ========================================
echo ========================================
echo Part 22: Python Data Analysis
echo ========================================
echo.

set /a CURRENT_TEST+=1
echo [%CURRENT_TEST%/%TOTAL_TESTS%] Running Python Data Analysis...
cd /d %PYTHON_DIR%
python data_analysis.py
if %errorlevel% neq 0 (
    echo WARNING: Python Data Analysis failed or not available!
) else (
    echo DONE: Python Data Analysis
)
echo.

REM ========================================
REM Part 23: Chart Generation Scripts
REM ========================================
echo ========================================
echo Part 23: Chart Generation Scripts
echo ========================================
echo.

set /a CURRENT_TEST+=1
echo [%CURRENT_TEST%/%TOTAL_TESTS%] Generating Cold Start and Thread Config Charts (PDF)...
cd /d %CHARTS_DIR%
python generate_cold_start_and_thread_charts.py
if %errorlevel% neq 0 (
    echo WARNING: Cold Start and Thread Config Charts generation failed!
) else (
    echo DONE: Cold Start and Thread Config Charts (PDF)
)
echo.

set /a CURRENT_TEST+=1
echo [%CURRENT_TEST%/%TOTAL_TESTS%] Generating Latency Boxplot (PDF)...
cd /d %CHARTS_DIR%
python generate_latency_boxplot.py
if %errorlevel% neq 0 (
    echo WARNING: Latency Boxplot generation failed!
) else (
    echo DONE: Latency Boxplot (PDF)
)
echo.

set /a CURRENT_TEST+=1
echo [%CURRENT_TEST%/%TOTAL_TESTS%] Generating Long Stability Memory Curve (PDF)...
cd /d %CHARTS_DIR%
python plot_rss_curve.py
if %errorlevel% neq 0 (
    echo WARNING: Long Stability Memory Curve generation failed!
) else (
    echo DONE: Long Stability Memory Curve (PDF)
)
echo.

set /a CURRENT_TEST+=1
echo [%CURRENT_TEST%/%TOTAL_TESTS%] Generating All PNG Format Charts...
cd /d %CHARTS_DIR%
python generate_charts_png.py
if %errorlevel% neq 0 (
    echo WARNING: PNG Format Charts generation failed!
) else (
    echo DONE: PNG Format Charts
)
echo.

:SKIP_PYTHON_TESTS

REM ========================================
REM Test Summary
REM ========================================
echo ========================================
echo Test Summary
echo ========================================
echo.
echo Total Tests: %TOTAL_TESTS%
echo Successful: %SUCCESS_COUNT%
echo Failed: %FAIL_COUNT%
if not "%FAILED_TESTS%"=="" (
    echo Failed Tests: %FAILED_TESTS%
)
echo.
if %FAIL_COUNT% equ 0 (
    echo ========================================
    echo ✓ All tests passed!
    echo ========================================
) else (
    echo ========================================
    echo ✗ %FAIL_COUNT% test(s) failed!
    echo ========================================
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
echo   - Go Baseline: go_baseline_result.txt, go_pure_inference_result.txt, go_advanced_session_result.txt
echo   - Go Architecture: go_architecture_benchmark_result.txt, go_architecture_quick_result.txt
echo   - Go Reinforced: go_reinforced_result.txt, go_reinforced_small_result.txt, go_yolo11n_reinforced_result.txt
echo   - Go Thread Config: go_thread_1_result.txt to go_thread_12_result.txt
echo   - Go Batch Inference: go_batch_inference_result.txt
echo   - Go Cold Start: go_cold_start_result.txt, go_cold_start_decomposition_result.txt
echo   - Go Memory: go_memory_standardization_result.txt, go_memory_copy_overhead_result.txt, go_memory_breakdown_result.txt
echo   - Go Stability: go_long_stability_result.txt, go_long_stability_enhanced_result.txt
echo   - Go Session: go_session_creation_result.txt
echo   - Go Consistency: go_output_consistency_result.txt
echo   - Go Stress: go_concurrent_stress_result.txt, go_concurrent_architecture_comparison_result.txt
echo   - Go Diagnostic: go_performance_diagnostic_result.txt
echo   - Go CPU: go_cpu_monitoring_result.txt
echo   - Go Warmup: go_warmup_effect_result.txt
echo   - Python Baseline: python_baseline_result.txt, python_pure_inference_result.txt, python_advanced_session_result.txt
echo   - Python Architecture: python_architecture_benchmark_result.txt
echo   - Python Reinforced: python_reinforced_result.txt, python_reinforced_small_result.txt, python_yolo11n_reinforced_result.txt
echo   - Python Thread Config: python_thread_1_result.txt to python_thread_12_result.txt
echo   - Python Cold Start: python_cold_start_result.txt, python_cold_start_decomposition_result.txt
echo   - Python Memory: python_memory_standardization_result.txt, python_memory_copy_overhead_result.txt
echo   - Python Stability: python_long_stability_result.txt
echo   - Python Session: python_session_creation_result.txt
echo   - Python Consistency: python_output_consistency_result.txt
echo   - Python Stress: python_concurrent_stress_result.txt
echo   - Python CPU: python_cpu_monitoring_result.txt
echo.
echo Chart Files:
echo   - PDF: latency_boxplot.pdf, cold_start_comparison.pdf, thread_config_comparison.pdf, rss_curve.pdf
echo   - PNG: cold_start_factor.png, cold_start_vs_stable.png, thread_config_*.png
echo.
echo ========================================
echo Test Suite Execution Complete
echo ========================================
pause
