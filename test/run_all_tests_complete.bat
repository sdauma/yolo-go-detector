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
set TOTAL_TESTS=41
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
go run go_concurrent_stress_test.go
if %errorlevel% neq 0 (
    echo ERROR: Go Concurrent Stress Test failed!
    set /a FAIL_COUNT+=1
    set FAILED_TESTS=!FAILED_TESTS! Go_Concurrent_Stress
)
echo DONE: Go Concurrent Stress Test
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
REM Part 11a: Go Three Architectures Comparison
REM ========================================
echo ========================================
echo Part 11a: Go Three Architectures Comparison
echo ========================================
echo.

set /a CURRENT_TEST+=1
echo [%CURRENT_TEST%/%TOTAL_TESTS%] Running Go Three Architectures Comparison Test...
cd /d %PROJECT_ROOT%\examples
go run test_three_architectures.go > %RESULTS_DIR%\three_architectures_run.log 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Go Three Architectures Comparison Test failed!
    set /a FAIL_COUNT+=1
    set FAILED_TESTS=!FAILED_TESTS! Go_Three_Architectures
) else (
    set /a SUCCESS_COUNT+=1
)
echo DONE: Go Three Architectures Comparison Test
echo.

REM ========================================
REM Part 11b: Go Examples Tests
REM ========================================
echo ========================================
echo Part 11b: Go Examples Tests
echo ========================================
echo.

set /a CURRENT_TEST+=1
echo [%CURRENT_TEST%/%TOTAL_TESTS%] Running GoYOLO-Engine Complete Test...
cd /d %PROJECT_ROOT%\examples
go run test_goyolo_engine.go > %RESULTS_DIR%\goyolo_engine_complete.log 2>&1
if %errorlevel% neq 0 (
    echo ERROR: GoYOLO-Engine Complete Test failed!
    set /a FAIL_COUNT+=1
    set FAILED_TESTS=!FAILED_TESTS! GoYOLO_Engine_Complete
) else (
    set /a SUCCESS_COUNT+=1
)
echo DONE: GoYOLO-Engine Complete Test
echo.

set /a CURRENT_TEST+=1
echo [%CURRENT_TEST%/%TOTAL_TESTS%] Running Go Basic Functionality Test...
cd /d %PROJECT_ROOT%\examples
go run test_basic.go > %RESULTS_DIR%\test_basic_functionality.log 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Go Basic Functionality Test failed!
    set /a FAIL_COUNT+=1
    set FAILED_TESTS=!FAILED_TESTS! Go_Basic_Functionality
) else (
    set /a SUCCESS_COUNT+=1
)
echo DONE: Go Basic Functionality Test
echo.

set /a CURRENT_TEST+=1
echo [%CURRENT_TEST%/%TOTAL_TESTS%] Running Go Concurrent Benchmark Example...
cd /d %PROJECT_ROOT%\examples
go run benchmark_concurrent.go > %RESULTS_DIR%\benchmark_concurrent_example.log 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Go Concurrent Benchmark Example failed!
    set /a FAIL_COUNT+=1
    set FAILED_TESTS=!FAILED_TESTS! Go_Concurrent_Benchmark_Example
) else (
    set /a SUCCESS_COUNT+=1
)
echo DONE: Go Concurrent Benchmark Example
echo.

set /a CURRENT_TEST+=1
echo [%CURRENT_TEST%/%TOTAL_TESTS%] Running Go ONNX Environment Test...
cd /d %PROJECT_ROOT%\examples
go run test_onnx.go > %RESULTS_DIR%\test_onnx_environment.log 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Go ONNX Environment Test failed!
    set /a FAIL_COUNT+=1
    set FAILED_TESTS=!FAILED_TESTS! Go_ONNX_Environment
) else (
    set /a SUCCESS_COUNT+=1
)
echo DONE: Go ONNX Environment Test
echo.

set /a CURRENT_TEST+=1
echo [%CURRENT_TEST%/%TOTAL_TESTS%] Running Go Real-time Detection Demo...
cd /d %PROJECT_ROOT%\examples
go run real_time_detect.go > %RESULTS_DIR%\real_time_detection_demo.log 2>&1
if %errorlevel% neq 0 (
    echo WARNING: Go Real-time Detection Demo failed!
    set /a FAIL_COUNT+=1
    set FAILED_TESTS=!FAILED_TESTS! Go_Realtime_Detection
) else (
    set /a SUCCESS_COUNT+=1
)
echo DONE: Go Real-time Detection Demo
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
python python_concurrent_stress_test.py
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
echo   - Go Baseline: go_baseline_result.txt, go_pure_inference_result.txt
echo   - Go Reinforced: go_reinforced_result.txt, go_reinforced_small_result.txt, go_yolo11n_reinforced_result.txt
echo   - Go Thread Config: go_thread_1_result.txt to go_thread_12_result.txt
echo   - Go Cold Start: go_cold_start_result.txt, go_cold_start_decomposition_result.txt
echo   - Go Memory: go_memory_standardization_result.txt, go_memory_copy_overhead_result.txt
echo   - Go Stability: go_long_stability_result.txt
echo   - Go Session: go_session_creation_result.txt
echo   - Go Consistency: go_output_consistency_result.txt
echo   - Go Stress: go_concurrent_stress_test_result.txt
echo   - Go Diagnostic: go_performance_diagnostic_result.txt
echo   - Go Examples Logs:
echo       - three_architectures_run.log
echo       - goyolo_engine_complete.log
echo       - test_basic_functionality.log
echo       - benchmark_concurrent_example.log
echo       - test_onnx_environment.log
echo       - real_time_detection_demo.log
echo   - Python Baseline: python_baseline_result.txt, python_pure_inference_result.txt
echo   - Python Reinforced: python_reinforced_result.txt, python_reinforced_small_result.txt, python_yolo11n_reinforced_result.txt
echo   - Python Thread Config: python_thread_1_result.txt to python_thread_12_result.txt
echo   - Python Cold Start: python_cold_start_result.txt, python_cold_start_decomposition_result.txt
echo   - Python Memory: python_memory_standardization_result.txt, python_memory_copy_overhead_result.txt
echo   - Python Stability: python_long_stability_result.txt
echo   - Python Session: python_session_creation_result.txt
echo   - Python Consistency: python_output_consistency_result.txt
echo   - Python Stress: python_concurrent_stress_test_result.txt
echo.
echo Chart Files:
echo   - PDF: latency_boxplot.pdf, cold_start_comparison.pdf, thread_config_comparison.pdf, rss_curve.pdf
echo   - PNG: cold_start_factor.png, cold_start_vs_stable.png, thread_config_*.png
echo.
echo ========================================
echo Test Suite Execution Complete
echo ========================================
pause
