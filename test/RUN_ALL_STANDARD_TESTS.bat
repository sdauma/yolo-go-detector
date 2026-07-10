@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ============================================
echo   YOLO-Go-Detector Full Standard Test Suite
echo ============================================
echo.

set TEST_DIR=%~dp0
set BENCHMARK_DIR=%~dp0benchmark
set CHARTS_DIR=%~dp0charts
set PYTHON_DIR=%~dp0python

for /f "delims=" %%a in ('powershell -Command "Get-Date -Format 'yyyyMMdd_HHmmss'"') do set TS=%%a
set LOG=%~dp0run_standard_tests_%TS%.log

echo [%date% %time%] START >> "%LOG%"

echo [CHECK] Environment check...
echo [CHECK] Environment check >> "%LOG%"

go version >nul 2>&1
if not !errorlevel!==0 goto :fatal_no_go
goto :check_python
:fatal_no_go
echo [FATAL] Go not found!
echo [FATAL] Go not found! >> "%LOG%"
pause & exit /b 1

:check_python
python --version >nul 2>&1
if not !errorlevel!==0 goto :fatal_no_python
goto :check_onnxruntime
:fatal_no_python
echo [FATAL] Python not found!
echo [FATAL] Python not found! >> "%LOG%"
pause & exit /b 1

:check_onnxruntime
python -c "import onnxruntime" >nul 2>&1
if !errorlevel!==0 (set SKIP_PYTHON=0) else (set SKIP_PYTHON=1 && echo [WARN] onnxruntime not installed. Python tests will be skipped. && echo [WARN] onnxruntime not installed >> "%LOG%")

echo [CHECK] Environment check completed >> "%LOG%"
echo.

REM --- 1. Go Tests ---
if not exist "%TEST_DIR%..\results\paper_full_benchmark_ablation.json" goto :go_tests_run
echo [SKIP] Go standard tests already completed, skipping
echo [SKIP] Go standard tests already completed >> "%LOG%"
goto :go_tests_done
:go_tests_run

echo ============================================
echo   [1/6] Go Standard Tests (Estimated 100-150 minutes)
echo ============================================
echo.

pushd %BENCHMARK_DIR%
echo [INFO] Working directory: %CD%
echo.

powershell -ExecutionPolicy Bypass -File "%~dp0run_go_tests.ps1" "%LOG%"

popd
:go_tests_done

REM --- 2. C API Benchmark Tests ---
echo.
echo ============================================
echo   [2/6] C API Benchmark Tests
echo ============================================
echo.
echo [INFO] C API benchmark test using MSVC-compiled cpp_baseline_benchmark.exe
echo [%date% %time%] [C_API] Starting >> "%LOG%"

pushd %BENCHMARK_DIR%
echo [INFO] Working directory: %CD%

if not exist cpp_baseline_benchmark.exe goto :c_api_no_exe
if exist "..\..\results\cpp_baseline_result.json" goto :c_api_skip
goto :c_api_run
:c_api_skip
echo [SKIP] C API benchmark result already exists, skipping
echo [SKIP] C API benchmark result already exists >> "%LOG%"
goto :c_api_done
:c_api_no_exe
echo [SKIP] cpp_baseline_benchmark.exe not found, skipping C API test
echo [SKIP] cpp_baseline_benchmark.exe not found >> "%LOG%"
goto :c_api_done
:c_api_run
echo [RUN] cpp_baseline_benchmark.exe yolo11x yolo11x.onnx 2000
echo [RUN] cpp_baseline_benchmark.exe >> "%LOG%"
cpp_baseline_benchmark.exe yolo11x yolo11x.onnx 2000
if not exist "..\..\results\cpp_baseline_result.json" (
    echo   [WARN] C API benchmark failed (result file not found)
    echo [WARN] C API benchmark failed >> "%LOG%"
) else (
    echo   [OK] C API benchmark completed
    echo [OK] C API benchmark completed >> "%LOG%"
)
:c_api_done

popd

REM --- 3. Python Tests ---
if "%SKIP_PYTHON%"=="1" goto :skip_python
if not exist "%TEST_DIR%..\results\python_session_pool_ablation.json" goto :python_tests_run
echo [SKIP] Python standard tests already completed, skipping
echo [SKIP] Python standard tests already completed >> "%LOG%"
goto :skip_python
:python_tests_run

echo.
echo ============================================
echo   [3/6] Python Standard Tests (Estimated 60-90 minutes)
echo ============================================
echo.

pushd %PYTHON_DIR%
echo [INFO] Working directory: %CD%
echo.

powershell -ExecutionPolicy Bypass -File "%~dp0run_python_tests.ps1"

popd

:skip_python

REM --- 4. Batch Verify Test ---
echo.
echo ============================================
echo   [4/6] Batch Verify (Production End-to-End)
echo ============================================
echo.

pushd %TEST_DIR%batch_verify
echo [INFO] Working directory: %CD%
echo.

set "IMAGE_DIR=D:\mlz\go\src\oracal\downloads\20260222"

if exist "..\..\output\detections.jsonl" goto :batch_verify_skip
goto :batch_verify_run
:batch_verify_skip
echo [SKIP] Batch verify result already exists, skipping
echo [SKIP] Batch verify result already exists >> "%LOG%"
goto :batch_verify_done
:batch_verify_run
echo [BUILD] go build -o batch_verify.exe .
if not !errorlevel!==0 goto :batch_verify_build_fail
echo [RUN] batch_verify.exe -dir "%IMAGE_DIR%" -limit 1000 -model ../../third_party/yolo11x.onnx
echo [RUN] batch_verify.exe -dir "%IMAGE_DIR%" -limit 1000 -model ../../third_party/yolo11x.onnx >> "%LOG%"
batch_verify.exe -dir "%IMAGE_DIR%" -limit 1000 -model ../../third_party/yolo11x.onnx
if !errorlevel!==0 (echo   [OK] Batch verify completed && echo [OK] Batch verify completed >> "%LOG%") else (echo   [WARN] Batch verify failed (ExitCode: !errorlevel!) && echo [WARN] Batch verify failed >> "%LOG%")
goto :batch_verify_done
:batch_verify_build_fail
echo [ERROR] Batch verify build failed!
echo [ERROR] Batch verify build failed >> "%LOG%"
:batch_verify_done

popd

REM --- 5. Cross-Language Consistency Test ---
echo.
echo ============================================
echo   [5/6] Cross-Language Consistency Verify
echo ============================================
echo.

pushd %TEST_DIR%compare
echo [INFO] Working directory: %CD%
echo.

if exist "bus_go_detections.txt" goto :compare_skip
goto :compare_run
:compare_skip
echo [SKIP] Cross-language consistency result already exists, skipping
echo [SKIP] Cross-language consistency result already exists >> "%LOG%"
goto :compare_done
:compare_run
echo [BUILD] go build -o compare.exe .
go build -o compare.exe .
if not !errorlevel!==0 goto :compare_build_fail
echo [RUN] python compare.py
echo [RUN] python compare.py >> "%LOG%"
python compare.py
if !errorlevel!==0 (echo   [OK] Compare test completed && echo [OK] Compare test completed >> "%LOG%") else (echo   [WARN] Compare test failed (ExitCode: !errorlevel!) && echo [WARN] Compare test failed >> "%LOG%")
goto :compare_done
:compare_build_fail
echo [ERROR] Compare Go build failed!
echo [ERROR] Compare Go build failed >> "%LOG%"
:compare_done

popd

REM --- 6. Generate Charts ---
echo.
echo ============================================
echo   [6/6] Generate Test Charts
echo ============================================
echo.

pushd %CHARTS_DIR%
echo [INFO] Working directory: %CD%
echo.

REM R3: Regenerate charts with updated architecture comparison data
echo [RUN] Regenerating charts...
echo [RUN] Regenerating charts >> "%LOG%"

call run_charts.bat
REM charts_done

popd

REM --- 7. P3 Pure Inference Benchmark (External Validity) ---
echo.
echo ============================================
echo   [7] P3 Pure Inference Benchmark (External Validity)
echo ============================================
echo.

pushd %TEST_DIR%batch_verify
echo [INFO] Working directory: %CD%
echo.

set "P3_IMAGE_DIR=D:\mlz\go\src\oracal\downloads\20260222"

REM (a) pool=1 / intraOp=6 -- 与 4.1 单Session 约6线程同配置
if exist "output_pure_1x6\stats.json" goto :p3a_skip
echo [RUN] (a) pool=1 / intraOp=6 -- 5000 images
echo [%date% %time%] [P3_A] Starting >> "%LOG%"
batch_verify_pure.exe -dir "%P3_IMAGE_DIR%" -pool 1 -intraop 6 -workers 2 -segment 500 -limit 0 -alert-sample 0 -out ./output_pure_1x6
if !errorlevel!==0 (echo   [OK] P3-A completed && echo [OK] P3-A completed >> "%LOG%") else (echo   [WARN] P3-A failed && echo [WARN] P3-A failed >> "%LOG%")
goto :p3a_done
:p3a_skip
echo [SKIP] P3 (a) pool=1/intraOp=6 already completed, skipping
echo [SKIP] P3 (a) pool=1/intraOp=6 already completed >> "%LOG%"
:p3a_done

REM (b) pool=2 / intraOp=6 -- 生产部署真实配置
if exist "output_pure_2x6\stats.json" goto :p3b_skip
echo [RUN] (b) pool=2 / intraOp=6 -- 5000 images
echo [%date% %time%] [P3_B] Starting >> "%LOG%"
batch_verify_pure.exe -dir "%P3_IMAGE_DIR%" -pool 2 -intraop 6 -workers 4 -segment 500 -limit 0 -alert-sample 0 -out ./output_pure_2x6
if !errorlevel!==0 (echo   [OK] P3-B completed && echo [OK] P3-B completed >> "%LOG%") else (echo   [WARN] P3-B failed && echo [WARN] P3-B failed >> "%LOG%")
goto :p3b_done
:p3b_skip
echo [SKIP] P3 (b) pool=2/intraOp=6 already completed, skipping
echo [SKIP] P3 (b) pool=2/intraOp=6 already completed >> "%LOG%"
:p3b_done

popd

REM --- 8. P3 Sweep: Configuration Scan (2000 images per config) ---
echo.
echo ============================================
echo   [8] P3 Sweep: pool=1/2 x intraOp=1-6 (2000 images each)
echo ============================================
echo.

pushd %TEST_DIR%batch_verify

set "P3_IMAGE_DIR=D:\mlz\go\src\oracal\downloads\20260222"

echo --- pool=1 series (workers=2) ---
for %%I in (1 2 3 4 5 6) do (
    set "SWEEP_OUT=./output_sweep_1x%%I"
    if not exist "!SWEEP_OUT!\stats.json" (
        echo [RUN] pool=1 intraOp=%%I -- 2000 images
        echo [%date% %time%] [SWEEP_1x%%I] Starting >> "%LOG%"
        batch_verify_pure.exe -dir "%P3_IMAGE_DIR%" -pool 1 -intraop %%I -workers 2 -segment 500 -limit 2000 -alert-sample 0 -out !SWEEP_OUT!
        if !errorlevel!==0 (echo   [OK] Sweep 1x%%I completed && echo [OK] Sweep 1x%%I completed >> "%LOG%") else (echo   [WARN] Sweep 1x%%I failed && echo [WARN] Sweep 1x%%I failed >> "%LOG%")
    ) else (
        echo [SKIP] Sweep 1x%%I already completed
    )
)

echo --- pool=2 series (workers=4) ---
for %%I in (1 2 3 4 5 6) do (
    set "SWEEP_OUT=./output_sweep_2x%%I"
    if not exist "!SWEEP_OUT!\stats.json" (
        echo [RUN] pool=2 intraOp=%%I -- 2000 images
        echo [%date% %time%] [SWEEP_2x%%I] Starting >> "%LOG%"
        batch_verify_pure.exe -dir "%P3_IMAGE_DIR%" -pool 2 -intraop %%I -workers 4 -segment 500 -limit 2000 -alert-sample 0 -out !SWEEP_OUT!
        if !errorlevel!==0 (echo   [OK] Sweep 2x%%I completed && echo [OK] Sweep 2x%%I completed >> "%LOG%") else (echo   [WARN] Sweep 2x%%I failed && echo [WARN] Sweep 2x%%I failed >> "%LOG%")
    ) else (
        echo [SKIP] Sweep 2x%%I already completed
    )
)

popd

REM --- 9. P3 Full-Scale: pool=2/intraOp=3 (Recommended Config, 39775 images) ---
echo.
echo ============================================
echo   [9] P3 Full-Scale: pool=2/intraOp=3 (Recommended, 39775 images)
echo ============================================
echo.

pushd %TEST_DIR%batch_verify
echo [INFO] Working directory: %CD%
echo.

set "P3_IMAGE_DIR=D:\mlz\go\src\oracal\downloads\20260222"

if exist batch_verify_pure.exe goto :p3c_have_bin
echo [BUILD] go build -o batch_verify_pure.exe .
go build -o batch_verify_pure.exe .
if not !errorlevel!==0 goto :p3c_build_fail
:p3c_have_bin

if exist "output_pure_2x3\stats.json" goto :p3c_skip
echo [RUN] pool=2 / intraOp=3 -- full 39775 images (estimated 5-8h)
echo [%date% %time%] [P3_C] Starting >> "%LOG%"
batch_verify_pure.exe -dir "%P3_IMAGE_DIR%" -pool 2 -intraop 3 -workers 4 -segment 500 -limit 0 -alert-sample 0 -out ./output_pure_2x3
if !errorlevel!==0 (echo   [OK] P3-C completed && echo [OK] P3-C completed >> "%LOG%") else (echo   [WARN] P3-C failed && echo [WARN] P3-C failed >> "%LOG%")
goto :p3c_done
:p3c_skip
echo [SKIP] P3 (c) pool=2/intraOp=3 already completed, skipping
echo [SKIP] P3 (c) pool=2/intraOp=3 already completed >> "%LOG%"
goto :p3c_done
:p3c_build_fail
echo [ERROR] batch_verify_pure.exe build failed!
echo [ERROR] batch_verify_pure.exe build failed! >> "%LOG%"
:p3c_done

popd

echo.
echo ============================================
echo   All tests and chart generation completed.
echo ============================================
echo   Log file: %LOG%
echo ============================================

echo [%date% %time%] END >> "%LOG%"
exit /b 0