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

REM === CLEAN_P3_ONLY mode: set to 1 to run ONLY the clean P3 1x6 + 2x6 re-runs ===
REM === (no-load external-validity confirmation; outputs to *_clean dirs; skips all other tests). ===
REM === To restore CPP_ONLY or FULL suite, set this to 0 (and adjust CPP_ONLY below). ===
set CLEAN_P3_ONLY=0
if "%CLEAN_P3_ONLY%"=="1" (
    echo [MODE] CLEAN_P3_ONLY=1: skipping all other tests; running clean P3 1x6 + 2x6 re-runs only.
    echo [%date% %time%] [MODE] CLEAN_P3_ONLY=1 >> "%LOG%"
    goto :clean_p3_section
)

REM === CPP_ONLY mode: set to 1 to run ONLY the C API arena ablation v2 (reproduce §4.7). ===
REM === Double-click this .bat to run in CPP_ONLY mode directly (no command line needed). ===
REM === To restore the FULL test suite: change to "set CPP_ONLY=0" or delete this line. ===
set CPP_ONLY=0
if "%CPP_ONLY%"=="1" (
    echo [MODE] CPP_ONLY=1: skipping environment checks; running C API arena ablation v2 only.
    echo [%date% %time%] [MODE] CPP_ONLY=1 >> "%LOG%"
    goto :cpp_arena_section
)

REM === PY_ARENA_ONLY mode: set to 1 to run ONLY the Python Arena ablation v2 re-run ===
REM === (v2 uses correct attribute enable_cpu_mem_arena, a SUPPLEMENTARY upgrade, ===
REM === NOT a bug fix: original run was a transparent, intentional negative control). ===
REM === Double-click this .bat to run in PY_ARENA_ONLY mode directly. ===
REM === To restore the FULL test suite: change to "set PY_ARENA_ONLY=0". ===
set PY_ARENA_ONLY=0
if "%PY_ARENA_ONLY%"=="1" (
    echo [MODE] PY_ARENA_ONLY=1: skipping environment checks; running Python Arena ablation re-run only.
    echo [%date% %time%] [MODE] PY_ARENA_ONLY=1 >> "%LOG%"
    goto :py_arena_section
)

REM === YOLO11N_THREAD_ONLY mode: set to 1 to run ONLY the YOLO11n same-thread ===
REM === Go + Python latency comparison (supports §4.3 cross-language claim). ===
REM === To run just this: set YOLO11N_THREAD_ONLY=1 (preempts REPRO_ONLY default). ===
set YOLO11N_THREAD_ONLY=1
if "%YOLO11N_THREAD_ONLY%"=="1" (
    echo [MODE] YOLO11N_THREAD_ONLY=1: skipping environment checks; running YOLO11n thread-config Go+Python only.
    echo [%date% %time%] [MODE] YOLO11N_THREAD_ONLY=1 >> "%LOG%"
    goto :yolo11n_thread_section
)

REM === REPRO_ONLY mode: set to 1 to run ONLY the Session-lifecycle production-incident ===
REM === reproduction (controlled Session-lifecycle memory drift, ORT issue #27089). ===
REM === To run just the reproduction: set REPRO_ONLY=1 AND set PY_ARENA_ONLY=0. ===
REM === 当前默认=1：双击即只跑 Session 生命周期受控复现（Go + Python）。 ===
set REPRO_ONLY=1
if "%REPRO_ONLY%"=="1" (
    echo [MODE] REPRO_ONLY=1: running Session-lifecycle reproduction Go + Python only.
    echo [%date% %time%] [MODE] REPRO_ONLY=1 >> "%LOG%"
    goto :repro_section
)

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

REM --- 2b. C API Arena / ExecMode Ablation v2 (resolves §4.7 latency anomaly) ---
:cpp_arena_section
echo.
echo ============================================
echo   [2b] C API Arena / ExecMode Ablation v2
echo ============================================
echo.
echo [INFO] Sweeps inter_op x intra_op x arena x exec to locate root cause of C-API latency anomaly
echo [%date% %time%] [C_API_ABLATION_V2] Starting >> "%LOG%"

pushd %BENCHMARK_DIR%
echo [INFO] Working directory: %CD%

if exist "..\..\results\cpp_arena_ablation_v2_result.json" goto :cpp_arena_skip
call build_cpp_arena.bat
if not exist cpp_arena_ablation_benchmark.exe goto :cpp_arena_no_exe
echo [RUN] cpp_arena_ablation_benchmark.exe yolo11x yolo11x.onnx 1000
cpp_arena_ablation_benchmark.exe yolo11x yolo11x.onnx 1000
if not exist "..\..\results\cpp_arena_ablation_v2_result.json" (
    echo   [WARN] C API arena ablation v2 failed (result file not found)
    echo [WARN] C API arena ablation v2 failed >> "%LOG%"
) else (
    echo   [OK] C API arena ablation v2 completed
    echo [OK] C API arena ablation v2 completed >> "%LOG%"
)
goto :cpp_arena_done
:cpp_arena_no_exe
echo [SKIP] cpp_arena_ablation_benchmark.exe not found, skipping
echo [SKIP] cpp_arena_ablation_benchmark.exe not found >> "%LOG%"
goto :cpp_arena_done
:cpp_arena_skip
echo [SKIP] C API arena ablation v2 result already exists, skipping
echo [SKIP] C API arena ablation v2 result already exists >> "%LOG%"
:cpp_arena_done

popd

REM In CPP_ONLY mode, finish after the C API arena ablation v2 (skip Go/Python/P3/charts)
if "%CPP_ONLY%"=="1" goto :all_done

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

REM --- 10. Fixed-Concurrency Arena Ablation (P0 Confounded Variable Control) ---
echo.
echo ============================================
echo   [10] Fixed-Concurrency Arena Ablation (P0 Control)
echo ============================================
echo.

pushd %BENCHMARK_DIR%
echo [INFO] Working directory: %CD%
echo.

if exist "..\..\results\go_arena_ablation_fixed_concurrency_result.txt" goto :fc_arena_skip
echo [BUILD] go build -o go_arena_ablation_fixed_concurrency.exe go_arena_ablation_fixed_concurrency.go
go build -o go_arena_ablation_fixed_concurrency.exe go_arena_ablation_fixed_concurrency.go
if not !errorlevel!==0 goto :fc_arena_build_fail
echo [RUN] go_arena_ablation_fixed_concurrency.exe
echo [%date% %time%] [FC_ARENA] Starting >> "%LOG%"
go_arena_ablation_fixed_concurrency.exe
if !errorlevel!==0 (echo   [OK] Fixed-concurrency arena ablation completed && echo [OK] Fixed-concurrency arena ablation completed >> "%LOG%") else (echo   [WARN] Fixed-concurrency arena ablation failed (ExitCode: !errorlevel!) && echo [WARN] Fixed-concurrency arena ablation failed >> "%LOG%")
goto :fc_arena_done
:fc_arena_skip
echo [SKIP] Fixed-concurrency arena ablation result already exists, skipping
echo [SKIP] Fixed-concurrency arena ablation result already exists >> "%LOG%"
goto :fc_arena_done
:fc_arena_build_fail
echo [ERROR] Fixed-concurrency arena ablation build failed!
echo [ERROR] Fixed-concurrency arena ablation build failed >> "%LOG%"
:fc_arena_done

popd

REM --- 11. Supplementary Ablation (Table8 4x8 + Table7 intra_op=12) ---
echo.
echo ============================================
echo   [11] Supplementary Ablation (Table8 4x8 + Table7 intra12)
echo ============================================
echo.

pushd %BENCHMARK_DIR%
echo [INFO] Working directory: %CD%
echo.

if exist "..\..\results\go_ablation_4x8_supplement.json" if exist "..\..\results\go_thread_config_12_supplement.txt" goto :suppl_skip
echo [BUILD] go build -o go_supplementary_ablation.exe go_supplementary_ablation.go
go build -o go_supplementary_ablation.exe go_supplementary_ablation.go
if not !errorlevel!==0 goto :suppl_build_fail
echo [RUN] go_supplementary_ablation.exe
echo [%date% %time%] [SUPPL] Starting >> "%LOG%"
go_supplementary_ablation.exe
if !errorlevel!==0 (echo   [OK] Supplementary ablation completed && echo [OK] Supplementary ablation completed >> "%LOG%") else (echo   [WARN] Supplementary ablation failed (ExitCode: !errorlevel!) && echo [WARN] Supplementary ablation failed >> "%LOG%")
goto :suppl_done
:suppl_skip
echo [SKIP] Supplementary ablation results already exist, skipping
echo [SKIP] Supplementary ablation results already exist >> "%LOG%"
goto :suppl_done
:suppl_build_fail
echo [ERROR] Supplementary ablation build failed!
echo [ERROR] Supplementary ablation build failed >> "%LOG%"
:suppl_done

popd

REM --- 12. YOLO11n Same-Thread Latency Comparison (Go + Python) ---
:yolo11n_thread_section
echo.
echo ============================================
echo   [12] YOLO11n Same-Thread Latency Comparison (Go + Python)
echo ============================================
echo.

REM === 12a. Go YOLO11n thread-config benchmark ===
pushd %BENCHMARK_DIR%
echo [INFO] Working directory: %CD%
echo.
if exist "..\..\results\go_thread_config_yolo11n_comprehensive.txt" goto :yolo11n_go_skip
echo [BUILD] go build -o go_thread_config_yolo11n.exe thread_config_benchmark_yolo11n.go
go build -o go_thread_config_yolo11n.exe thread_config_benchmark_yolo11n.go
if not !errorlevel!==0 goto :yolo11n_go_build_fail
echo [RUN] go_thread_config_yolo11n.exe
echo [%date% %time%] [YOLO11N_GO] Starting >> "%LOG%"
go_thread_config_yolo11n.exe
if !errorlevel!==0 (echo   [OK] YOLO11n Go thread-config completed && echo [OK] YOLO11n Go thread-config completed >> "%LOG%") else (echo   [WARN] YOLO11n Go thread-config failed (ExitCode: !errorlevel!) && echo [WARN] YOLO11n Go thread-config failed >> "%LOG%")
goto :yolo11n_go_done
:yolo11n_go_skip
echo [SKIP] YOLO11n Go thread-config result already exists, skipping
echo [SKIP] YOLO11n Go thread-config result already exists >> "%LOG%"
goto :yolo11n_go_done
:yolo11n_go_build_fail
echo [ERROR] YOLO11n Go thread-config build failed!
echo [ERROR] YOLO11n Go thread-config build failed! >> "%LOG%"
:yolo11n_go_done
popd

REM === 12b. Python YOLO11n thread-config benchmark ===
pushd %PYTHON_DIR%
echo [INFO] Working directory: %CD%
echo.
python -c "import onnxruntime" >nul 2>&1
if not !errorlevel!==0 (
    echo [SKIP] onnxruntime not installed, skipping Python YOLO11n thread-config
    echo [SKIP] onnxruntime not installed >> "%LOG%"
    goto :yolo11n_py_done
)
if exist "..\..\results\python_thread_config_yolo11n_comprehensive.txt" goto :yolo11n_py_skip
echo [RUN] python python_thread_config_yolo11n_benchmark.py
echo [%date% %time%] [YOLO11N_PY] Starting >> "%LOG%"
python python_thread_config_yolo11n_benchmark.py
if !errorlevel!==0 (echo   [OK] YOLO11n Python thread-config completed && echo [OK] YOLO11n Python thread-config completed >> "%LOG%") else (echo   [WARN] YOLO11n Python thread-config failed (ExitCode: !errorlevel!) && echo [WARN] YOLO11n Python thread-config failed >> "%LOG%")
goto :yolo11n_py_done
:yolo11n_py_skip
echo [SKIP] YOLO11n Python thread-config result already exists, skipping
echo [SKIP] YOLO11n Python thread-config result already exists >> "%LOG%"
:yolo11n_py_done
popd

REM In YOLO11N_THREAD_ONLY mode, finish after this comparison (skip P3/charts)
if "%YOLO11N_THREAD_ONLY%"=="1" goto :all_done

echo.

REM --- Clean P3 Re-run (no-load external validity: paired 1x6 + 2x6) ---
:clean_p3_section
echo.
echo ============================================
echo   [CLEAN] P3 Clean Re-run: pool=1/2, intraOp=6 (no-load)
echo ============================================
echo.

pushd %TEST_DIR%batch_verify
echo [INFO] Working directory: %CD%
echo.

set "P3_IMAGE_DIR=D:\mlz\go\src\oracal\downloads\20260222"

echo [BUILD] go build -o batch_verify_pure.exe .
go build -o batch_verify_pure.exe .
if not !errorlevel!==0 (
    echo [ERROR] batch_verify_pure.exe build failed!
    echo [ERROR] batch_verify_pure.exe build failed! >> "%LOG%"
    popd
    goto :all_done
)

REM (a) pool=1 / intraOp=6 -- 与 §4.1 单Session 约6线程同配置 (clean)
if exist "output_pure_1x6_clean\stats.json" goto :clean_1x6_skip
echo [RUN] (a) pool=1 / intraOp=6 -- full 39775 images (clean, est. ~8.5h)
echo [%date% %time%] [CLEAN_P3_1x6] Starting >> "%LOG%"
batch_verify_pure.exe -dir "%P3_IMAGE_DIR%" -pool 1 -intraop 6 -workers 2 -segment 500 -limit 0 -alert-sample 0 -out ./output_pure_1x6_clean
if !errorlevel!==0 (echo   [OK] CLEAN P3 1x6 completed && echo [OK] CLEAN P3 1x6 completed >> "%LOG%") else (echo   [WARN] CLEAN P3 1x6 failed && echo [WARN] CLEAN P3 1x6 failed >> "%LOG%")
goto :clean_1x6_done
:clean_1x6_skip
echo [SKIP] CLEAN P3 1x6 already exists, skipping
echo [SKIP] CLEAN P3 1x6 already exists >> "%LOG%"
:clean_1x6_done

REM (b) pool=2 / intraOp=6 -- 生产部署真实配置 (clean)
if exist "output_pure_2x6_clean\stats.json" goto :clean_2x6_skip
echo [RUN] (b) pool=2 / intraOp=6 -- full 39775 images (clean, est. ~10.5h)
echo [%date% %time%] [CLEAN_P3_2x6] Starting >> "%LOG%"
batch_verify_pure.exe -dir "%P3_IMAGE_DIR%" -pool 2 -intraop 6 -workers 4 -segment 500 -limit 0 -alert-sample 0 -out ./output_pure_2x6_clean
if !errorlevel!==0 (echo   [OK] CLEAN P3 2x6 completed && echo [OK] CLEAN P3 2x6 completed >> "%LOG%") else (echo   [WARN] CLEAN P3 2x6 failed && echo [WARN] CLEAN P3 2x6 failed >> "%LOG%")
goto :clean_2x6_done
:clean_2x6_skip
echo [SKIP] CLEAN P3 2x6 already exists, skipping
echo [SKIP] CLEAN P3 2x6 already exists >> "%LOG%"
:clean_2x6_done

popd

REM --- 12. Python Arena Ablation v2 Re-run (supplementary; correct attribute enable_cpu_mem_arena) ---
:py_arena_section
echo.
echo ============================================
echo   [12] Python Arena Ablation v2 Re-run (supplementary; correct attribute)
echo ============================================
echo.

pushd %PYTHON_DIR%
echo [INFO] Working directory: %CD%
echo.

python -c "import onnxruntime" >nul 2>&1
if not !errorlevel!==0 (
    echo [SKIP] onnxruntime not installed, skipping Python Arena ablation
    echo [SKIP] onnxruntime not installed >> "%LOG%"
    goto :py_arena_done
)

if exist "..\..\results\python_arena_ablation_v3_result.txt" goto :py_arena_skip
echo [RUN] python python_arena_ablation.py
echo [%date% %time%] [PY_ARENA] Starting >> "%LOG%"
python python_arena_ablation.py
if !errorlevel!==0 (echo   [OK] Python Arena ablation completed && echo [OK] Python Arena ablation completed >> "%LOG%") else (echo   [WARN] Python Arena ablation failed (ExitCode: !errorlevel!) && echo [WARN] Python Arena ablation failed >> "%LOG%")
goto :py_arena_done
:py_arena_skip
echo [SKIP] Python Arena ablation result already exists, skipping
echo [SKIP] Python Arena ablation result already exists >> "%LOG%"
:py_arena_done

popd

REM --- 13. Session-Lifecycle Controlled Reproduction (ORT #27089) ---
echo.
echo ============================================
echo   [13] Session-Lifecycle Controlled Reproduction
echo ============================================
echo.

pushd %PYTHON_DIR%
echo [INFO] Working directory: %CD%
if exist "..\..\results\repro_lifecycle_python_summary.txt" goto :py_repro_skip
python python_session_lifecycle_repro.py
if !errorlevel!==0 (echo   [OK] Python lifecycle repro completed) else (echo   [WARN] Python lifecycle repro failed)
goto :py_repro_done
:py_repro_skip
echo [SKIP] Python lifecycle repro result already exists, skipping
:py_repro_done
popd

pushd %BENCHMARK_DIR%
echo [INFO] Working directory: %CD%
if exist "..\..\results\repro_lifecycle_go_summary.txt" goto :go_repro_skip
echo [BUILD] go build -o go_session_lifecycle_repro.exe go_session_lifecycle_repro.go
go build -o go_session_lifecycle_repro.exe go_session_lifecycle_repro.go
if not !errorlevel!==0 goto :go_repro_build_fail
echo [RUN] go_session_lifecycle_repro.exe
go_session_lifecycle_repro.exe
if not exist "..\..\results\repro_lifecycle_go_summary.txt" (echo   [WARN] Go lifecycle repro failed) else (echo   [OK] Go lifecycle repro completed)
goto :go_repro_done
:go_repro_build_fail
echo [ERROR] Go lifecycle repro build failed!
:go_repro_done
popd

REM --- 14. Session Pool Fault Injection (isolation + auto-rebuild) ---
echo.
echo ============================================
echo   [14] Session Pool Fault Injection (isolation + auto-rebuild)
echo ============================================
echo.

pushd %BENCHMARK_DIR%
echo [INFO] Working directory: %CD%
if exist "..\..\results\go_session_pool_fault_injection_result.txt" goto :fault_injection_skip
echo [BUILD] go build -o go_session_pool_fault_injection.exe go_session_pool_fault_injection.go
go build -o go_session_pool_fault_injection.exe go_session_pool_fault_injection.go
if not !errorlevel!==0 goto :fault_injection_build_fail
echo [RUN] go_session_pool_fault_injection.exe
go_session_pool_fault_injection.exe
if not exist "..\..\results\go_session_pool_fault_injection_result.txt" (echo   [WARN] Fault injection test failed) else (echo   [OK] Fault injection test completed)
goto :fault_injection_done
:fault_injection_build_fail
echo [ERROR] Fault injection test build failed!
:fault_injection_skip
echo [SKIP] Fault injection result already exists, skipping
:fault_injection_done
popd

goto :all_done

REM --- Dedicated mode: Session-Lifecycle Reproduction only ---
:repro_section
echo.
echo ============================================
echo   [REPRO x5] Session-Lifecycle Controlled Reproduction (Go + Python)
echo   Runs 5 independent rounds automatically (no manual intervention).
echo   IMPORTANT: each round takes a few minutes. DO NOT press Ctrl+C.
echo   Each round's outputs are saved as:
echo     repro_lifecycle_go_summary_1..5.txt
echo     repro_lifecycle_go_perrequest_series_1..5.csv
echo     repro_lifecycle_python_summary_1..5.txt
echo     repro_lifecycle_python_series_1..5.csv
echo   EXISTING *_N files are NEVER overwritten.
echo ============================================
echo.

set "RESULTS=%~dp0..\results"

REM --- Pre-build Go executable once (shared by all 5 rounds) ---
pushd %BENCHMARK_DIR%
echo [BUILD] go build -o go_session_lifecycle_repro.exe go_session_lifecycle_repro.go
go build -o go_session_lifecycle_repro.exe go_session_lifecycle_repro.go
if not !errorlevel!==0 (
    echo [ERROR] Go lifecycle repro build failed! Aborting x5 run.
    popd
    goto :all_done
)
popd

REM --- Archive any previous x5 run into a timestamped subfolder so a fresh run
REM --- starts clean. Old outputs are MOVED (never deleted, never overwritten). ---
set "ARCHIVE=%RESULTS%\_archive_%TS%"
if exist "%RESULTS%\repro_lifecycle_go_summary_1.txt" (
    if not exist "%ARCHIVE%" mkdir "%ARCHIVE%"
    for %%f in (
        "%RESULTS%\repro_lifecycle_*_1.txt"
        "%RESULTS%\repro_lifecycle_*_2.txt"
        "%RESULTS%\repro_lifecycle_*_3.txt"
        "%RESULTS%\repro_lifecycle_*_4.txt"
        "%RESULTS%\repro_lifecycle_*_5.txt"
        "%RESULTS%\repro_lifecycle_*_1.csv"
        "%RESULTS%\repro_lifecycle_*_2.csv"
        "%RESULTS%\repro_lifecycle_*_3.csv"
        "%RESULTS%\repro_lifecycle_*_4.csv"
        "%RESULTS%\repro_lifecycle_*_5.csv"
        "%RESULTS%\repro_lifecycle_*_prev.txt"
        "%RESULTS%\repro_lifecycle_*_prev.csv"
        "%RESULTS%\repro_lifecycle_go_summary.txt"
        "%RESULTS%\repro_lifecycle_go_perrequest_series.csv"
        "%RESULTS%\repro_lifecycle_python_summary.txt"
        "%RESULTS%\repro_lifecycle_python_series.csv"
    ) do (
        if exist "%%~f" move /y "%%~f" "%ARCHIVE%\" >nul 2>nul
    )
    echo [INFO] Archived previous repro_lifecycle results into %ARCHIVE%
)

for /l %%i in (1,1,5) do (
    echo.
    echo ============================================
    echo   [REPRO ROUND %%i / 5]  please wait, do NOT press Ctrl+C
    echo ============================================

    REM --- Python repro ---
    pushd %PYTHON_DIR%
    echo [INFO] Python repro [round %%i] working dir: %CD%
    python python_session_lifecycle_repro.py
    if not !errorlevel!==0 echo [WARN] Round %%i Python repro exited with error !errorlevel!
    popd

    REM --- Go repro (exe already built) ---
    pushd %BENCHMARK_DIR%
    echo [INFO] Go repro [round %%i] working dir: %CD%
    go_session_lifecycle_repro.exe
    if not !errorlevel!==0 echo [WARN] Round %%i Go repro exited with error !errorlevel!
    popd

    REM --- Save this round's outputs as _%%i (NEVER overwrite existing) ---
    if exist "%RESULTS%\repro_lifecycle_python_summary.txt" (
        if not exist "%RESULTS%\repro_lifecycle_python_summary_%%i.txt" (
            copy /y "%RESULTS%\repro_lifecycle_python_summary.txt" "%RESULTS%\repro_lifecycle_python_summary_%%i.txt" >nul
            if !errorlevel!==0 (echo [OK] Saved repro_lifecycle_python_summary_%%i.txt) else (echo [FAIL] copy repro_lifecycle_python_summary_%%i.txt)
        ) else (
            echo [WARN] repro_lifecycle_python_summary_%%i.txt exists, NOT overwritten
        )
    ) else (
        echo [WARN] Round %%i Python summary missing, skip save
    )

    if exist "%RESULTS%\repro_lifecycle_python_series.csv" (
        if not exist "%RESULTS%\repro_lifecycle_python_series_%%i.csv" (
            copy /y "%RESULTS%\repro_lifecycle_python_series.csv" "%RESULTS%\repro_lifecycle_python_series_%%i.csv" >nul
            if !errorlevel!==0 (echo [OK] Saved repro_lifecycle_python_series_%%i.csv) else (echo [FAIL] copy repro_lifecycle_python_series_%%i.csv)
        ) else (
            echo [WARN] repro_lifecycle_python_series_%%i.csv exists, NOT overwritten
        )
    ) else (
        echo [WARN] Round %%i Python series missing, skip save
    )

    if exist "%RESULTS%\repro_lifecycle_go_summary.txt" (
        if not exist "%RESULTS%\repro_lifecycle_go_summary_%%i.txt" (
            copy /y "%RESULTS%\repro_lifecycle_go_summary.txt" "%RESULTS%\repro_lifecycle_go_summary_%%i.txt" >nul
            if !errorlevel!==0 (echo [OK] Saved repro_lifecycle_go_summary_%%i.txt) else (echo [FAIL] copy repro_lifecycle_go_summary_%%i.txt)
        ) else (
            echo [WARN] repro_lifecycle_go_summary_%%i.txt exists, NOT overwritten
        )
    ) else (
        echo [WARN] Round %%i Go summary missing, skip save
    )

    if exist "%RESULTS%\repro_lifecycle_go_perrequest_series.csv" (
        if not exist "%RESULTS%\repro_lifecycle_go_perrequest_series_%%i.csv" (
            copy /y "%RESULTS%\repro_lifecycle_go_perrequest_series.csv" "%RESULTS%\repro_lifecycle_go_perrequest_series_%%i.csv" >nul
            if !errorlevel!==0 (echo [OK] Saved repro_lifecycle_go_perrequest_series_%%i.csv) else (echo [FAIL] copy repro_lifecycle_go_perrequest_series_%%i.csv)
        ) else (
            echo [WARN] repro_lifecycle_go_perrequest_series_%%i.csv exists, NOT overwritten
        )
    ) else (
        echo [WARN] Round %%i Go series missing, skip save
    )
)

goto :all_done

:all_done
echo ============================================
echo   All tests and chart generation completed.
echo ============================================
echo   Log file: %LOG%
echo ============================================

echo [%date% %time%] END >> "%LOG%"
echo.
echo [DONE] Press any key to close this window...
pause >nul
exit /b 0