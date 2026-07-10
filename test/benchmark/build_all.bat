@echo off
cd /d "%~dp0"
echo ========================================
echo   Build All Go Test Executables
echo ========================================
echo.

setlocal enabledelayedexpansion
set FAILED=0
set TOTAL=0
set BUILT=0

for %%F in (paper_full_benchmark go_session_pool_ablation go_baseline_minimal go_reinforced_benchmark go_reinforced_yolo11n go_reinforced_benchmark_small go_pure_inference_benchmark go_concurrent_stress_fixed cold_start_benchmark go_memory_breakdown go_memory_copy_overhead thread_config_benchmark go_session_creation_benchmark go_output_consistency go_cpu_monitoring go_warmup_effect go_advanced_session_supplementary go_performance_diagnostic go_memory_standardization go_cold_start_decomposition go_batch_inference go_architecture_benchmark go_concurrent_architecture_comparison go_72h_stability go_arena_ablation) do (
    set /a TOTAL+=1
    echo [BUILD] %%F.exe ...
    go build -o %%F.exe %%F.go 2>&1
    if !ERRORLEVEL!==0 (set /a BUILT+=1 && echo [OK]   %%F.exe) else (set /a FAILED+=1 && echo [FAIL] %%F.exe)
    echo.
)

echo ========================================
echo   Build Summary: !BUILT!/!TOTAL! succeeded, !FAILED! failed
echo ========================================
endlocal

REM --- Build C API benchmark (MSVC) ---
echo.
echo ========================================
echo   Build C API Benchmark (MSVC)
echo ========================================
setlocal disabledelayedexpansion

REM MSVC paths
set "MSVC_BIN=d:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64"
set "MSVC_INC=d:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\include"
set "SDK_INC=C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0"
set "SDK_LIB=C:\Program Files (x86)\Windows Kits\10\Lib\10.0.26100.0"
set "ORT_INC=d:\mlz\trae_projects\1\yolo-go-detector\third_party\onnxruntime-win-x64-1.23.2\include"
set "ORT_LIB=d:\mlz\trae_projects\1\yolo-go-detector\third_party\onnxruntime-win-x64-1.23.2\lib"
set "MSVC_LIB=d:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\lib\x64"

if not exist "%MSVC_BIN%\cl.exe" goto :no_msvc
echo [BUILD] cpp_baseline_benchmark.exe ...
"%MSVC_BIN%\cl.exe" /EHsc /O2 /I"%MSVC_INC%" /I"%SDK_INC%\ucrt" /I"%SDK_INC%\um" /I"%SDK_INC%\shared" /I"%ORT_INC%" cpp_baseline_benchmark.cpp /link /LIBPATH:"%MSVC_LIB%" /LIBPATH:"%SDK_LIB%\ucrt\x64" /LIBPATH:"%SDK_LIB%\um\x64" /LIBPATH:"%ORT_LIB%" onnxruntime.lib /OUT:cpp_baseline_benchmark.exe
if not %ERRORLEVEL%==0 goto :build_fail
echo [OK]   cpp_baseline_benchmark.exe
goto :build_done
:no_msvc
echo [WARN] MSVC (cl.exe) not found, skipping C API build
if exist cpp_baseline_benchmark.exe (
    echo [INFO] cpp_baseline_benchmark.exe already exists, skipping build
) else (
    echo [WARN] cpp_baseline_benchmark.exe not found
)
goto :build_done
:build_fail
echo [FAIL] cpp_baseline_benchmark.exe
:build_done
endlocal

REM --- Build batch_verify (P3 Pure Inference Benchmark) ---
echo.
echo ========================================
echo   Build batch_verify (P3 Pure Inference)
echo ========================================
setlocal enabledelayedexpansion
set BV_DIR=%~dp0..\batch_verify
echo [BUILD] batch_verify_pure.exe ...
pushd "%BV_DIR%"
go build -o batch_verify_pure.exe . 2>&1
if !ERRORLEVEL!==0 (echo [OK]   batch_verify_pure.exe) else (echo [FAIL] batch_verify_pure.exe)
popd
endlocal