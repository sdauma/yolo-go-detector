@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul

cd /d d:\mlz\trae_projects\1\yolo-go-detector\test\batch_verify

set IMG_DIR=D:\mlz\go\src\oracal\downloads\20260222
set LIMIT=2000

echo ============================================================
echo   P3 Quick Sweep: pool=2 x intraOp=1/2/3/4/5 (%LIMIT% images each)
echo ============================================================
echo.

for %%I in (1 2 3 4 5) do (
    set OUT=./output_sweep_2x%%I
    if not exist "!OUT!\stats.json" (
        echo [RUN] pool=2 intraOp=%%I ...
        batch_verify_pure.exe -dir "%IMG_DIR%" -pool 2 -intraop %%I -workers 4 -segment 500 -limit %LIMIT% -alert-sample 0 -out !OUT!
        echo.
    ) else (
        echo [SKIP] pool=2 intraOp=%%I already done
    )
)

echo ============================================================
echo   P3 Quick Sweep: pool=1 x intraOp=1/2/3/4/5 (%LIMIT% images each)
echo ============================================================
echo.

for %%I in (1 2 3 4 5) do (
    set OUT=./output_sweep_1x%%I
    if not exist "!OUT!\stats.json" (
        echo [RUN] pool=1 intraOp=%%I ...
        batch_verify_pure.exe -dir "%IMG_DIR%" -pool 1 -intraop %%I -workers 2 -segment 500 -limit %LIMIT% -alert-sample 0 -out !OUT!
        echo.
    ) else (
        echo [SKIP] pool=1 intraOp=%%I already done
    )
)

echo ============================================================
echo   Sweep Complete! Reading results...
echo ============================================================
echo.

echo | set /p="config | avg_pure_ms | p50_pure_ms | p99_pure_ms | rss_drift_mb_per_h"
echo.
echo -----------------------------------------------------------------------
for %%P in (1 2) do (
    for %%I in (1 2 3 4 5 6) do (
        set STATS=./output_sweep_%%Px%%I/stats.json
        if exist "!STATS!" (
            for /f "tokens=*" %%A in ('python -c "import json;d=json.load(open(r'!STATS!',encoding='utf-8'));print(f'pool={d[\"pool_size\"]} x intra={d[\"intra_op\"]} | {d[\"avg_pure_infer_ms\"]:.2f} | {d[\"p50_pure_infer_ms\"]:.2f} | {d[\"p99_pure_infer_ms\"]:.2f} | {d[\"rss_drift_mb_per_hour\"]:.2f}')"') do echo %%A
        )
    )
)

echo.
echo Done. Check output_sweep_* directories for details.
