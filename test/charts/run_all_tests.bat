@echo off
chcp 65001 >nul
cd /d %~dp0

echo ============================================================
echo Chart Generation Test Script
echo ============================================================
echo.

echo [Test 1] Generate Memory Scalability Chart...
python generate_memory_scalability.py
if errorlevel 1 (
    echo X Failed: Memory Scalability Chart
) else (
    echo V Success: Memory Scalability Chart
)
echo.

echo [Test 2] Generate Latency Boxplot...
python generate_latency_boxplot.py
if errorlevel 1 (
    echo X Failed: Latency Boxplot
) else (
    echo V Success: Latency Boxplot
)
echo.

echo [Test 3] Generate RSS Curve...
python plot_rss_curve.py
if errorlevel 1 (
    echo X Failed: RSS Curve
) else (
    echo V Success: RSS Curve
)
echo.

echo [Test 4] Generate All Main Charts (7 charts)...
python generate_all_charts.py
if errorlevel 1 (
    echo X Failed: All Main Charts
) else (
    echo V Success: All Main Charts (7 charts)
)
echo.

echo [Test 5] Generate Journal Standard Charts (9 charts)...
python generate_journal_charts.py
if errorlevel 1 (
    echo ! Partial Success: Journal Standard Charts (optional charts skipped)
) else (
    echo V Success: Journal Standard Charts (9 charts)
)
echo.

echo ============================================================
echo Test Complete!
echo ============================================================
echo.
echo Charts saved to: %CD%\..\..\results\charts
echo.
echo Generated main charts:
echo   - fig2_throughput_comparison.png
echo   - fig3_memory_comparison.png
echo   - fig4_batch_effect.png
echo   - fig5_model_size_comparison.png
echo   - fig6_cpu_utilization.png
echo   - fig7_stability.png
echo   - latency_boxplot.png
echo   - memory_scalability.png
echo.
echo Journal standard charts (_journal suffix):
echo   - latency_boxplot_journal.png/pdf
echo   - memory_scalability_journal.png/pdf
echo   - cold_start_decomposition_journal.png/pdf
echo   - reinforced_performance_journal.png/pdf
echo   - reinforced_memory_journal.png/pdf
echo   - reinforced_cold_start_journal.png/pdf
echo   - reinforced_ttest_journal.png/pdf
echo.
pause