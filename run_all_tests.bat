@echo off

REM 运行所有测试的批处理脚本
REM 包括Go和Python的各4个测试程序，共8个测试文件

set "PROJECT_ROOT=%~dp0"
set "TEST_DIR=%PROJECT_ROOT%test"
set "RESULTS_DIR=%PROJECT_ROOT%results"
set "DATA_DIR=%TEST_DIR%\data"

REM 创建结果目录
if not exist "%RESULTS_DIR%" (
    echo 创建结果目录: %RESULTS_DIR%
    mkdir "%RESULTS_DIR%"
)

REM 创建数据目录
if not exist "%DATA_DIR%" (
    echo 创建数据目录: %DATA_DIR%
    mkdir "%DATA_DIR%"
)

REM 检查输入数据是否存在，如果不存在则生成
if not exist "%DATA_DIR%\input_data.bin" (
    echo 输入数据文件不存在，正在生成...
    cd "%TEST_DIR%"
    python generate_input_data.py
    if errorlevel 1 (
        echo 生成输入数据失败！
        pause
        exit /b 1
    )
    echo 输入数据生成完成。
    cd "%PROJECT_ROOT%"
) else (
    echo 输入数据文件已存在，跳过生成步骤。
)

REM 显示开始信息
echo 开始运行所有测试...
echo 当前目录: %PROJECT_ROOT%
echo 测试目录: %TEST_DIR%
echo 结果目录: %RESULTS_DIR%
echo.

REM 运行Go测试
echo ====================================
echo 运行 Go 测试
echo ====================================
echo.

REM 运行Go线程配置测试
echo 1. 运行 Go 线程配置测试 (thread_config_benchmark.go)
echo ====================================
cd "%TEST_DIR%\benchmark"
go run thread_config_benchmark.go
if errorlevel 1 (
    echo Go 线程配置测试失败！
    pause
    exit /b 1
)
echo Go 线程配置测试完成。
echo.

REM 运行Go冷启动测试
echo 2. 运行 Go 冷启动测试 (cold_start_benchmark.go)
echo ====================================
go run cold_start_benchmark.go
if errorlevel 1 (
    echo Go 冷启动测试失败！
    pause
    exit /b 1
)
echo Go 冷启动测试完成。
echo.

REM 运行Go基准测试
echo 3. 运行 Go 基准测试 (go_baseline_minimal.go)
echo ====================================
go run go_baseline_minimal.go
if errorlevel 1 (
    echo Go 基准测试失败！
    pause
    exit /b 1
)
echo Go 基准测试完成。
echo.

REM 运行Go长时间稳定性测试
echo 4. 运行 Go 长时间稳定性测试 (go_long_stability.go)
echo ====================================
echo 注意：此测试将运行10分钟，请耐心等待...
go run go_long_stability.go
if errorlevel 1 (
    echo Go 长时间稳定性测试失败！
    pause
    exit /b 1
)
echo Go 长时间稳定性测试完成。
echo.

REM 运行Python测试
echo ====================================
echo 运行 Python 测试
echo ====================================
echo.

REM 运行Python线程配置测试
echo 5. 运行 Python 线程配置测试 (python_thread_config_benchmark.py)
echo ====================================
cd "%TEST_DIR%\python"
python python_thread_config_benchmark.py
if errorlevel 1 (
    echo Python 线程配置测试失败！
    pause
    exit /b 1
)
echo Python 线程配置测试完成。
echo.

REM 运行Python冷启动测试
echo 6. 运行 Python 冷启动测试 (python_cold_start_benchmark.py)
echo ====================================
python python_cold_start_benchmark.py
if errorlevel 1 (
    echo Python 冷启动测试失败！
    pause
    exit /b 1
)
echo Python 冷启动测试完成。
echo.

REM 运行Python基准测试
echo 7. 运行 Python 基准测试 (python_baseline.py)
echo ====================================
python python_baseline.py
if errorlevel 1 (
    echo Python 基准测试失败！
    pause
    exit /b 1
)
echo Python 基准测试完成。
echo.

REM 运行Python长时间稳定性测试
echo 8. 运行 Python 长时间稳定性测试 (python_long_stability.py)
echo ====================================
echo 注意：此测试将运行10分钟，请耐心等待...
python python_long_stability.py
if errorlevel 1 (
    echo Python 长时间稳定性测试失败！
    pause
    exit /b 1
)
echo Python 长时间稳定性测试完成。
echo.

REM 生成图表
echo ====================================
echo 生成测试结果图表
echo ====================================
cd "%TEST_DIR%\charts"
python generate_latency_boxplot.py
if errorlevel 1 (
    echo 生成延迟箱线图失败！
    pause
    exit /b 1
)

python plot_rss_curve.py
if errorlevel 1 (
    echo 生成内存使用曲线失败！
    pause
    exit /b 1
)

python generate_cold_start_and_thread_charts.py
if errorlevel 1 (
    echo 生成冷启动和线程配置图表失败！
    pause
    exit /b 1
)
echo 图表生成完成。
echo.

REM 验证所有测试结果文件是否生成
echo ====================================
echo 验证测试结果文件
 echo ====================================
cd "%RESULTS_DIR%"
dir /b
if errorlevel 1 (
    echo 列出结果文件失败！
    pause
    exit /b 1
)
echo.

REM 显示完成信息
echo ====================================
echo 所有测试运行完成！
echo ====================================
echo 测试结果已保存到: %RESULTS_DIR%
echo 图表已保存到: %RESULTS_DIR%

echo.
echo 按任意键退出...
pause > nul
