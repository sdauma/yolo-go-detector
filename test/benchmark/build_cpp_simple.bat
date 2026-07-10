@echo off
chcp 65001 >nul
REM ========================================
REM C++ ONNX Runtime 基准测试编译脚本 (简化版)
REM 使用项目中已有的 onnxruntime.dll
REM ========================================
cd /d "%~dp0"
echo ===== C++ ONNX Runtime 基准测试编译 =====
echo.

setlocal

REM 设置路径
set BASE_DIR=..\..
set ONNXRUNTIME_DIR=%BASE_DIR%\third_party
set MODEL_DIR=%BASE_DIR%\third_party

REM 检查ONNX Runtime DLL
if exist "%ONNXRUNTIME_DIR%\onnxruntime.dll" goto :check_compiler
echo 错误: 未找到 onnxruntime.dll
echo 请确保 %ONNXRUNTIME_DIR% 目录下存在 onnxruntime.dll
pause
exit /b 1

:check_compiler
echo 检测到 ONNX Runtime: %ONNXRUNTIME_DIR%\onnxruntime.dll
echo.

REM 检查是否有编译器
echo 检查编译环境...

REM 尝试查找g++ (MinGW)
where g++ >nul 2>nul
if %errorlevel%==0 (echo 找到 g++ (MinGW) && goto :compile_gcc)

REM 尝试查找cl (Visual Studio)
where cl >nul 2>nul
if %errorlevel%==0 (echo 找到 cl (Visual Studio) && goto :compile_msvc)

REM 尝试设置Visual Studio环境
echo 尝试设置 Visual Studio 环境...
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>nul
where cl >nul 2>nul
if %errorlevel%==0 (echo 找到 Visual Studio 编译环境 && goto :compile_msvc)

echo.
echo 错误: 未找到 C++ 编译器
echo.
echo 请选择以下方式之一：
echo 1. 安装 Visual Studio 2022 (推荐)
echo    - 下载地址: https://visualstudio.microsoft.com/zh-hans/free-developer-offers/
echo    - 安装时选择 "桌面开发用C++" 工作负载
echo.
echo 2. 安装 MinGW
echo    - 下载地址: https://sourceforge.net/projects/mingw-w64/files/
echo    - 添加安装目录的 bin 文件夹到 PATH
echo.
pause
exit /b 1

:compile_msvc
echo.
echo 使用 Visual Studio 编译...
cl /EHsc /O2 /std:c++17 ^
   /I"%BASE_DIR%\third_party\include" ^
   cpp_baseline_benchmark.cpp ^
   /link "%ONNXRUNTIME_DIR%\onnxruntime.lib" ^
   /OUT:cpp_baseline_benchmark.exe
goto :compile_end

:compile_gcc
echo.
echo 使用 MinGW 编译...
g++ -O2 -std=c++17 ^
    -I"%BASE_DIR%\third_party\include" ^
    cpp_baseline_benchmark.cpp ^
    -L"%ONNXRUNTIME_DIR%" ^
    -lonnxruntime ^
    -o cpp_baseline_benchmark.exe

:compile_end
if %errorlevel%==0 goto :compile_ok
echo.
echo 编译失败，请检查错误信息
goto :end
:compile_ok
echo.
echo 编译成功！可执行文件: cpp_baseline_benchmark.exe
echo.
echo 复制 ONNX Runtime DLL 到当前目录...
copy "%ONNXRUNTIME_DIR%\onnxruntime.dll" . >nul
echo.
echo 使用方法:
echo   cpp_baseline_benchmark.exe                     （默认YOLO11x, 2000次推理）
echo   cpp_baseline_benchmark.exe yolo11n yolo11n.onnx 1000  （YOLO11n, 1000次）
:end
endlocal
pause
