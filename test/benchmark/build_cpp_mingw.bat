@echo off
chcp 65001 >nul
REM ========================================
REM C++ ONNX Runtime 基准测试编译脚本 (MinGW版)
REM 使用 msys64 中已有的 MinGW 编译器
REM ========================================
cd /d "%~dp0"
echo ===== C++ ONNX Runtime 基准测试编译 =====
echo.

setlocal

set MINGW_BIN=D:\lnmp\msys64\mingw64\bin
set BASE_DIR=..\..
set ONNXRUNTIME_DIR=%BASE_DIR%\third_party\onnxruntime-win-x64-1.23.2

REM 检查编译器
if exist "%MINGW_BIN%\g++.exe" goto :check_onnx
echo 错误: 未找到 g++.exe
echo 路径: %MINGW_BIN%\g++.exe
pause
exit /b 1

:check_onnx
echo 找到 MinGW 编译器: %MINGW_BIN%\g++.exe

REM 检查ONNX Runtime
if exist "%ONNXRUNTIME_DIR%\include\onnxruntime_cxx_api.h" goto :do_compile
echo 错误: 未找到 ONNX Runtime 头文件
echo 路径: %ONNXRUNTIME_DIR%\include\onnxruntime_cxx_api.h
pause
exit /b 1

:do_compile
echo 检测到 ONNX Runtime: %ONNXRUNTIME_DIR%
echo.

REM 设置 PATH
set PATH=%MINGW_BIN%;%PATH%

echo 使用 MinGW g++ 编译...
"%MINGW_BIN%\g++" -O2 -std=c++17 ^
    -I"%ONNXRUNTIME_DIR%\include" ^
    cpp_baseline_benchmark.cpp ^
    -L"%ONNXRUNTIME_DIR%\lib" ^
    -lonnxruntime ^
    -o cpp_baseline_benchmark.exe

if %errorlevel%==0 goto :compile_ok
echo.
echo 编译失败，请检查错误信息
goto :end
:compile_ok
echo.
echo 编译成功！可执行文件: cpp_baseline_benchmark.exe
echo.
echo 复制 ONNX Runtime DLL 到当前目录...
copy "%ONNXRUNTIME_DIR%\lib\onnxruntime.dll" . >nul
copy "%ONNXRUNTIME_DIR%\lib\onnxruntime_providers_shared.dll" . >nul
echo.
echo 使用方法:
echo   cpp_baseline_benchmark.exe                     （默认YOLO11x, 2000次推理）
echo   cpp_baseline_benchmark.exe yolo11n yolo11n.onnx 1000  （YOLO11n, 1000次）
:end
endlocal
pause