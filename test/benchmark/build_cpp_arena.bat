echo off
setlocal
cd /d "%~dp0"

set "MSVC_BIN=d:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64"
set "MSVC_INC=d:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\include"
set "MSVC_LIB=d:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\lib\x64"
set "SDK_INC=C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0"
set "SDK_LIB=C:\Program Files (x86)\Windows Kits\10\Lib\10.0.26100.0"
set "ORT_INC=..\..\third_party\onnxruntime-win-x64-1.23.2\include"
set "ORT_LIB=..\..\third_party\onnxruntime-win-x64-1.23.2\lib"

if not exist "%MSVC_BIN%\cl.exe" goto :no_msvc

echo Compiling cpp_arena_ablation_benchmark.cpp with MSVC...
"%MSVC_BIN%\cl.exe" /EHsc /O2 /utf-8 /std:c++17 /I"%MSVC_INC%" /I"%SDK_INC%\ucrt" /I"%SDK_INC%\um" /I"%SDK_INC%\shared" /I"%ORT_INC%" cpp_arena_ablation_benchmark.cpp /link /LIBPATH:"%MSVC_LIB%" /LIBPATH:"%SDK_LIB%\ucrt\x64" /LIBPATH:"%SDK_LIB%\um\x64" /LIBPATH:"%ORT_LIB%" onnxruntime.lib /OUT:cpp_arena_ablation_benchmark.exe
if %errorlevel%==0 (
    echo [OK] Build succeeded
) else (
    echo [ERROR] Build failed
)
goto :end

:no_msvc
echo [ERROR] MSVC (cl.exe) not found at: %MSVC_BIN%
if exist cpp_arena_ablation_benchmark.exe (
    echo [INFO] cpp_arena_ablation_benchmark.exe already exists, skipping build
)
goto :end

:end
endlocal
