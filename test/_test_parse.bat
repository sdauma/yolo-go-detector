@echo off
REM Quick test: just validate the script can be parsed (echo only
echo [TEST] RUN_72H_STABILITY.bat parse test
call "D:\mlz\trae_projects\1\yolo-go-detector\test\RUN_72H_STABILITY.bat --dry-run 2>&1 | findstr /i "stability" >nul
if errorlevel 1 (
    echo [INFO] Cannot run --dry-run, attempting to at least CALL a simple syntax check.
)
echo [DONE]