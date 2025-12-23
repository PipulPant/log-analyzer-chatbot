@echo off
REM Windows batch script to run log analysis
REM Usage: run_analysis.bat [logfile]

cd /d "%~dp0"

if "%1"=="" (
    set LOG_FILE=data\server.log
) else (
    set LOG_FILE=%1
)

echo ==========================================
echo Log Analyzer - Running Analysis
echo ==========================================
echo.

REM Run analysis
python scripts\analyze.py --logfile "%LOG_FILE%"

REM Open HTML report if it exists
if exist "reports\analysis_report.html" (
    echo.
    echo Opening HTML report...
    start reports\analysis_report.html
)

pause

