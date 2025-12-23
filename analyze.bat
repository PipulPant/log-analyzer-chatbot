@echo off
REM Quick analysis script - analyzes logs and opens HTML report
echo ==========================================
echo Log Analyzer - Running Analysis
echo ==========================================
echo.

REM Run analysis (automatically uses ensemble_config.json if available)
python scripts\analyze.py --logfile data\server.log

REM Open HTML report if it exists
if exist "reports\analysis_report.html" (
    echo.
    echo Opening HTML report...
    start reports\analysis_report.html
)

pause

