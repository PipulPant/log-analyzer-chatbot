@echo off
REM Run web application (Windows)

cd /d "%~dp0"

echo Starting Log Analyzer Web Application...
echo ========================================
echo.
echo Checking for available port (starting from 5000)...
echo The server will display the actual URL when ready
echo.
echo Press Ctrl+C to stop the server
echo.

python app.py

pause

