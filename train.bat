@echo off
REM Quick training script - combines logs from data/logs/ folder and trains basic model
echo ============================================
echo Training Basic Model from data/logs/ folder
echo ============================================
echo.

python scripts\train.py

echo.
pause

