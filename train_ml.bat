@echo off
REM Windows batch script to train ML models
REM Trains all ML models on data/logs/

cd /d "%~dp0"

echo ==========================================
echo Log Analyzer - Training ML Models
echo ==========================================
echo.

python scripts\train_ml_models.py --train-all

echo.
pause

