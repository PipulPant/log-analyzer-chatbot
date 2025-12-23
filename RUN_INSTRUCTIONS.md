# How to Run Log Analyzer - macOS & Windows Guide

## 🍎 macOS Instructions

### Prerequisites

1. **Python 3.8+** (check: `python3 --version`)
2. **pip** (usually comes with Python)

### Step 1: Install Dependencies

```bash
# Navigate to project directory
cd /path/to/PYTHON_LOGS_ANALYSER

# Install required packages
pip3 install -r requirements.txt

# Or if using virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Train Models

```bash
# Train all ML models (recommended)
python3 scripts/train_ml_models.py --train-all

# Or train basic model
python3 scripts/train.py
```

### Step 3: Analyze Logs

```bash
# Basic analysis
python3 scripts/analyze.py --logfile data/server.log

# With custom output
python3 scripts/analyze.py --logfile data/server.log -o reports/my_report.json

# Skip HTML report
python3 scripts/analyze.py --logfile data/server.log --no-html
```

### Quick Commands (macOS)

```bash
# Training
python3 scripts/train_ml_models.py --train-all

# Analysis
python3 scripts/analyze.py --logfile data/server.log

# View results
open reports/analysis_report.html
```

---

## 🪟 Windows Instructions

### Prerequisites

1. **Python 3.8+** (download from python.org)
2. **pip** (usually comes with Python)

### Step 1: Install Dependencies

```cmd
REM Navigate to project directory
cd C:\path\to\PYTHON_LOGS_ANALYSER

REM Install required packages
pip install -r requirements.txt

REM Or if using virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2: Train Models

**Option A: Using Batch File (Easiest)**
```cmd
train.bat
```

**Option B: Using Python Script**
```cmd
REM Train all ML models (recommended)
python scripts\train_ml_models.py --train-all

REM Or train basic model
python scripts\train.py
```

### Step 3: Analyze Logs

**Option A: Using Batch File (Easiest)**
```cmd
analyze.bat
```

**Option B: Using Python Script**
```cmd
REM Basic analysis
python scripts\analyze.py --logfile data\server.log

REM With custom output
python scripts\analyze.py --logfile data\server.log -o reports\my_report.json

REM Skip HTML report
python scripts\analyze.py --logfile data\server.log --no-html
```

### Quick Commands (Windows)

```cmd
REM Training
train.bat
REM or
python scripts\train_ml_models.py --train-all

REM Analysis
analyze.bat
REM or
python scripts\analyze.py --logfile data\server.log

REM View results (opens in default browser)
start reports\analysis_report.html
```

---

## 📋 Complete Workflow Examples

### macOS Complete Workflow

```bash
# 1. Navigate to project
cd ~/Downloads/PYTHON_LOGS_ANALYSER

# 2. Install dependencies (first time only)
pip3 install -r requirements.txt

# 3. Train models (first time or when adding new training data)
python3 scripts/train_ml_models.py --train-all

# 4. Analyze a log file
python3 scripts/analyze.py --logfile data/server.log

# 5. View results
open reports/analysis_report.html
```

### Windows Complete Workflow

```cmd
REM 1. Navigate to project
cd C:\Users\YourName\Downloads\PYTHON_LOGS_ANALYSER

REM 2. Install dependencies (first time only)
pip install -r requirements.txt

REM 3. Train models (first time or when adding new training data)
train.bat
REM or
python scripts\train_ml_models.py --train-all

REM 4. Analyze a log file
analyze.bat
REM or
python scripts\analyze.py --logfile data\server.log

REM 5. View results
start reports\analysis_report.html
```

---

## 🔧 Platform-Specific Notes

### macOS

- Use `python3` command (not `python`)
- Use forward slashes `/` in paths
- Use `open` command to view HTML files
- Virtual environment activation: `source .venv/bin/activate`

### Windows

- Use `python` command (or `py` if Python Launcher installed)
- Use backslashes `\` in paths (or forward slashes work too)
- Use `start` command to view HTML files
- Virtual environment activation: `.venv\Scripts\activate`
- Batch files (`.bat`) are provided for convenience

---

## 🚀 Quick Start Scripts

### macOS Shell Script

Create `run_analysis.sh`:
```bash
#!/bin/bash
cd "$(dirname "$0")"
python3 scripts/analyze.py --logfile data/server.log
open reports/analysis_report.html
```

Make it executable:
```bash
chmod +x run_analysis.sh
./run_analysis.sh
```

### Windows PowerShell Script

Create `run_analysis.ps1`:
```powershell
cd $PSScriptRoot
python scripts\analyze.py --logfile data\server.log
Start-Process reports\analysis_report.html
```

Run it:
```powershell
.\run_analysis.ps1
```

---

## 📝 Common Commands Reference

### Training Commands

| Task | macOS | Windows |
|------|-------|---------|
| Train all ML models | `python3 scripts/train_ml_models.py --train-all` | `python scripts\train_ml_models.py --train-all` |
| Train basic model | `python3 scripts/train.py` | `python scripts\train.py` |
| Train classification only | `python3 scripts/train_ml_models.py --train-classification` | `python scripts\train_ml_models.py --train-classification` |
| Train anomaly detector | `python3 scripts/train_ml_models.py --train-anomaly` | `python scripts\train_ml_models.py --train-anomaly` |

### Analysis Commands

| Task | macOS | Windows |
|------|-------|---------|
| Basic analysis | `python3 scripts/analyze.py --logfile data/server.log` | `python scripts\analyze.py --logfile data\server.log` |
| Custom output | `python3 scripts/analyze.py -o reports/my.json` | `python scripts\analyze.py -o reports\my.json` |
| With ML config | `python3 scripts/analyze.py --ml-config data/models/ensemble_config.json` | `python scripts\analyze.py --ml-config data\models\ensemble_config.json` |
| Skip HTML | `python3 scripts/analyze.py --no-html` | `python scripts\analyze.py --no-html` |

### View Results

| Task | macOS | Windows |
|------|-------|---------|
| Open HTML report | `open reports/analysis_report.html` | `start reports\analysis_report.html` |
| View JSON report | `cat reports/analysis_report.json` | `type reports\analysis_report.json` |

---

## 🐛 Troubleshooting

### macOS Issues

**Issue: `python3: command not found`**
```bash
# Install Python via Homebrew
brew install python3

# Or download from python.org
```

**Issue: Permission denied**
```bash
# Make scripts executable
chmod +x scripts/*.py
```

**Issue: Module not found**
```bash
# Ensure you're in project directory
cd /path/to/PYTHON_LOGS_ANALYSER

# Install dependencies
pip3 install -r requirements.txt
```

### Windows Issues

**Issue: `python: command not found`**
- Add Python to PATH during installation
- Or use `py` command instead: `py scripts\analyze.py`

**Issue: Execution policy error (PowerShell)**
```powershell
# Allow script execution
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Issue: Module not found**
```cmd
REM Ensure you're in project directory
cd C:\path\to\PYTHON_LOGS_ANALYSER

REM Install dependencies
pip install -r requirements.txt
```

**Issue: Path with spaces**
```cmd
REM Use quotes around paths
python scripts\analyze.py --logfile "C:\My Files\server.log"
```

---

## 💡 Tips

1. **Use Virtual Environment**: Keeps dependencies isolated
2. **Check Python Version**: `python3 --version` (macOS) or `python --version` (Windows)
3. **Use Batch Files**: On Windows, `.bat` files make it easier
4. **Relative Paths**: All paths in configs are relative - works on any device
5. **Training Data**: Add new `.txt` files to `data/logs/` and retrain

---

## 📚 Next Steps

After running analysis:
1. Check `reports/analysis_report.html` for visual results
2. Review `reports/analysis_report.json` for detailed data
3. Check recommendations in the HTML report
4. Review flow analysis for transaction patterns

For more details, see:
- `README.md` - Main documentation
- `docs/ML_MODELS_GUIDE.md` - ML models usage
- `docs/OPTIMIZATION_GUIDE.md` - Pattern analysis features

