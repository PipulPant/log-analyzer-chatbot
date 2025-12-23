# Quick Start Guide

## 🚀 One-Minute Quick Start

### macOS/Linux

```bash
# 1. Install dependencies
pip3 install -r requirements.txt

# 2. Train models (first time only)
python3 scripts/train_ml_models.py --train-all

# 3. Analyze logs
./run_analysis.sh

# Done! HTML report opens automatically
```

### Windows

```cmd
REM 1. Install dependencies
pip install -r requirements.txt

REM 2. Train models (first time only)
train_ml.bat

REM 3. Analyze logs
analyze.bat

REM Done! HTML report opens automatically
```

---

## 📋 Common Commands

### Training

| Task | macOS/Linux | Windows |
|------|-------------|---------|
| Train all ML models | `python3 scripts/train_ml_models.py --train-all` | `train_ml.bat` |
| Train basic model | `python3 scripts/train.py` | `train.bat` |

### Analysis

| Task | macOS/Linux | Windows |
|------|-------------|---------|
| Analyze default log | `./run_analysis.sh` | `analyze.bat` |
| Analyze custom log | `python3 scripts/analyze.py --logfile path/to/log.txt` | `python scripts\analyze.py --logfile path\to\log.txt` |

### View Results

| Task | macOS/Linux | Windows |
|------|-------------|---------|
| Open HTML report | `open reports/analysis_report.html` | `start reports\analysis_report.html` |

---

## 📁 File Locations

- **Training Data**: `data/logs/*.txt` (all .txt files automatically combined)
- **Trained Models**: `data/models/`
- **Analysis Reports**: `reports/`
  - `analysis_report.json` - Detailed JSON data
  - `analysis_report.html` - Visual HTML report

---

## ⚡ Quick Tips

1. **First Time**: Run training once to create models
2. **Adding Data**: Just add `.txt` files to `data/logs/` and retrain
3. **View Results**: HTML report opens automatically (or manually open `reports/analysis_report.html`)
4. **Portable**: All paths are relative - works on any device!

For detailed instructions, see `RUN_INSTRUCTIONS.md`

