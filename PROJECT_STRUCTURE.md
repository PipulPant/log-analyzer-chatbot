# Production-Ready Project Structure

## 📁 Directory Layout

```
PYTHON_LOGS_ANALYSER/
├── src/                          # Source code
│   └── log_analyzer/
│       ├── __init__.py           # Package exports
│       ├── analyzer.py           # Main analyzer class
│       ├── pattern_analysis.py   # Pattern recognition & flow analysis
│       └── ml_models.py          # ML models (classification, anomaly, NLP)
│
├── scripts/                       # Executable scripts
│   ├── train.py                  # Basic training script
│   ├── train_ml_models.py         # Advanced ML models training
│   ├── analyze.py                # Analysis script
│   └── generate_report.py        # HTML report generator
│
├── data/                         # Data directory
│   ├── logs/                     # Training log files (78 files)
│   ├── models/                   # Trained models
│   │   ├── trained_model.pkl           # Basic classifier
│   │   ├── classification_xgboost.pkl  # XGBoost classifier
│   │   ├── anomaly_detector.pkl        # Isolation Forest detector
│   │   └── ensemble_config.json        # Ensemble configuration
│   └── server.log                # Sample log file
│
├── docs/                         # Documentation
│   ├── README.md                 # Main documentation
│   ├── TRAINING_DATA_GUIDE.md    # Training data management
│   ├── ML_MODELS_GUIDE.md        # ML models usage
│   ├── OPTIMIZATION_GUIDE.md     # Pattern analysis features
│   └── FILE_ANALYSIS.md          # File structure analysis
│
├── reports/                      # Analysis reports (generated)
├── config/                       # Configuration (empty, for future use)
│
├── requirements.txt              # Python dependencies
├── setup.py                      # Package installation
├── .gitignore                    # Git ignore rules
├── README.md                     # Quick start guide
├── PROJECT_STRUCTURE.md          # This file
├── WHERE_ARE_FILES.md            # File locations guide
├── train.bat                     # Quick training (Windows)
└── analyze.bat                   # Quick analysis (Windows)
```

## 🎯 Key Features

### Organized Structure
- **src/**: All source code in proper package structure
- **scripts/**: Standalone executable scripts
- **data/**: All data files organized by type
- **docs/**: Documentation separated from code

### Production Ready
- Proper Python package structure
- Installable via `setup.py`
- Clear separation of concerns
- Easy to maintain and extend

### Easy to Use
- Simple batch files for Windows
- Clear script interfaces
- Comprehensive documentation

## 🚀 Usage

### Training
```bash
# Using script
python scripts/train.py

# Using batch file (Windows)
train.bat
```

### Analysis
```bash
# Using script
python scripts/analyze.py --logfile data/server.log

# Using batch file (Windows)
analyze.bat
```

## 📝 Notes

- All paths are relative to project root
- Models saved in `data/models/` (with relative paths for portability)
- Training logs stored in `data/logs/` (automatically combined)
- Reports generated in `reports/` directory
- Configuration files use relative paths (portable across devices)

