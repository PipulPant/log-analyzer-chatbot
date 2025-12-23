# Log Analyzer - Production Ready

Enhanced log analyzer with supervised learning, ML models, and comprehensive pattern analysis for failure detection. Production-ready structure with organized modules and scripts.

Access the chatbot via: https://log-analyzer-chatbot.onrender.com/

## 📁 Project Structure

```
PYTHON_LOGS_ANALYSER/
├── src/
│   └── log_analyzer/
│       ├── __init__.py          # Package initialization
│       ├── analyzer.py          # Main analyzer class
│       ├── pattern_analysis.py  # Pattern recognition & flow analysis
│       └── ml_models.py         # ML models (classification, anomaly detection)
├── scripts/
│   ├── train.py                 # Basic training script
│   ├── train_ml_models.py       # Advanced ML models training
│   ├── analyze.py               # Analysis script
│   └── generate_report.py      # HTML report generator
├── web/                         # Web application files
│   ├── templates/
│   │   └── index.html          # Chat interface HTML
│   └── static/
│       ├── css/
│       │   └── style.css       # Styling
│       └── js/
│           └── app.js          # Frontend logic
├── app.py                       # Flask web application
├── run_web.sh                   # Web app launcher (macOS/Linux)
└── run_web.bat                  # Web app launcher (Windows)
├── data/
│   ├── logs/                    # Training log files (78 files)
│   ├── models/                  # Trained models
│   │   ├── trained_model.pkl           # Basic classifier
│   │   ├── classification_xgboost.pkl  # XGBoost classifier
│   │   ├── anomaly_detector.pkl        # Isolation Forest detector
│   │   └── ensemble_config.json        # Ensemble configuration
│   └── server.log               # Sample log file
├── docs/
│   ├── README.md                # Main documentation
│   ├── TRAINING_DATA_GUIDE.md   # Training data management
│   ├── ML_MODELS_GUIDE.md       # ML models usage
│   ├── OPTIMIZATION_GUIDE.md    # Pattern analysis features
│   └── FILE_ANALYSIS.md         # File structure analysis
├── reports/                     # Analysis reports (generated)
├── config/                      # Configuration files (empty, for future use)
├── requirements.txt             # Python dependencies
├── setup.py                     # Package installation
├── train.bat                    # Quick training (Windows)
└── analyze.bat                  # Quick analysis (Windows)
```

## 🚀 Quick Start

### Platform-Specific Instructions

- **macOS/Linux**: See `RUN_INSTRUCTIONS.md` for detailed macOS instructions
- **Windows**: See `RUN_INSTRUCTIONS.md` for detailed Windows instructions

### 1. Install Dependencies

**macOS:**
```bash
pip3 install -r requirements.txt
```

**Windows:**
```cmd
pip install -r requirements.txt
```

**Required packages:**
- numpy, scikit-learn, matplotlib, scipy (core)
- xgboost, sentence-transformers, torch (ML models - recommended)

### 2. Train the Model

**Training Data Location:** All training log files should be placed in `data/logs/` directory. The training scripts automatically combine all `.txt` and `.log` files from this directory.

**Option A: Basic Training (Recommended for first-time users)**
```bash
# Trains on all files in data/logs/
python scripts/train.py
```

**Option B: Using batch file (Windows)**
```cmd
train.bat
```

**Option D: Using shell script (macOS/Linux)**
```bash
# Train ML models
python3 scripts/train_ml_models.py --train-all
```

**Option C: Advanced ML Models Training**
```bash
# Train classification, anomaly detection, and ensemble models
python scripts/train_ml_models.py --train-all

# Train specific models
python scripts/train_ml_models.py --train-classification --model-type xgboost
python scripts/train_ml_models.py --train-anomaly
python scripts/train_ml_models.py --train-ensemble
```

### 3. Analyze Logs

**Option A: Web Application (ChatGPT-style Interface)**
```bash
# macOS/Linux
./run_web.sh
# Or: python3 app.py

# Windows
run_web.bat
# Or: python app.py
```
Then open `http://localhost:5000` in your browser to use the chat interface.

**Option B: Command-Line Analysis**

**macOS/Linux:**
```bash
# Basic analysis
python3 scripts/analyze.py --logfile data/server.log

# Or use shell script (opens HTML report automatically)
./run_analysis.sh

# With custom log file
./run_analysis.sh data/server.log
```

**Windows:**
```cmd
REM Using batch file (easiest - opens HTML report automatically)
analyze.bat

REM Or using Python directly
python scripts\analyze.py --logfile data\server.log

REM Or use run script
run_analysis.bat
```

**Note:** The analyzer automatically uses `ensemble_config.json` if available for enhanced ML-based analysis.

## 📖 Detailed Usage

### Training

```bash
# Basic training (combines all files from data/logs/)
python scripts/train.py

# Advanced ML models training
python scripts/train_ml_models.py --train-all --logs-dir data/logs

# Train specific model type
python scripts/train_ml_models.py --train-classification --model-type random_forest
```

### Analysis

```bash
# Basic analysis
python scripts/analyze.py --logfile data/server.log

# With trained model
python scripts/analyze.py --logfile data/server.log --model data/models/trained_model.pkl

# Custom output location
python scripts/analyze.py --logfile data/server.log -o reports/my_analysis.json

# Skip HTML report generation
python scripts/analyze.py --logfile data/server.log --no-html
```

### Using ML Models in Analysis

```python
from log_analyzer import LogAnalyzer
from pathlib import Path
import json

# Load ensemble configuration
with open('data/models/ensemble_config.json') as f:
    ml_config = json.load(f)

# Create analyzer with ML models
analyzer = LogAnalyzer(ml_models_config=ml_config)

# Analyze logs
results = analyzer.analyze(Path('data/server.log'))

# Access enhanced results
print(f"Connection Issues: {results['stats']['connection_issues']}")
print(f"HTTP Errors: {results['stats']['http_errors']}")
print(f"Recommendations: {len(results['recommendations'])}")
```

## 🎯 Features

### ✅ Comprehensive Failure Detection
- Explicit failure statuses
- HTTP status codes (4xx, 5xx)
- JSON-embedded errors
- Error codes and patterns
- "No Record Found" patterns

### ✅ Advanced Pattern Analysis
- **Error Pattern Recognition**: Connection, HTTP, database, service, authentication, timeout errors
- **State Machine Analysis**: Tracks transaction flow (START → PROCESSING → EXTERNAL_CALL → RESPONSE → END)
- **Timing Analysis**: Detects timeouts and performance bottlenecks
- **Data Consistency Checks**: ID consistency, transformation validation, mandatory fields
- **Flow Break Detection**: Identifies incomplete transaction flows

### ✅ Machine Learning Models
- **Classification Models**: XGBoost, Random Forest, Gradient Boosting, SVM
- **Anomaly Detection**: Isolation Forest for novel pattern detection
- **NLP Models**: Sentence Transformers for semantic similarity
- **Ensemble Approach**: Combines multiple models with weighted voting

### ✅ Supervised Learning
- Trainable Random Forest classifier
- Model persistence
- High accuracy detection
- Automatic feature extraction

### ✅ Multiple Log Formats
- Automatic format detection
- Support for multiple log structures
- Pipe-separated format support

### ✅ HTML Reports
- Visual dashboard with statistics
- Detailed failure cards
- Color-coded severity indicators
- Flow analysis visualization
- Recommendations section
- Expandable raw logs

### ✅ Web Application (NEW!)
- **ChatGPT-style Interface**: Modern chat UI for log analysis
- **File Upload**: Drag-and-drop or click to upload log files
- **Text Input**: Paste log content directly in chat
- **Real-Time Analysis**: Instant AI-powered analysis responses
- **Formatted Results**: Beautiful markdown-formatted responses

## 📊 Output Files

After analysis, you get:
- `reports/analysis_report.json` - Detailed JSON report with:
  - Statistics (failures, errors, timing metrics)
  - Flow analysis (transaction flow statistics)
  - Issues categorized by severity (critical, warning, info)
  - Recommendations with priority levels
  - ML model predictions and scores
- `reports/analysis_report.html` - Beautiful HTML report (auto-generated)

## 🔧 Configuration

### Command-Line Arguments

```bash
# Analysis script
python scripts/analyze.py --help

# Training scripts
python scripts/train.py --help
python scripts/train_ml_models.py --help
```

### ML Models Configuration

The `data/models/ensemble_config.json` file contains:
- Model paths (relative paths for portability)
- Model types and parameters
- Ensemble weights
- Contamination settings for anomaly detection

## 📝 Notes

- **Training Data**: All training log files go in `data/logs/` - scripts automatically combine all `.txt` and `.log` files
- **Models**: Trained models are saved in `data/models/` with relative paths for portability
- **Reports**: Analysis reports are generated in `reports/` directory
- **Portability**: Configuration files use relative paths - works on any device/location
- All paths are relative to project root

## 📚 Additional Documentation

- **Web Application Guide**: `WEB_APP_GUIDE.md` - Complete guide for the ChatGPT-style web interface
- **Quick Start**: `QUICK_START.md` - One-minute quick reference
- **Run Instructions**: `RUN_INSTRUCTIONS.md` - Platform-specific instructions (macOS & Windows)
- **Training Data Guide**: `docs/TRAINING_DATA_GUIDE.md` - How to manage training data
- **ML Models Guide**: `docs/ML_MODELS_GUIDE.md` - Advanced ML models usage and examples
- **Optimization Guide**: `docs/OPTIMIZATION_GUIDE.md` - Pattern analysis features and configuration
- **File Locations**: `WHERE_ARE_FILES.md` - Where analysis results are stored

## 🛠️ Development

### Install in Development Mode

```bash
pip install -e .
```

### Project Structure

See `PROJECT_STRUCTURE.md` for detailed project organization.

## 🔍 Example Workflow

```bash
# 1. Add training data
cp new_logs/*.txt data/logs/

# 2. Train models
python scripts/train_ml_models.py --train-all

# 3. Analyze logs
python scripts/analyze.py --logfile data/server.log

# 4. View results
open reports/analysis_report.html
```

## 📄 License

See LICENSE file for details.

