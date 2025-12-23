# Log Analyzer with Supervised Learning

Enhanced log analyzer that detects failures using rule-based detection and machine learning. Trained to identify errors, status codes, and failure patterns in log files.

## Features

✅ **Comprehensive Failure Detection**
- Explicit failure statuses ("failure", "error", "failed")
- HTTP status codes (4xx, 5xx) - including 404, 500, 505, etc.
- JSON-embedded errors (`"code": "404"`, `"status": "Failed"`)
- Error codes (EC123, ERR456, etc.)
- "No Record Found" patterns
- Failure-related keywords and patterns

✅ **Supervised Learning**
- Trainable Random Forest classifier
- Model persistence (save/load)
- High accuracy detection
- Automatic feature extraction

✅ **Multiple Log Formats**
- New format: `TIMESTAMP|QUEUE_TIME_MS|PROCESS_TIME_MS|PROCESSOR_UUID|PROCESSOR_NAME|COMPONENT|RELATIONSHIP|PG_UUID|PG_NAME|IDENTIFIER|LOG`
- Original format: `TIMESTAMP|NUM|NUM|UUID|COMPONENT|PJMS|STATUS|UUID|OPERATION|TXN_ID|MESSAGE`
- Automatic format detection

## Quick Start

### 1. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

Or if Python is not in PATH:
```bash
C:\Users\pipul.pant\AppData\Local\Programs\Python\Python313\python.exe -m pip install -r requirements.txt
```

### 2. Train the Model

**Option A: Train from logs/ folder (Recommended)**
```bash
python combine_and_train.py
```

Or double-click: `train.bat`

This will:
- Combine all log files from `logs/` folder
- Train a model on the combined data
- Save model to `trained_model.pkl`

**Option B: Train from a single log file**
```bash
python python_logs_analyser.py --train --logfile server.log --model trained_model.pkl
```

### 3. Analyze Logs

**Using trained model:**
```bash
python python_logs_analyser.py --model trained_model.pkl --logfile server.log
```

Or double-click: `analyze.bat`

This will generate:
- `analysis_report.json` - Detailed JSON report
- `analysis_report.html` - Beautiful HTML report (opens in browser)

**Basic analysis (rule-based only):**
```bash
python python_logs_analyser.py --logfile server.log
```

### 4. View HTML Report

After analysis, open `analysis_report.html` in your browser to see:
- 📊 Visual statistics dashboard
- 🚨 Detailed failure cards with all information
- 🎨 Color-coded severity indicators
- 📋 Expandable raw log lines
- 🔍 Keyword tags for each failure

## Detailed Usage

### Training Options

```bash
# Train with default settings (Random Forest)
python combine_and_train.py

# Train from specific log file
python python_logs_analyser.py --train --logfile server.log --model trained_model.pkl

# Train with different model type
python python_logs_analyser.py --train --logfile server.log --model trained_model.pkl --model-type logistic_regression

# Available model types: random_forest, logistic_regression, svm, naive_bayes
```

### Analysis Options

```bash
# Analyze with trained model
python python_logs_analyser.py --model trained_model.pkl --logfile server.log

# Save to custom output file
python python_logs_analyser.py --model trained_model.pkl --logfile server.log -o my_report.json

# Limit number of failures in output
python python_logs_analyser.py --model trained_model.pkl --logfile server.log --max-failures 50

# Verbose output
python python_logs_analyser.py --model trained_model.pkl --logfile server.log --verbose
```

## File Structure

```
PYTHON_LOGS_ANALYSER/
├── python_logs_analyser.py    # Main analysis script
├── combine_and_train.py        # Script to train from logs/ folder
├── trained_model.pkl           # Trained model (generated after training)
├── requirements.txt             # Python dependencies
├── train.bat                   # Quick training (double-click)
├── analyze.bat                 # Quick analysis (double-click)
├── README.md                   # This file
├── server.log                  # Your log file to analyze
└── logs/                       # Training data folder (92 log files)
    └── *.txt                   # Individual log files
```

## What Gets Detected

### 1. Status Codes
- **4xx codes**: 400, 401, 403, 404, 405, 408, 409, 410, 422, 429, etc.
- **5xx codes**: 500, 501, 502, 503, 504, 505, 507, 508, 510, 511, etc.
- Detected in: JSON (`"code": "404"`), text (`Status Code: 500`), fields (`respCode: 404`)

### 2. JSON Errors
- `"code": "404"` → Detected as `json_code_404`
- `"status": "Failed"` → Detected as `json_status_failed`
- `"message": "Customer not found"` → Detected as `json_error_message`

### 3. Failure Statuses
- Explicit: "failure", "error", "failed"
- Patterns: "No Record Found", "exception", "timeout"
- Keywords: connection refused, out of memory, unauthorized, etc.

## Example Output

```
[SUCCESS] Analysis complete! Results saved to analysis_report.json
================================================================================
LOG ANALYSIS SUMMARY
================================================================================
Analyzed file: server.log
File size: 0.0 MB
Detected format: new_format
Total lines processed: 88
Parsed entries: 87
Rule-based failures: 12
ML-detected failures: 0
Critical errors: 11
Warnings: 0
Processing time: 0.09s
Throughput: 930 lines/sec
```

## Training Process

1. **Data Collection**: Combines all `.txt` files from `logs/` folder
2. **Format Detection**: Automatically detects log format
3. **Label Generation**: Uses rule-based detection to create training labels
4. **Feature Extraction**: Extracts text and structured features
5. **Model Training**: Trains Random Forest classifier
6. **Evaluation**: Tests on 20% of data, shows accuracy metrics
7. **Model Saving**: Saves to `trained_model.pkl`

## Model Performance

Current model trained on:
- **2,087 log entries** from 92 files
- **14 failures** (0.7%)
- **Accuracy**: 100.0%
- **F1 Score**: 1.000
- **Precision**: 1.000
- **Recall**: 1.000

## Troubleshooting

### Python Not Found
If `python` command doesn't work, use full path:
```bash
C:\Users\pipul.pant\AppData\Local\Programs\Python\Python313\python.exe python_logs_analyser.py --help
```

### Dependencies Missing
```bash
python -m pip install -r requirements.txt
```

### No Failures Detected
- Check log format matches expected patterns
- Verify log file has failure entries
- Use `--verbose` flag for detailed output

### Model Not Found
Train the model first:
```bash
python combine_and_train.py
```

## Command Reference

| Command | Description |
|---------|-------------|
| `python combine_and_train.py` | Train from logs/ folder |
| `python python_logs_analyser.py --train --logfile FILE` | Train from single file |
| `python python_logs_analyser.py --model MODEL.pkl --logfile FILE` | Analyze with model |
| `python python_logs_analyser.py --logfile FILE` | Basic analysis (no model) |
| `.\train.bat` | Quick train (Windows) |
| `.\analyze.bat` | Quick analyze (Windows) |

## Output Files

After running analysis, you get:
- **analysis_report.json**: Detailed analysis results with all failures (JSON format)
- **analysis_report.html**: Beautiful HTML report with visual dashboard (auto-generated)

**Note:** These files are regenerated each time you run analysis. You can delete them if needed.

### View HTML Report

The HTML report includes:
- **Visual Statistics Dashboard** - All metrics in cards
- **Detailed Failure Cards** - Each failure with full details
- **Color-coded Severity** - Visual severity indicators
- **Expandable Raw Logs** - Click to see full log lines
- **Keyword Tags** - All detected keywords highlighted
- **Model Information** - Accuracy and performance metrics

Just open `analysis_report.html` in any web browser!

### Generate HTML Report Manually

If you want to generate HTML from existing JSON:
```bash
python generate_html_report.py --json analysis_report.json -o report.html
```

## Notes

- The model learns from rule-based labels automatically
- More training data = better accuracy
- Model can be retrained anytime with new data
- Analysis reports can be regenerated (not essential to keep)

