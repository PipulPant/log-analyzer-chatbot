# Where Are Analysis Files Stored?

## 📍 File Locations

### Analysis Reports (Issue Identification Results)

**Default Location:** `reports/` folder in project root

When you run analysis, the following files are created:

1. **`reports/analysis_report.json`**
   - **Location:** `PYTHON_LOGS_ANALYSER/reports/analysis_report.json`
   - **Contains:** Complete analysis results in JSON format
   - **Includes:**
     - All detected failures
     - Statistics (total lines, failures, etc.)
     - Model information
     - Detailed failure data (timestamp, component, severity, keywords, etc.)

2. **`reports/analysis_report.html`**
   - **Location:** `PYTHON_LOGS_ANALYSER/reports/analysis_report.html`
   - **Contains:** Beautiful HTML report with visual dashboard
   - **Features:**
     - Visual statistics cards
     - Detailed failure cards
     - Color-coded severity indicators
     - Expandable raw log lines
     - Keyword tags
     - Model performance metrics

### Custom Output Location

You can specify a custom location:

```bash
python scripts/analyze.py --logfile data/server.log -o my_custom_report.json
```

This will create:
- `my_custom_report.json`
- `my_custom_report.html` (auto-generated)

## 📂 Complete File Structure

```
PYTHON_LOGS_ANALYSER/
├── reports/                          ← ANALYSIS RESULTS HERE
│   ├── analysis_report.json          ← JSON report with all failures
│   └── analysis_report.html          ← HTML report (open in browser)
│
├── data/
│   ├── logs/                         ← Training log files
│   ├── models/
│   │   └── trained_model.pkl         ← Trained ML model
│   └── server.log                    ← Your log file to analyze
│
├── scripts/                          ← Analysis scripts
└── src/                              ← Source code
```

## 🔍 How to View Results

### Option 1: HTML Report (Recommended)
1. Navigate to `reports/` folder
2. Open `analysis_report.html` in your web browser
3. View all failures with visual dashboard

### Option 2: JSON Report
1. Navigate to `reports/` folder
2. Open `analysis_report.json` in any text editor
3. Search for specific failures or use JSON viewer

### Option 3: Command Line
The analysis script prints a summary to the console showing:
- Total failures found
- Critical errors
- Processing time
- File locations

## 📝 Example Workflow

1. **Run Analysis:**
   ```bash
   python scripts/analyze.py --logfile data/server.log
   ```

2. **Results Saved To:**
   - `reports/analysis_report.json`
   - `reports/analysis_report.html`

3. **View Results:**
   - Open `reports/analysis_report.html` in browser
   - Or check `reports/analysis_report.json` for detailed data

## 🎯 Quick Reference

| File Type | Location | Purpose |
|-----------|----------|---------|
| JSON Report | `reports/analysis_report.json` | Machine-readable detailed results |
| HTML Report | `reports/analysis_report.html` | Human-readable visual dashboard |
| Trained Model | `data/models/trained_model.pkl` | ML model for analysis |
| Log Files | `data/logs/*.txt` | Training data |
| Sample Log | `data/server.log` | Example log to analyze |

## 💡 Tips

- **Multiple Analyses:** Each analysis overwrites the default report files
- **Custom Names:** Use `-o` flag to save with custom names/timestamps
- **Organize Reports:** Create dated folders for historical analysis:
  ```bash
  python scripts/analyze.py -o reports/2025-12-01_analysis.json
  ```

