# Training Data Guide

## Training Data Location

All training data is stored in: **`data/logs/`**

This directory contains multiple log files (`.txt` files) that will be automatically combined during training.

## How Training Works

### Automatic Combination

When you run training scripts, all log files from `data/logs/` are automatically:

1. **Found**: Scripts scan for all `.txt` files in `data/logs/`
2. **Combined**: All files are merged into a single training dataset
3. **Cleaned**: Headers and invalid lines are removed
4. **Used**: Combined data is used for model training

### Training Scripts

#### 1. Basic Model Training (`train.py`)

```bash
# Trains the basic supervised learning model
python scripts/train.py
```

This script:
- Combines all files from `data/logs/`
- Trains a Random Forest classifier
- Saves model to `data/models/trained_model.pkl`

#### 2. ML Models Training (`train_ml_models.py`)

```bash
# Train all ML models using all data/logs files
python scripts/train_ml_models.py --train-all

# Train specific models
python scripts/train_ml_models.py --train-classification --model-type xgboost
python scripts/train_ml_models.py --train-anomaly
python scripts/train_ml_models.py --train-ensemble
```

This script:
- Combines all files from `data/logs/` (default)
- Trains classification, anomaly detection, and ensemble models
- Saves models to `data/models/`

### Manual Training Data Selection

You can also specify a single log file:

```bash
# Use a specific log file instead of combining all
python scripts/train_ml_models.py --logfile data/server.log --train-all
```

## Training Data Format

**Supported File Extensions:** `.txt` and `.log`

Both file types are automatically detected and combined during training.

Each log file in `data/logs/` should follow this format:

```
TIMESTAMP|QUEUE_TIME_MS|PROCESS_TIME_MS|PROCESSOR_UUID|PROCESSOR_NAME|COMPONENT|RELATIONSHIP|PG_UUID|PG_NAME|IDENTIFIER|LOG
2025-11-28 21:14:00,004|1|3|7a2a172d|Fetch Pending Verification Trx from DB|ESQLR|success|fd69eddb|Background External Check Status|
...
```

**Important:**
- Header line (starting with `TIMESTAMP|`) is automatically skipped
- Empty lines are ignored
- Lines with less than 5 pipe separators (`|`) are skipped
- Files are processed in sorted order (alphabetical)

## Adding New Training Data

### Method 1: Add to `data/logs/`

Simply add new `.txt` or `.log` files to `data/logs/`:

```bash
# Copy new log files (both .txt and .log supported)
cp new_logs/*.txt data/logs/
cp new_logs/*.log data/logs/

# Training will automatically include all .txt and .log files
python scripts/train.py
```

### Method 2: Replace Files

Replace existing files in `data/logs/` with updated versions.

### Method 3: Use Specific Directory

```bash
# Train on a different directory
python scripts/train_ml_models.py --logs-dir /path/to/other/logs --train-all
```

## Training Data Statistics

After combining, you'll see output like:

```
Found 78 log files in data/logs
Combining 78 log files...
Combined 12345 log lines from 78 files
Training data: 12345 entries (1234 failures, 11111 successes)
```

## Best Practices

1. **Include Diverse Data**: Add logs from different scenarios, time periods, and error types
2. **Balance Failures**: Ensure you have both failure and success cases
3. **Regular Updates**: Retrain models when new patterns emerge
4. **Validation**: Keep some data separate for validation/testing
5. **Clean Data**: Ensure log files are properly formatted

## File Organization

```
data/
├── logs/                    ← All training data here
│   ├── file1.txt
│   ├── file2.txt
│   └── ... (78 files)
├── models/                  ← Trained models saved here
│   ├── trained_model.pkl
│   ├── classification_xgboost.pkl
│   ├── anomaly_detector.pkl
│   └── ensemble_config.json
└── server.log              ← Sample/test log file
```

## Troubleshooting

### No Log Files Found

```
Error: No .txt or .log files found in data/logs
```

**Solution**: Ensure `.txt` or `.log` files exist in `data/logs/`

### Insufficient Data

```
Error: Insufficient data: 5 entries (minimum 10 required)
```

**Solution**: Add more log files or ensure files contain valid log entries

### No Failures Found

```
Warning: No failures found in training data!
```

**Solution**: Add log files that contain failure cases (with `failure`, `error`, or HTTP error status codes)

### Memory Issues

If you have too many log files:

1. **Process in batches**: Split files into smaller directories
2. **Sample data**: Use a subset of files for initial training
3. **Increase system memory**: For very large datasets

## Example Workflow

```bash
# 1. Add training data
cp new_logs/*.txt data/logs/

# 2. Train basic model
python scripts/train.py

# 3. Train ML models
python scripts/train_ml_models.py --train-all

# 4. Verify models
ls -lh data/models/

# 5. Use for analysis
python scripts/analyze.py --logfile data/server.log
```

## Data Quality Checklist

Before training, ensure:

- [ ] Log files are properly formatted
- [ ] Headers are present (will be skipped automatically)
- [ ] Both success and failure cases are included
- [ ] Files are not corrupted
- [ ] Timestamps are consistent
- [ ] Sufficient data volume (100+ entries recommended)

## Next Steps

After training:

1. **Test Models**: Run analysis on test logs
2. **Evaluate Performance**: Check accuracy and precision
3. **Fine-tune**: Adjust model parameters if needed
4. **Deploy**: Use trained models in production analysis

