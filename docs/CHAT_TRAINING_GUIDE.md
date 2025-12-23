# Chat Data Collection & Model Fine-Tuning Guide

## Overview

The Log Analyzer now includes **automatic data collection** from chat interactions. Every analysis performed through the web interface is automatically saved and can be used to fine-tune the ML models, enabling continuous learning and improvement.

## How It Works

### 1. Automatic Data Collection

Every time a user:
- Uploads a log file
- Pastes log content in the chat
- Receives an analysis response

The system automatically saves:
- User query/log content
- Full analysis results
- Formatted response
- Timestamp
- Failure details

### 2. Data Storage

Chat interactions are stored in:
```
data/chat_interactions/chat_interactions.jsonl
```

Format: JSON Lines (one JSON object per line) for efficient appending.

### 3. Training Data Extraction

The collected data is automatically converted to training format:
- Log entries extracted from failures
- Labels (failure/success) from analysis results
- Features (component, operation, patterns, etc.)
- Metadata (timestamps, transaction IDs, etc.)

## Usage

### View Chat Statistics

**Via API:**
```bash
curl http://localhost:5000/api/chat/stats
```

**Response:**
```json
{
  "success": true,
  "stats": {
    "total_interactions": 150,
    "total_logs_analyzed": 150,
    "total_failures_detected": 450,
    "data_file": "data/chat_interactions/chat_interactions.jsonl",
    "data_file_size_mb": 2.5
  }
}
```

### Export Training Data

**Via API:**
```bash
curl -X POST http://localhost:5000/api/training/export
```

**Via Script:**
```bash
python3 scripts/convert_chat_to_training.py
```

This creates:
```
data/chat_training_data.json
```

### Convert to Log Format

To convert chat data to log file format (for direct training):

```bash
python3 scripts/convert_chat_to_training.py --output-log-format
```

This creates:
```
data/logs/chat_training/chat_interactions.log
```

### Retrain Models with Chat Data

**Option 1: Automatic (Recommended)**

The training scripts automatically include chat data when available:

```bash
python3 scripts/train_ml_models.py --train-all
```

**Option 2: Manual Merge**

```bash
# 1. Convert chat data to log format
python3 scripts/convert_chat_to_training.py --output-log-format

# 2. Train with all data (logs + chat)
python3 scripts/train_ml_models.py --train-all
```

## Data Structure

### Chat Interaction Format

```json
{
  "timestamp": "2025-12-23T17:30:00",
  "user_query": "Uploaded file: server.log",
  "log_content": "2024-01-01 10:00:00|ERROR|...",
  "log_filename": "server.log",
  "analysis_results": {
    "stats": {...},
    "failures": [...],
    "recommendations": [...]
  },
  "response_text": "## Analysis Summary\n...",
  "user_feedback": null,
  "is_useful": null
}
```

### Training Data Format

```json
{
  "source": "chat_interactions",
  "total_examples": 450,
  "export_timestamp": "2025-12-23T17:35:00",
  "training_data": [
    {
      "log_entry": "2024-01-01 10:00:00|ERROR|...",
      "component": "IHTTP",
      "operation": "process_request",
      "level": "ERROR",
      "is_failure": true,
      "severity": 0.95,
      "error_patterns": {...},
      "keywords": ["error", "failed"],
      "timestamp": "2024-01-01 10:00:00",
      "transaction_id": "txn-123",
      "source": "chat_interaction",
      "interaction_timestamp": "2025-12-23T17:30:00"
    }
  ]
}
```

## Best Practices

### 1. Regular Retraining

Retrain models periodically to incorporate new patterns:

```bash
# Weekly retraining
python3 scripts/train_ml_models.py --train-all
```

### 2. Data Cleanup

Remove old interactions (older than 90 days):

```python
from log_analyzer import ChatDataCollector
from pathlib import Path

collector = ChatDataCollector(Path("data/chat_interactions"))
removed = collector.clear_old_data(days=90)
print(f"Removed {removed} old interactions")
```

### 3. Monitor Data Quality

Check statistics regularly:

```bash
curl http://localhost:5000/api/chat/stats
```

### 4. Export Before Major Updates

Before updating models, export training data:

```bash
python3 scripts/convert_chat_to_training.py --output data/backup_training_data.json
```

## Integration with Training Pipeline

The chat data is automatically integrated:

1. **Data Collection**: Every chat interaction is saved
2. **Data Extraction**: Training examples are extracted from interactions
3. **Training**: Models are trained on combined data (logs + chat)
4. **Improvement**: Models improve with each interaction

## API Endpoints

### GET `/api/chat/stats`
Get statistics about collected chat interactions.

**Response:**
```json
{
  "success": true,
  "stats": {
    "total_interactions": 150,
    "total_logs_analyzed": 150,
    "total_failures_detected": 450,
    "data_file": "...",
    "data_file_size_mb": 2.5
  }
}
```

### POST `/api/training/export`
Export chat interactions as training data.

**Response:**
```json
{
  "success": true,
  "exported_examples": 450,
  "output_file": "data/chat_training_data.json",
  "message": "Exported 450 training examples"
}
```

## Privacy & Security

- **Data Storage**: All chat data is stored locally
- **No External Sharing**: Data never leaves your system
- **User Control**: You can delete old data anytime
- **Secure**: Data files are stored in `data/chat_interactions/`

## Troubleshooting

### No Data Collected

Check if the collector is initialized:
```python
from log_analyzer import ChatDataCollector
collector = ChatDataCollector(Path("data/chat_interactions"))
stats = collector.get_stats()
print(stats)
```

### Export Fails

Ensure the output directory exists:
```bash
mkdir -p data/
```

### Training Doesn't Use Chat Data

Make sure chat data is in the correct format:
```bash
python3 scripts/convert_chat_to_training.py --output-log-format
```

## Next Steps

1. **Start Using**: Just use the web interface - data is collected automatically
2. **Monitor**: Check stats regularly via API
3. **Retrain**: Retrain models weekly or monthly
4. **Improve**: Models will improve with each interaction!

## Example Workflow

```bash
# 1. Use web interface (data collected automatically)
# ... analyze logs through chat ...

# 2. Check statistics
curl http://localhost:5000/api/chat/stats

# 3. Export training data
python3 scripts/convert_chat_to_training.py --output-log-format

# 4. Retrain models
python3 scripts/train_ml_models.py --train-all

# 5. Models are now improved with chat data!
```

## Benefits

✅ **Continuous Learning**: Models improve with every interaction  
✅ **Real-World Data**: Training on actual usage patterns  
✅ **Automatic Collection**: No manual data entry required  
✅ **Easy Integration**: Works seamlessly with existing training pipeline  
✅ **Privacy-First**: All data stays local  

