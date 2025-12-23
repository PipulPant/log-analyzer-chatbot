# ML Models Guide

## Overview

The log analyzer now includes advanced ML models for error categorization, anomaly detection, and semantic analysis. This guide explains how to use these models.

## Available Models

### 1. Classification Models

**Supported Types:**
- **XGBoost** (recommended): Fast, accurate, handles non-linear patterns
- **Random Forest**: Robust, interpretable, good for mixed data types
- **Gradient Boosting**: High accuracy, good for complex patterns
- **SVM**: Good for high-dimensional data

**Features Extracted:**
- Temporal: hour_of_day, day_of_week, is_business_hours
- Performance: process_time_ms, queue_time_ms, time_since_last_log_ms
- Content: has_http_call, has_database_call, has_external_service
- Pattern: error_keyword_count, severity_score
- Context: previous_log_was_error, next_log_exists
- Sequence: position_in_flow
- Encoded: component_type, relationship_type

**Usage:**
```python
from log_analyzer import ClassificationModel, LogAnalyzer

# Train model
analyzer = LogAnalyzer()
entries, _ = analyzer._process_log_file(log_file, stats, None)
labels = [1 if e.rule_fail else 0 for e in entries]

model = ClassificationModel(model_type='xgboost')
model.train(entries, labels)
model.save(Path('models/classification_xgboost.pkl'))

# Use for prediction
predictions, probabilities = model.predict(new_entries)
```

### 2. Anomaly Detection

**Model:** Isolation Forest

**Features:**
- Processing time deviation
- Queue time deviation
- Error frequency
- Sequence anomaly scores

**Usage:**
```python
from log_analyzer import AnomalyDetector

detector = AnomalyDetector(contamination=0.1)  # 10% expected anomalies
detector.train(entries)
detector.save(Path('models/anomaly_detector.pkl'))

# Detect anomalies
anomalies, scores = detector.predict(new_entries)
```

### 3. NLP Models

**Model:** Sentence Transformers (all-MiniLM-L6-v2)

**Capabilities:**
- Semantic similarity matching
- Error pattern comparison
- Text embedding generation

**Usage:**
```python
from log_analyzer import NLPModel

nlp = NLPModel()

# Calculate similarity
similarity = nlp.similarity(
    "Failed to connect to database",
    "Connection refused to DB"
)

# Find similar errors
similar_indices = nlp.find_similar(
    query_text="Timeout error",
    reference_texts=["Request timeout", "Connection timeout", "Success"],
    threshold=0.7
)
```

### 4. Ensemble Detector

**Combines:** Classification + Anomaly Detection + NLP + Rule-based

**Default Weights:**
- Classification: 0.3
- Anomaly: 0.3
- NLP: 0.2
- Rule-based: 0.2

**Usage:**
```python
from log_analyzer import EnsembleDetector, ClassificationModel, AnomalyDetector

ensemble = EnsembleDetector()

# Add models
cls_model = ClassificationModel(model_type='xgboost')
cls_model.load(Path('models/classification_xgboost.pkl'))
ensemble.add_classification_model(cls_model)

anomaly_detector = AnomalyDetector()
anomaly_detector.load(Path('models/anomaly_detector.pkl'))
ensemble.add_anomaly_detector(anomaly_detector)

# Custom weights
ensemble.set_weights({
    'classification': 0.4,
    'anomaly': 0.3,
    'nlp': 0.2,
    'rule_based': 0.1
})

# Predict
results = ensemble.predict(entries, rule_based_scores=[e.severity_score for e in entries])
```

## Training Models

### Quick Start

```bash
# Train all models
python scripts/train_ml_models.py --train-all --logfile data/server.log

# Train specific models
python scripts/train_ml_models.py --train-classification --model-type xgboost
python scripts/train_ml_models.py --train-anomaly --anomaly-contamination 0.1
python scripts/train_ml_models.py --train-ensemble
```

### Training on Multiple Log Files

```bash
# Train on all logs in data/logs/
python scripts/train_ml_models.py --train-all --logs-dir data/logs
```

### Advanced Training

```python
from log_analyzer import ClassificationModel, AnomalyDetector, EnsembleDetector
from pathlib import Path

# Prepare data
entries = [...]  # Your log entries
labels = [1 if e.rule_fail else 0 for e in entries]

# Train classification
cls_model = ClassificationModel(model_type='xgboost')
cls_model.train(entries, labels)
cls_model.save(Path('models/cls.pkl'))

# Train anomaly detector
anomaly = AnomalyDetector(contamination=0.1)
anomaly.train(entries)
anomaly.save(Path('models/anomaly.pkl'))

# Create ensemble
ensemble = EnsembleDetector()
ensemble.add_classification_model(cls_model)
ensemble.add_anomaly_detector(anomaly)
```

## Using Models in Analysis

### Option 1: Via Configuration File

Create `ml_config.json`:
```json
{
  "classification_model_path": "data/models/classification_xgboost.pkl",
  "anomaly_detector_path": "data/models/anomaly_detector.pkl",
  "classification_type": "xgboost",
  "anomaly_contamination": 0.1,
  "ensemble_weights": {
    "classification": 0.3,
    "anomaly": 0.3,
    "nlp": 0.2,
    "rule_based": 0.2
  }
}
```

Use in analyzer:
```python
import json
from log_analyzer import LogAnalyzer
from pathlib import Path

with open('ml_config.json') as f:
    ml_config = json.load(f)

analyzer = LogAnalyzer(ml_models_config=ml_config)
results = analyzer.analyze(Path('data/server.log'))
```

### Option 2: Programmatic Usage

```python
from log_analyzer import LogAnalyzer, ClassificationModel, AnomalyDetector
from pathlib import Path

# Load models
cls_model = ClassificationModel()
cls_model.load(Path('models/classification_xgboost.pkl'))

anomaly = AnomalyDetector()
anomaly.load(Path('models/anomaly_detector.pkl'))

# Configure analyzer
ml_config = {
    'classification_model_path': 'models/classification_xgboost.pkl',
    'anomaly_detector_path': 'models/anomaly_detector.pkl'
}

analyzer = LogAnalyzer(ml_models_config=ml_config)
results = analyzer.analyze(Path('data/server.log'))
```

## Model Comparison

| Model | Use Case | Pros | Cons |
|-------|----------|------|------|
| XGBoost | Classification | Fast, accurate, handles non-linear | Requires tuning |
| Random Forest | Classification | Robust, interpretable | Can overfit |
| Isolation Forest | Anomaly Detection | Unsupervised, fast | Needs contamination parameter |
| Sentence Transformers | NLP | Semantic understanding | Requires GPU for large scale |

## Best Practices

1. **Start Simple**: Begin with XGBoost + Isolation Forest
2. **Train on Representative Data**: Include both failures and successes
3. **Validate Models**: Use cross-validation or hold-out test set
4. **Monitor Performance**: Track accuracy, precision, recall
5. **Retrain Periodically**: Models degrade over time
6. **Use Ensemble**: Combine multiple models for better accuracy

## Feature Engineering

The `FeatureExtractor` class automatically extracts:

- **Temporal Features**: Time-based patterns
- **Performance Features**: Processing and queue times
- **Content Features**: HTTP, database, external service indicators
- **Pattern Features**: Error keywords and severity
- **Context Features**: Previous/next log context
- **Sequence Features**: Position in transaction flow

Custom features can be added by extending `FeatureExtractor`.

## Troubleshooting

### XGBoost Not Available

```bash
pip install xgboost
```

### Sentence Transformers Not Available

```bash
pip install sentence-transformers torch
```

### Low Accuracy

1. Check label distribution (need both failures and successes)
2. Increase training data
3. Try different model types
4. Tune hyperparameters

### Memory Issues

1. Reduce batch size
2. Use smaller models (Random Forest instead of XGBoost)
3. Process logs in chunks

## Advanced: Custom Models

You can extend the base classes to add custom models:

```python
from log_analyzer.ml_models import ClassificationModel

class CustomModel(ClassificationModel):
    def _create_model(self):
        # Your custom model
        return YourCustomModel()
```

## Future Enhancements

Planned additions:
- LSTM for sequence modeling
- Autoencoders for unsupervised learning
- BERT fine-tuning for semantic analysis
- Graph Neural Networks for component relationships
- Time series models (Prophet, LSTM-Autoencoder)
- Reinforcement Learning for adaptive thresholds

## Examples

See `scripts/train_ml_models.py` for complete training examples.

## Support

For issues:
1. Check model files exist and are valid
2. Verify dependencies are installed
3. Check log format compatibility
4. Review training data quality

