# All Models Training & Usage Guide

## ✅ Complete Model Training & Analysis Pipeline

### Step 1: Train All Models

```bash
# Train all ML models (Classification, Anomaly, Ensemble)
python3 scripts/train_ml_models.py --train-all --logs-dir data/logs
```

**What Gets Trained:**
1. ✅ **Classification Model (XGBoost)** → `classification_xgboost.pkl`
2. ✅ **Anomaly Detector (Isolation Forest)** → `anomaly_detector.pkl`
3. ✅ **Ensemble Configuration** → `ensemble_config.json`
4. ✅ **NLP Model Check** → Verifies if sentence-transformers is available

### Step 2: Analysis Uses All Models

When you run analysis, **ALL available models are automatically used**:

```bash
python3 scripts/analyze.py --logfile data/server.log
```

**Models Used During Analysis:**
1. ✅ **Supervised Classifier (Random Forest)** - From `trained_model.pkl`
2. ✅ **Classification Model (XGBoost)** - From `classification_xgboost.pkl`
3. ✅ **Anomaly Detector (Isolation Forest)** - From `anomaly_detector.pkl`
4. ✅ **NLP Model (Sentence Transformers)** - If `sentence-transformers` installed
5. ✅ **Rule-Based Analysis** - Always active
6. ✅ **Root Cause Analysis** - Always active

## 📊 Model Usage Verification

During analysis, you'll see:
```
INFO - Using models: Classification (xgboost), Anomaly Detection, Rule-Based Analysis
INFO - Ensemble ML models found X additional potential failures
```

If NLP is available:
```
INFO - Using models: Classification (xgboost), Anomaly Detection, NLP (Sentence Transformers), Rule-Based Analysis
```

## 🔍 Complete Analysis Pipeline

```
Log File
   ↓
[1] Rule-Based Pattern Analysis ✅ (Always Active)
   ↓
[2] Supervised Classifier ✅ (If trained_model.pkl exists)
   ↓
[3] Ensemble Detector ✅ (Combines all ML models)
    ├─ Classification Model (XGBoost) ✅ - 30% weight
    ├─ Anomaly Detector (Isolation Forest) ✅ - 30% weight
    ├─ NLP Model (Sentence Transformers) ✅ - 20% weight (if available)
    └─ Rule-Based Scores ✅ - 20% weight
   ↓
[4] Root Cause Analysis ✅ (Identifies primary failures)
   ↓
Final Results with All Models' Contributions
```

## 📝 Training Summary

After training, you'll see:
```
[1/3] Training Classification Model...
[2/3] Training Anomaly Detector...
[3/3] Training Ensemble Detector...
[4/4] Checking NLP Model...
✓ NLP Model (Sentence Transformers) is available and will be used during analysis
```

## 🎯 Key Features

### Automatic Model Detection
- ✅ Automatically loads all trained models
- ✅ Uses ensemble to combine predictions
- ✅ Gracefully handles missing models
- ✅ Logs which models are being used

### Model Weights (Configurable)
From `ensemble_config.json`:
```json
{
  "ensemble_weights": {
    "classification": 0.3,  // XGBoost
    "anomaly": 0.3,         // Isolation Forest
    "nlp": 0.2,             // Sentence Transformers (if available)
    "rule_based": 0.2       // Pattern matching
  }
}
```

### Comprehensive Detection
- **Rule-Based**: Catches explicit failures, HTTP errors, timeouts
- **Supervised Learning**: Learns from training data patterns
- **Classification**: XGBoost for complex pattern recognition
- **Anomaly Detection**: Finds novel/unusual patterns
- **NLP**: Semantic similarity to known error patterns
- **Root Cause**: Identifies primary failures and cascades

## 🚀 Quick Start

### 1. Train All Models
```bash
python3 scripts/train_ml_models.py --train-all
```

### 2. Analyze Logs (Uses All Models)
```bash
python3 scripts/analyze.py --logfile data/server.log
```

### 3. View Results
```bash
open reports/analysis_report.html
```

## ✅ Verification Checklist

After training and analysis, verify:

- [x] Classification model trained (`classification_xgboost.pkl`)
- [x] Anomaly detector trained (`anomaly_detector.pkl`)
- [x] Ensemble config created (`ensemble_config.json`)
- [x] All models loaded during analysis
- [x] "Using models: ..." log shows all active models
- [x] Ensemble predictions combine all models
- [x] Root cause analysis identifies primary failures

## 📈 Model Contributions

Each model contributes to final predictions:

1. **Rule-Based (20%)**: Fast, reliable for known patterns
2. **Classification (30%)**: High accuracy for learned patterns
3. **Anomaly (30%)**: Catches novel failures
4. **NLP (20%)**: Semantic understanding of error messages

**Total: 100%** - All models work together for comprehensive detection!

## 🎉 Result

**All models are trained and used automatically!**

The system ensures:
- ✅ All trained models are loaded
- ✅ All available models contribute to predictions
- ✅ Ensemble combines models with proper weights
- ✅ Root cause analysis identifies primary failures
- ✅ Comprehensive failure detection with high accuracy

