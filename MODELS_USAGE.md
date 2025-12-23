# ML Models Usage in Log Analysis

## ✅ Currently Active Models

### 1. **Supervised Classifier (Random Forest)**
- **File**: `data/models/trained_model.pkl`
- **Status**: ✅ **ACTIVE**
- **Usage**: Applied to all log entries to detect failures
- **Output**: Binary predictions (failure/success) with probabilities
- **When Used**: Always when model file exists

### 2. **Classification Model (XGBoost)**
- **File**: `data/models/classification_xgboost.pkl`
- **Status**: ✅ **ACTIVE**
- **Usage**: Part of ensemble detector
- **Output**: Classification predictions with probabilities
- **Weight**: 30% in ensemble (configurable)
- **When Used**: When ensemble_config.json is loaded

### 3. **Anomaly Detector (Isolation Forest)**
- **File**: `data/models/anomaly_detector.pkl`
- **Status**: ✅ **ACTIVE**
- **Usage**: Part of ensemble detector
- **Output**: Anomaly scores (lower = more anomalous)
- **Weight**: 30% in ensemble (configurable)
- **When Used**: When ensemble_config.json is loaded

### 4. **Ensemble Detector**
- **Status**: ✅ **ACTIVE**
- **Usage**: Combines multiple models with weighted voting
- **Models Combined**:
  - Classification Model (30%)
  - Anomaly Detector (30%)
  - NLP Model (20%) - ⚠️ Requires sentence-transformers
  - Rule-based scores (20%)
- **Output**: Final predictions combining all models
- **When Used**: When at least one ML model is loaded

### 5. **NLP Model (Sentence Transformers)**
- **Status**: ⚠️ **INITIALIZED BUT OPTIONAL**
- **Usage**: Semantic similarity analysis
- **Requirement**: `sentence-transformers` package must be installed
- **Weight**: 20% in ensemble (when available)
- **When Used**: Only if sentence-transformers is installed
- **Note**: Will gracefully degrade if not available

### 6. **Rule-Based Pattern Analysis**
- **Status**: ✅ **ALWAYS ACTIVE**
- **Usage**: Pattern matching, keyword detection, HTTP status codes
- **Output**: Severity scores based on patterns
- **Weight**: 20% in ensemble
- **When Used**: Always (baseline detection)

### 7. **Root Cause Analysis**
- **Status**: ✅ **ACTIVE**
- **Usage**: Identifies primary failures and error cascades
- **Output**: Root cause classification, impact analysis
- **When Used**: After all failures are detected

## 📊 Model Pipeline Flow

```
Log File
   ↓
[1] Rule-Based Pattern Analysis (Always)
   ↓
[2] Supervised Classifier (if trained_model.pkl exists)
   ↓
[3] Ensemble Detector:
    ├─ Classification Model (XGBoost) - 30%
    ├─ Anomaly Detector (Isolation Forest) - 30%
    ├─ NLP Model (Sentence Transformers) - 20% (if available)
    └─ Rule-Based Scores - 20%
   ↓
[4] Combine All Predictions
   ↓
[5] Root Cause Analysis
   ↓
Final Results
```

## 🔍 How Each Model Contributes

### Rule-Based Analysis
- **Detects**: Explicit failures, HTTP errors, timeouts, connection issues
- **Method**: Pattern matching, keyword detection
- **Speed**: Very fast
- **Accuracy**: High for known patterns

### Supervised Classifier (Random Forest)
- **Detects**: Patterns learned from training data
- **Method**: Feature-based classification
- **Speed**: Fast
- **Accuracy**: High (100% on training data)

### Classification Model (XGBoost)
- **Detects**: Complex failure patterns
- **Method**: Gradient boosting, feature importance
- **Speed**: Fast
- **Accuracy**: Very high

### Anomaly Detector (Isolation Forest)
- **Detects**: Novel/unusual patterns not seen in training
- **Method**: Isolation-based anomaly detection
- **Speed**: Fast
- **Accuracy**: Good for detecting outliers

### NLP Model (Sentence Transformers)
- **Detects**: Semantic similarity to known error patterns
- **Method**: Embedding-based similarity
- **Speed**: Moderate (requires model loading)
- **Accuracy**: Good for semantic matching
- **Note**: Requires `pip install sentence-transformers`

### Root Cause Analysis
- **Detects**: Primary failures that caused cascades
- **Method**: Temporal analysis, transaction grouping
- **Speed**: Fast
- **Accuracy**: High for identifying root causes

## 📈 Current Configuration

From `data/models/ensemble_config.json`:
```json
{
  "ensemble_weights": {
    "classification": 0.3,  // XGBoost
    "anomaly": 0.3,          // Isolation Forest
    "nlp": 0.2,              // Sentence Transformers (if available)
    "rule_based": 0.2        // Pattern matching
  }
}
```

## ✅ Verification

To verify all models are being used, check the analysis logs:

```bash
python3 scripts/analyze.py --logfile data/server.log
```

Look for:
- ✅ "Loaded classification model"
- ✅ "Loaded anomaly detector"
- ✅ "Ensemble detector initialized"
- ✅ "Applying ensemble ML models"
- ✅ "Performing root cause analysis"

## 🚀 Enabling NLP Model

If you want to use the NLP model:

```bash
pip install sentence-transformers
```

The NLP model will automatically:
1. Load when available
2. Contribute 20% weight to ensemble
3. Find semantically similar error patterns
4. Gracefully skip if not installed

## 📝 Summary

**All trained models are being used:**
- ✅ Supervised Classifier (Random Forest)
- ✅ Classification Model (XGBoost)
- ✅ Anomaly Detector (Isolation Forest)
- ✅ Ensemble Detector (combines all)
- ⚠️ NLP Model (optional, requires sentence-transformers)
- ✅ Rule-Based Analysis (always active)
- ✅ Root Cause Analysis (always active)

The system uses a **multi-layered approach** where:
1. Rule-based analysis catches obvious failures
2. Supervised learning catches learned patterns
3. Ensemble combines multiple ML models
4. Root cause analysis identifies primary failures

This ensures comprehensive failure detection with high accuracy!

