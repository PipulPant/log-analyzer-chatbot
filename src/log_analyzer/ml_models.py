#!/usr/bin/env python3
"""
Advanced ML Models for Log Analysis
===================================
Classification, anomaly detection, sequence models, and NLP models for comprehensive log analysis.
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# ==================== FEATURE ENGINEERING ====================

@dataclass
class FeatureExtractor:
    """Extract comprehensive features from log entries."""
    
    def extract_features(self, logs: List[Any], index: int = None) -> Dict[str, Any]:
        """Extract features from log entries."""
        if index is not None:
            return self._extract_single(logs, index)
        else:
            return [self._extract_single(logs, i) for i in range(len(logs))]
    
    def _extract_single(self, logs: List[Any], index: int) -> Dict[str, Any]:
        """Extract features for a single log entry."""
        log = logs[index]
        
        # Parse timestamp
        try:
            from datetime import datetime
            ts_str = getattr(log, 'ts', '')
            if ts_str:
                dt = datetime.strptime(ts_str.split(',')[0], '%Y-%m-%d %H:%M:%S')
                hour_of_day = dt.hour
                day_of_week = dt.weekday()
                is_business_hours = 1 if 9 <= hour_of_day <= 17 else 0
            else:
                hour_of_day = 0
                day_of_week = 0
                is_business_hours = 0
        except:
            hour_of_day = 0
            day_of_week = 0
            is_business_hours = 0
        
        # Performance features
        process_time = getattr(log, 'process_time_ms', 0) or 0
        queue_time = getattr(log, 'queue_time_ms', 0) or 0
        
        # Calculate time gap
        time_since_last = 0
        if index > 0:
            try:
                prev_ts = datetime.strptime(logs[index-1].ts.split(',')[0], '%Y-%m-%d %H:%M:%S')
                curr_ts = datetime.strptime(log.ts.split(',')[0], '%Y-%m-%d %H:%M:%S')
                time_since_last = (curr_ts - prev_ts).total_seconds() * 1000
            except:
                pass
        
        # Content features
        log_message = getattr(log, 'msg', '') or ''
        processor_name = getattr(log, 'processor_name', '') or ''
        component = getattr(log, 'component', '') or ''
        relationship = getattr(log, 'relationship', '') or ''
        
        full_text = f"{log_message} {processor_name} {component}".lower()
        
        # Pattern features
        error_keywords = getattr(log, 'keywords_found', set()) or set()
        error_keyword_count = len(error_keywords)
        
        # Context features
        previous_log_was_error = 0
        if index > 0:
            prev_log = logs[index-1]
            prev_relationship = getattr(prev_log, 'relationship', '').lower()
            prev_fail = getattr(prev_log, 'rule_fail', False)
            if prev_fail or 'failure' in prev_relationship or 'error' in prev_relationship:
                previous_log_was_error = 1
        
        next_log_exists = 1 if index < len(logs) - 1 else 0
        
        # Sequence features
        position_in_flow = index / max(len(logs), 1)
        
        # Component type encoding (simplified)
        component_type = self._encode_component_type(component)
        relationship_type = self._encode_relationship_type(relationship)
        
        # HTTP and database indicators
        has_http_call = 1 if any(x in full_text for x in ['http', 'ihttp', 'api', 'rest']) else 0
        has_database_call = 1 if any(x in full_text for x in ['sql', 'psql', 'esqlr', 'database', 'db']) else 0
        has_external_service = 1 if any(x in full_text for x in ['external', 'call', 'invoke']) else 0
        
        # HTTP status
        http_status = getattr(log, 'http_status', None)
        has_http_status = 1 if http_status is not None else 0
        
        # Error pattern categories
        error_patterns = getattr(log, 'error_patterns', {}) or {}
        has_error_keywords = 1 if error_patterns.get('categories') else 0
        
        feature_vector = {
            # Temporal features
            'hour_of_day': hour_of_day,
            'day_of_week': day_of_week,
            'is_business_hours': is_business_hours,
            
            # Performance features
            'process_time_ms': float(process_time),
            'queue_time_ms': float(queue_time),
            'time_since_last_log_ms': float(time_since_last),
            
            # Content features
            'has_http_call': has_http_call,
            'has_database_call': has_database_call,
            'has_external_service': has_external_service,
            'has_http_status': has_http_status,
            'has_error_keywords': has_error_keywords,
            
            # Pattern features
            'error_keyword_count': error_keyword_count,
            'severity_score': float(getattr(log, 'severity_score', 0) or 0),
            
            # Context features
            'previous_log_was_error': previous_log_was_error,
            'next_log_exists': next_log_exists,
            
            # Sequence features
            'position_in_flow': position_in_flow,
            
            # Encoded features
            'component_type': component_type,
            'relationship_type': relationship_type,
        }
        
        return feature_vector
    
    def _encode_component_type(self, component: str) -> int:
        """Encode component type to numeric value."""
        component_lower = component.lower()
        if 'http' in component_lower or 'ihttp' in component_lower:
            return 1
        elif 'sql' in component_lower or 'psql' in component_lower or 'esqlr' in component_lower:
            return 2
        elif 'pjms' in component_lower:
            return 3
        elif 'roa' in component_lower:
            return 4
        elif 'ua' in component_lower:
            return 5
        elif 'ejp' in component_lower:
            return 6
        else:
            return 0
    
    def _encode_relationship_type(self, relationship: str) -> int:
        """Encode relationship type to numeric value."""
        rel_lower = relationship.lower()
        if rel_lower == 'success':
            return 1
        elif 'failure' in rel_lower or 'error' in rel_lower:
            return 2
        elif 'no record found' in rel_lower:
            return 3
        elif 'no retry' in rel_lower:
            return 4
        elif 'matched' in rel_lower:
            return 5
        elif 'unmatched' in rel_lower:
            return 6
        else:
            return 0

# ==================== CLASSIFICATION MODELS ====================

class ClassificationModel:
    """Base class for classification models."""
    
    def __init__(self, model_type: str = 'xgboost'):
        self.model_type = model_type
        self.model = None
        self.feature_extractor = FeatureExtractor()
        self.is_trained = False
        
    def _create_model(self):
        """Create the appropriate model."""
        if self.model_type == 'xgboost':
            try:
                import xgboost as xgb
                return xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    eval_metric='logloss'
                )
            except ImportError:
                logger.warning("XGBoost not available, falling back to Random Forest")
                self.model_type = 'random_forest'
                return self._create_model()
        
        elif self.model_type == 'random_forest':
            try:
                from sklearn.ensemble import RandomForestClassifier
                return RandomForestClassifier(
                    n_estimators=100,
                    max_depth=20,
                    min_samples_split=5,
                    random_state=42,
                    n_jobs=-1,
                    class_weight='balanced'
                )
            except ImportError:
                raise ImportError("scikit-learn is required")
        
        elif self.model_type == 'gradient_boosting':
            try:
                from sklearn.ensemble import GradientBoostingClassifier
                return GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                )
            except ImportError:
                raise ImportError("scikit-learn is required")
        
        elif self.model_type == 'svm':
            try:
                from sklearn.svm import SVC
                return SVC(
                    kernel='rbf',
                    probability=True,
                    random_state=42,
                    class_weight='balanced'
                )
            except ImportError:
                raise ImportError("scikit-learn is required")
        
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def prepare_features(self, logs: List[Any]) -> np.ndarray:
        """Prepare feature matrix from logs."""
        features = self.feature_extractor.extract_features(logs)
        
        # Convert to numpy array
        feature_names = sorted(features[0].keys())
        X = np.array([[f[name] for name in feature_names] for f in features])
        
        return X
    
    def train(self, logs: List[Any], labels: List[int], 
              target_classes: Optional[List[str]] = None) -> Dict[str, Any]:
        """Train the classification model."""
        if not logs or not labels:
            raise ValueError("Logs and labels are required")
        
        if len(logs) != len(labels):
            raise ValueError(f"Mismatch: {len(logs)} logs but {len(labels)} labels")
        
        logger.info(f"Training {self.model_type} classifier on {len(logs)} samples...")
        
        # Prepare features
        X = self.prepare_features(logs)
        y = np.array(labels)
        
        # Create model
        self.model = self._create_model()
        
        # Train
        self.model.fit(X, y)
        self.is_trained = True
        
        # Evaluate
        try:
            from sklearn.metrics import accuracy_score, classification_report
            y_pred = self.model.predict(X)
            accuracy = accuracy_score(y, y_pred)
            
            logger.info(f"Training accuracy: {accuracy:.3f}")
            
            return {
                'accuracy': float(accuracy),
                'model_type': self.model_type,
                'n_samples': len(logs),
                'n_features': X.shape[1]
            }
        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")
            return {'model_type': self.model_type, 'n_samples': len(logs)}
    
    def predict(self, logs: List[Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Predict error categories."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X = self.prepare_features(logs)
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        return predictions, probabilities
    
    def save(self, path: Path):
        """Save the model."""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'is_trained': self.is_trained
        }
        
        with path.open('wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {path}")
    
    def load(self, path: Path):
        """Load the model."""
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        with path.open('rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.is_trained = model_data.get('is_trained', True)
        
        logger.info(f"Model loaded from {path}")

# ==================== ANOMALY DETECTION ====================

class AnomalyDetector:
    """Anomaly detection using Isolation Forest."""
    
    def __init__(self, contamination: float = 0.1):
        self.contamination = contamination
        self.detector = None
        self.feature_extractor = FeatureExtractor()
        self.is_trained = False
    
    def _create_detector(self):
        """Create Isolation Forest detector."""
        try:
            from sklearn.ensemble import IsolationForest
            return IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_jobs=-1
            )
        except ImportError:
            raise ImportError("scikit-learn is required for anomaly detection")
    
    def prepare_features(self, logs: List[Any]) -> np.ndarray:
        """Prepare features with anomaly-specific metrics."""
        features = self.feature_extractor.extract_features(logs)
        
        # Add anomaly-specific features
        process_times = [f['process_time_ms'] for f in features]
        queue_times = [f['queue_time_ms'] for f in features]
        
        if process_times:
            mean_process = np.mean(process_times)
            std_process = np.std(process_times) if len(process_times) > 1 else 1.0
            for i, f in enumerate(features):
                f['processing_time_deviation'] = abs(f['process_time_ms'] - mean_process) / max(std_process, 1.0)
        else:
            for f in features:
                f['processing_time_deviation'] = 0.0
        
        if queue_times:
            mean_queue = np.mean(queue_times)
            std_queue = np.std(queue_times) if len(queue_times) > 1 else 1.0
            for i, f in enumerate(features):
                f['queue_time_deviation'] = abs(f['queue_time_ms'] - mean_queue) / max(std_queue, 1.0)
        else:
            for f in features:
                f['queue_time_deviation'] = 0.0
        
        # Error frequency
        error_counts = [f['error_keyword_count'] for f in features]
        for i, f in enumerate(features):
            window_size = min(10, len(features))
            start = max(0, i - window_size // 2)
            end = min(len(features), i + window_size // 2)
            f['error_frequency'] = sum(error_counts[start:end]) / window_size
        
        # Convert to numpy array
        feature_names = sorted(features[0].keys())
        X = np.array([[f[name] for name in feature_names] for f in features])
        
        return X
    
    def train(self, logs: List[Any]):
        """Train the anomaly detector."""
        logger.info(f"Training anomaly detector on {len(logs)} samples...")
        
        X = self.prepare_features(logs)
        self.detector = self._create_detector()
        self.detector.fit(X)
        self.is_trained = True
        
        logger.info("Anomaly detector trained")
    
    def predict(self, logs: List[Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Predict anomalies."""
        if not self.is_trained:
            raise ValueError("Detector must be trained before prediction")
        
        X = self.prepare_features(logs)
        predictions = self.detector.predict(X)  # -1 for anomaly, 1 for normal
        scores = self.detector.score_samples(X)  # Lower score = more anomalous
        
        # Convert to 0/1 (0 = normal, 1 = anomaly)
        anomaly_predictions = (predictions == -1).astype(int)
        
        return anomaly_predictions, scores
    
    def save(self, path: Path):
        """Save the detector."""
        if not self.is_trained:
            raise ValueError("No trained detector to save")
        
        model_data = {
            'detector': self.detector,
            'contamination': self.contamination,
            'is_trained': self.is_trained
        }
        
        with path.open('wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Anomaly detector saved to {path}")
    
    def load(self, path: Path):
        """Load the detector."""
        if not path.exists():
            raise FileNotFoundError(f"Detector file not found: {path}")
        
        with path.open('rb') as f:
            model_data = pickle.load(f)
        
        self.detector = model_data['detector']
        self.contamination = model_data.get('contamination', 0.1)
        self.is_trained = model_data.get('is_trained', True)
        
        logger.info(f"Anomaly detector loaded from {path}")

# ==================== NLP MODELS ====================

class NLPModel:
    """NLP model for semantic analysis using Sentence Transformers."""
    
    def __init__(self):
        self.model = None
        self.is_loaded = False
    
    def _load_model(self):
        """Load Sentence Transformer model."""
        try:
            from sentence_transformers import SentenceTransformer
            # Use a lightweight model
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.is_loaded = True
            logger.info("Sentence Transformer model loaded")
        except ImportError:
            logger.warning("sentence-transformers not available. Install with: pip install sentence-transformers")
            self.model = None
        except Exception as e:
            logger.warning(f"Failed to load Sentence Transformer: {e}")
            self.model = None
    
    def encode(self, texts: List[str]) -> Optional[np.ndarray]:
        """Encode texts to embeddings."""
        if not self.is_loaded:
            self._load_model()
        
        if not self.model or not self.is_loaded:
            return None
        
        try:
            # Filter out empty texts
            valid_texts = [t if t else "" for t in texts]
            embeddings = self.model.encode(valid_texts, convert_to_numpy=True, show_progress_bar=False)
            return embeddings
        except Exception as e:
            logger.debug(f"Encoding failed: {e}")
            return None
    
    def similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts."""
        embeddings = self.encode([text1, text2])
        if embeddings is None or len(embeddings) < 2:
            return 0.0
        
        from sklearn.metrics.pairwise import cosine_similarity
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity)
    
    def find_similar(self, query_text: str, reference_texts: List[str], 
                    threshold: float = 0.7) -> List[Tuple[int, float]]:
        """Find similar texts to query."""
        all_texts = [query_text] + reference_texts
        embeddings = self.encode(all_texts)
        
        if embeddings is None:
            return []
        
        from sklearn.metrics.pairwise import cosine_similarity
        query_embedding = embeddings[0:1]
        reference_embeddings = embeddings[1:]
        
        similarities = cosine_similarity(query_embedding, reference_embeddings)[0]
        
        similar_indices = [
            (i, float(sim)) for i, sim in enumerate(similarities) 
            if sim >= threshold
        ]
        similar_indices.sort(key=lambda x: x[1], reverse=True)
        
        return similar_indices

# ==================== ENSEMBLE MODEL ====================

class EnsembleDetector:
    """Ensemble approach combining multiple models."""
    
    def __init__(self):
        self.classification_model = None
        self.anomaly_detector = None
        self.nlp_model = NLPModel()
        self.weights = {
            'classification': 0.3,
            'anomaly': 0.3,
            'nlp': 0.2,
            'rule_based': 0.2
        }
        self.is_ready = False
    
    def add_classification_model(self, model: ClassificationModel):
        """Add classification model."""
        self.classification_model = model
        self._check_ready()
    
    def add_anomaly_detector(self, detector: AnomalyDetector):
        """Add anomaly detector."""
        self.anomaly_detector = detector
        self._check_ready()
    
    def _check_ready(self):
        """Check if ensemble is ready."""
        self.is_ready = (
            self.classification_model is not None and 
            self.classification_model.is_trained
        ) or (
            self.anomaly_detector is not None and 
            self.anomaly_detector.is_trained
        )
    
    def predict(self, logs: List[Any], rule_based_scores: Optional[List[float]] = None) -> Dict[str, Any]:
        """Get ensemble predictions."""
        if not self.is_ready:
            raise ValueError("At least one model must be trained")
        
        predictions = {}
        scores = []
        
        # Classification model
        if self.classification_model and self.classification_model.is_trained:
            try:
                cls_pred, cls_proba = self.classification_model.predict(logs)
                predictions['classification'] = cls_pred
                scores.append(cls_proba[:, 1] * self.weights['classification'])  # Assuming binary classification
            except Exception as e:
                logger.warning(f"Classification prediction failed: {e}")
        
        # Anomaly detector
        if self.anomaly_detector and self.anomaly_detector.is_trained:
            try:
                anomaly_pred, anomaly_scores = self.anomaly_detector.predict(logs)
                predictions['anomaly'] = anomaly_pred
                # Convert scores to probabilities (lower score = higher anomaly probability)
                anomaly_proba = 1 - (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min() + 1e-10)
                scores.append(anomaly_proba * self.weights['anomaly'])
            except Exception as e:
                logger.warning(f"Anomaly detection failed: {e}")
        
        # NLP similarity - compare against known error patterns
        if self.nlp_model and self.nlp_model.is_loaded:
            try:
                # Extract error messages from logs
                error_messages = []
                for log in logs:
                    if hasattr(log, 'msg'):
                        error_messages.append(log.msg)
                    elif isinstance(log, dict):
                        error_messages.append(log.get('msg', ''))
                    else:
                        error_messages.append(str(log))
                
                # Known error patterns (from training data or common patterns)
                known_error_patterns = [
                    "connection refused", "connection timeout", "connection failed",
                    "404 not found", "500 internal server error", "503 service unavailable",
                    "authentication failed", "unauthorized", "forbidden",
                    "timeout", "timed out", "request timeout",
                    "no record found", "record not found", "not found",
                    "invalid", "validation failed", "bad request",
                    "exception", "error", "failure", "failed"
                ]
                
                # Calculate similarity scores
                nlp_scores = []
                if error_messages:
                    # Encode all messages at once for efficiency
                    message_embeddings = self.nlp_model.encode(error_messages)
                    pattern_embeddings = self.nlp_model.encode(known_error_patterns)
                    
                    if message_embeddings is not None and pattern_embeddings is not None:
                        from sklearn.metrics.pairwise import cosine_similarity
                        # Find max similarity to any error pattern for each message
                        similarities = cosine_similarity(message_embeddings, pattern_embeddings)
                        max_similarities = np.max(similarities, axis=1)
                        # Higher similarity = higher error probability
                        nlp_scores = max_similarities * self.weights['nlp']
                        scores.append(nlp_scores)
                        predictions['nlp'] = (max_similarities > 0.6).astype(int)  # Threshold for NLP
            except Exception as e:
                logger.warning(f"NLP prediction failed: {e}")
        
        # Rule-based scores
        if rule_based_scores:
            rule_scores = np.array(rule_based_scores)
            scores.append(rule_scores * self.weights['rule_based'])
        
        # Combine scores
        if scores:
            final_scores = np.sum(scores, axis=0)
            final_predictions = (final_scores > 0.5).astype(int)
        else:
            final_scores = np.zeros(len(logs))
            final_predictions = np.zeros(len(logs))
        
        return {
            'predictions': final_predictions,
            'scores': final_scores,
            'individual_predictions': predictions
        }
    
    def set_weights(self, weights: Dict[str, float]):
        """Set model weights."""
        self.weights.update(weights)
        # Normalize weights
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

