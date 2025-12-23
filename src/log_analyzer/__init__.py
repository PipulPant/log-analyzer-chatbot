"""
Log Analyzer Package
====================
Enhanced log analyzer with supervised learning for failure detection.
"""

from .analyzer import LogAnalyzer, train_model, LogParser, SupervisedErrorClassifier
from .pattern_analysis import PatternAnalyzer, THRESHOLDS, FLOW_STATES, ERROR_KEYWORDS
from .ml_models import (
    ClassificationModel,
    AnomalyDetector,
    NLPModel,
    EnsembleDetector,
    FeatureExtractor
)
from .chat_data_collector import ChatDataCollector, ChatInteraction
from .intelligent_chat import IntelligentChatHandler

__version__ = "1.0.0"
__all__ = [
    'LogAnalyzer', 
    'train_model', 
    'LogParser', 
    'SupervisedErrorClassifier',
    'PatternAnalyzer',
    'THRESHOLDS',
    'FLOW_STATES',
    'ERROR_KEYWORDS',
    'ClassificationModel',
    'AnomalyDetector',
    'NLPModel',
    'EnsembleDetector',
    'FeatureExtractor',
    'ChatDataCollector',
    'ChatInteraction',
    'IntelligentChatHandler'
]

