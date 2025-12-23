#!/usr/bin/env python3
"""
Train ML Models for Log Analysis
=================================
Train classification models, anomaly detectors, and ensemble models.
"""

import sys
import json
from pathlib import Path
from typing import List, Optional
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from log_analyzer import LogAnalyzer
from log_analyzer.analyzer import LogEntry
from log_analyzer.ml_models import ClassificationModel, AnomalyDetector, EnsembleDetector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def prepare_labels(entries: List[LogEntry]) -> List[int]:
    """Prepare labels from log entries (1 = failure, 0 = success)."""
    return [1 if entry.rule_fail else 0 for entry in entries]

def train_classification_model(
    entries: List[LogEntry],
    labels: List[int],
    model_type: str = 'xgboost',
    output_path: Path = None
) -> ClassificationModel:
    """Train a classification model."""
    logger.info(f"Training {model_type} classification model...")
    
    model = ClassificationModel(model_type=model_type)
    metrics = model.train(entries, labels)
    
    logger.info(f"Training complete! Accuracy: {metrics.get('accuracy', 'N/A')}")
    
    if output_path:
        model.save(output_path)
        logger.info(f"Model saved to {output_path}")
    
    return model

def train_anomaly_detector(
    entries: List[LogEntry],
    contamination: float = 0.1,
    output_path: Path = None
) -> AnomalyDetector:
    """Train an anomaly detector."""
    logger.info(f"Training anomaly detector (contamination={contamination})...")
    
    detector = AnomalyDetector(contamination=contamination)
    detector.train(entries)
    
    if output_path:
        detector.save(output_path)
        logger.info(f"Detector saved to {output_path}")
    
    return detector

def train_ensemble(
    entries: List[LogEntry],
    labels: List[int],
    models_dir: Path,
    ensemble_config: Optional[dict] = None
) -> EnsembleDetector:
    """Train and configure ensemble detector."""
    logger.info("Training ensemble detector...")
    
    ensemble_config = ensemble_config or {}
    
    # Train classification model
    cls_model = None
    if ensemble_config.get('use_classification', True):
        cls_type = ensemble_config.get('classification_type', 'xgboost')
        cls_path = models_dir / f"classification_{cls_type}.pkl"
        
        cls_model = train_classification_model(
            entries, labels, cls_type, cls_path
        )
    
    # Train anomaly detector
    anomaly_detector = None
    if ensemble_config.get('use_anomaly', True):
        contamination = ensemble_config.get('anomaly_contamination', 0.1)
        anomaly_path = models_dir / "anomaly_detector.pkl"
        
        anomaly_detector = train_anomaly_detector(
            entries, contamination, anomaly_path
        )
    
    # Create ensemble
    ensemble = EnsembleDetector()
    if cls_model:
        ensemble.add_classification_model(cls_model)
    if anomaly_detector:
        ensemble.add_anomaly_detector(anomaly_detector)
    
    # Set weights
    if ensemble_config.get('weights'):
        ensemble.set_weights(ensemble_config['weights'])
    
    # Save ensemble config with relative paths
    config_path = models_dir / "ensemble_config.json"
    
    # Convert absolute paths to relative paths from project root
    project_root = Path(__file__).parent.parent
    def make_relative(path):
        if path is None:
            return None
        try:
            return str(Path(path).relative_to(project_root))
        except ValueError:
            # If path is already relative or outside project, return as-is
            return str(path)
    
    config_data = {
        'classification_model_path': make_relative(str(cls_path)) if cls_model else None,
        'anomaly_detector_path': make_relative(str(anomaly_path)) if anomaly_detector else None,
        'classification_type': ensemble_config.get('classification_type', 'xgboost'),
        'anomaly_contamination': ensemble_config.get('anomaly_contamination', 0.1),
        'ensemble_weights': ensemble.weights
    }
    
    with config_path.open('w') as f:
        json.dump(config_data, f, indent=2)
    
    logger.info(f"Ensemble configuration saved to {config_path}")
    
    return ensemble

def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train ML models for log analysis')
    parser.add_argument('--logfile', type=Path,
                       default=None,
                       help='Single log file to train on (optional if --logs-dir provided)')
    parser.add_argument('--logs-dir', type=Path,
                       default=project_root / "data" / "logs",
                       help='Directory with training log files (default: data/logs)')
    parser.add_argument('--models-dir', type=Path,
                       default=project_root / "data" / "models",
                       help='Directory to save models')
    parser.add_argument('--model-type', choices=['xgboost', 'random_forest', 'gradient_boosting', 'svm'],
                       default='xgboost',
                       help='Classification model type')
    parser.add_argument('--anomaly-contamination', type=float, default=0.1,
                       help='Expected anomaly percentage (0.0-1.0)')
    parser.add_argument('--train-classification', action='store_true',
                       help='Train classification model')
    parser.add_argument('--train-anomaly', action='store_true',
                       help='Train anomaly detector')
    parser.add_argument('--train-ensemble', action='store_true',
                       help='Train ensemble detector')
    parser.add_argument('--train-all', action='store_true',
                       help='Train all models')
    
    args = parser.parse_args()
    
    # Create models directory
    args.models_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare training data
    logger.info("Preparing training data...")
    analyzer = LogAnalyzer()
    
    # Determine which log file(s) to use
    combined_log = None
    
    # Priority: logs-dir > logfile
    if args.logs_dir.exists() and args.logs_dir.is_dir():
        # Support both .txt and .log files
        txt_files = sorted(args.logs_dir.glob("*.txt"))
        log_files_ext = sorted(args.logs_dir.glob("*.log"))
        log_files = txt_files + log_files_ext
        if log_files:
            txt_count = len(txt_files)
            log_count = len(log_files_ext)
            logger.info(f"Found {txt_count} .txt files and {log_count} .log files ({len(log_files)} total) in {args.logs_dir}")
            
            # Combine all log files into one temporary file
            combined_log = args.models_dir / "combined_training_logs.txt"
            logger.info(f"Combining {len(log_files)} log files...")
            
            total_lines = 0
            with combined_log.open('w', encoding='utf-8') as out:
                for log_file in log_files:
                    try:
                        with log_file.open('r', encoding='utf-8', errors='replace') as f:
                            for line in f:
                                line = line.strip()
                                if not line:
                                    continue
                                # Skip header lines
                                if line.startswith('TIMESTAMP|'):
                                    continue
                                # Skip empty lines or invalid lines
                                if not line or line.count('|') < 5:
                                    continue
                                out.write(line + '\n')
                                total_lines += 1
                    except Exception as e:
                        logger.warning(f"Error reading {log_file}: {e}")
                        continue
            
            logger.info(f"Combined {total_lines} log lines from {len(log_files)} files")
            args.logfile = combined_log
        else:
            logger.warning(f"No .txt or .log files found in {args.logs_dir}")
    
    # Fallback to single logfile if provided
    if not args.logfile or not args.logfile.exists():
        if args.logfile:
            logger.error(f"Log file not found: {args.logfile}")
        else:
            logger.error("No log file or logs directory provided!")
        return 1
    
    # Parse logs
    from log_analyzer.analyzer import AnalysisStats
    stats = AnalysisStats()
    entries, _ = analyzer._process_log_file(args.logfile, stats, None)
    
    # Clean up temporary combined log file
    if combined_log and combined_log.exists():
        combined_log.unlink()
        logger.debug(f"Cleaned up temporary combined log file")
    
    if len(entries) < 10:
        logger.error(f"Insufficient data: {len(entries)} entries (minimum 10 required)")
        return 1
    
    # Prepare labels
    labels = prepare_labels(entries)
    failure_count = sum(labels)
    success_count = len(labels) - failure_count
    
    logger.info(f"Training data: {len(entries)} entries ({failure_count} failures, {success_count} successes)")
    
    if failure_count == 0:
        logger.warning("No failures found in training data!")
    
    # Train models - always train all when --train-all is used
    train_all = args.train_all or (not args.train_classification and not args.train_anomaly and not args.train_ensemble)
    
    logger.info("=" * 80)
    logger.info("TRAINING ALL MODELS")
    logger.info("=" * 80)
    
    # Always train classification model
    if train_all or args.train_classification:
        logger.info("\n[1/3] Training Classification Model...")
        cls_path = args.models_dir / f"classification_{args.model_type}.pkl"
        train_classification_model(entries, labels, args.model_type, cls_path)
    else:
        logger.info("\n[1/3] Skipping Classification Model (not requested)")
    
    # Always train anomaly detector
    if train_all or args.train_anomaly:
        logger.info("\n[2/3] Training Anomaly Detector...")
        anomaly_path = args.models_dir / "anomaly_detector.pkl"
        train_anomaly_detector(entries, args.anomaly_contamination, anomaly_path)
    else:
        logger.info("\n[2/3] Skipping Anomaly Detector (not requested)")
    
    # Always train ensemble (combines all models)
    if train_all or args.train_ensemble:
        logger.info("\n[3/3] Training Ensemble Detector...")
        ensemble_config = {
            'use_classification': True,
            'use_anomaly': True,
            'classification_type': args.model_type,
            'anomaly_contamination': args.anomaly_contamination
        }
        train_ensemble(entries, labels, args.models_dir, ensemble_config)
    else:
        logger.info("\n[3/3] Skipping Ensemble Detector (not requested)")
    
    # Check NLP model availability
    logger.info("\n[4/4] Checking NLP Model...")
    try:
        from log_analyzer.ml_models import NLPModel
        nlp = NLPModel()
        nlp._load_model()
        if nlp.is_loaded:
            logger.info("✓ NLP Model (Sentence Transformers) is available and will be used during analysis")
        else:
            logger.info("⚠ NLP Model (Sentence Transformers) not available - install with: pip install sentence-transformers")
            logger.info("  Analysis will continue without NLP features")
    except Exception as e:
        logger.debug(f"NLP model check failed: {e}")
    
    logger.info("=" * 80)
    logger.info("Training complete!")
    logger.info("=" * 80)
    logger.info(f"Models saved to: {args.models_dir}")
    
    # Show what was trained
    trained_models = []
    if train_all or args.train_classification:
        trained_models.append(f"classification_{args.model_type}.pkl")
    if train_all or args.train_anomaly:
        trained_models.append("anomaly_detector.pkl")
    if train_all or args.train_ensemble:
        trained_models.append("ensemble_config.json")
    
    if trained_models:
        logger.info(f"\nTrained models:")
        for model in trained_models:
            logger.info(f"  - {model}")
    
    logger.info("\nUsage:")
    logger.info(f"  # Using ensemble config:")
    logger.info(f"  python scripts/analyze.py --logfile <log_file>")
    logger.info(f"  # Or programmatically:")
    logger.info(f"  from log_analyzer import LogAnalyzer")
    logger.info(f"  import json")
    logger.info(f"  with open('{args.models_dir / 'ensemble_config.json'}') as f:")
    logger.info(f"      ml_config = json.load(f)")
    logger.info(f"  analyzer = LogAnalyzer(ml_models_config=ml_config)")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

