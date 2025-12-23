#!/usr/bin/env python3
"""
Convert Chat Interactions to Training Data
=========================================
Converts collected chat interactions into training data format compatible with training scripts.
"""

import sys
import json
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from log_analyzer.chat_data_collector import ChatDataCollector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Convert chat interactions to training data."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert chat interactions to training data')
    parser.add_argument('--output', type=Path,
                       default=project_root / 'data' / 'chat_training_data.json',
                       help='Output file for training data')
    parser.add_argument('--merge-with-logs', action='store_true',
                       help='Merge with existing log files in data/logs/')
    parser.add_argument('--output-log-format', action='store_true',
                       help='Output in log file format (for direct training)')
    
    args = parser.parse_args()
    
    # Initialize collector
    collector = ChatDataCollector(project_root / "data" / "chat_interactions")
    
    # Get stats
    stats = collector.get_stats()
    logger.info(f"Chat interaction stats: {stats}")
    
    if stats['total_interactions'] == 0:
        logger.warning("No chat interactions found. Nothing to convert.")
        return 1
    
    # Export training data
    logger.info(f"Exporting training data to {args.output}...")
    count = collector.export_for_training(args.output)
    
    if count == 0:
        logger.warning("No training examples extracted.")
        return 1
    
    logger.info(f"✓ Exported {count} training examples")
    
    # If merge requested, combine with existing log files
    if args.merge_with_logs:
        logs_dir = project_root / "data" / "logs"
        if logs_dir.exists():
            logger.info("Merging with existing log files...")
            # This would be handled by training scripts that read from both sources
            logger.info("Training scripts will automatically combine data from data/logs/ and chat data")
    
    # If log format requested, create log files
    if args.output_log_format:
        logger.info("Creating log file format output...")
        training_data = collector.get_training_data()
        
        # Group by component/operation for better organization
        log_output_dir = project_root / "data" / "logs" / "chat_training"
        log_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a combined log file
        combined_log = log_output_dir / "chat_interactions.log"
        with combined_log.open('w', encoding='utf-8') as f:
            for example in training_data:
                log_entry = example.get('log_entry', '')
                if log_entry:
                    f.write(log_entry + '\n')
        
        logger.info(f"✓ Created log file: {combined_log}")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

