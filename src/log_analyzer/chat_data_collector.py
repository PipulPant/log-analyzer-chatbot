#!/usr/bin/env python3
"""
Chat Data Collector
===================
Collects chat interactions for model fine-tuning and continuous learning.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import threading

logger = logging.getLogger(__name__)

@dataclass
class ChatInteraction:
    """Represents a single chat interaction."""
    timestamp: str
    user_query: str
    log_content: str
    log_filename: Optional[str]
    analysis_results: Dict[str, Any]
    response_text: str
    user_feedback: Optional[str] = None
    is_useful: Optional[bool] = None
    conversation_id: Optional[str] = None  # Group interactions by conversation
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class ChatDataCollector:
    """Collects and manages chat interaction data for training."""
    
    def __init__(self, data_dir: Path):
        """
        Initialize the collector.
        
        Args:
            data_dir: Directory to store chat interaction data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.interactions_file = self.data_dir / "chat_interactions.jsonl"
        self.lock = threading.Lock()
        
        logger.info(f"Chat data collector initialized: {self.data_dir}")
    
    def save_interaction(self, user_query: str, log_content: str, 
                        log_filename: Optional[str], analysis_results: Dict[str, Any],
                        response_text: str, conversation_id: Optional[str] = None) -> None:
        """
        Save a chat interaction.
        
        Args:
            user_query: User's query/message
            log_content: Log content that was analyzed
            log_filename: Filename if file was uploaded
            analysis_results: Full analysis results
            response_text: Formatted response text
        """
        # Generate conversation ID if not provided
        if not conversation_id:
            conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        interaction = ChatInteraction(
            timestamp=datetime.now().isoformat(),
            user_query=user_query,
            log_content=log_content,
            log_filename=log_filename,
            analysis_results=analysis_results,
            response_text=response_text,
            conversation_id=conversation_id
        )
        
        with self.lock:
            try:
                # Append to JSONL file (one JSON object per line)
                with self.interactions_file.open('a', encoding='utf-8') as f:
                    f.write(json.dumps(interaction.to_dict(), ensure_ascii=False) + '\n')
                
                logger.debug(f"Saved chat interaction: {interaction.timestamp}")
            except Exception as e:
                logger.error(f"Failed to save chat interaction: {e}")
    
    def get_interactions(self, limit: Optional[int] = None) -> List[ChatInteraction]:
        """
        Get all interactions.
        
        Args:
            limit: Maximum number of interactions to return
            
        Returns:
            List of interactions
        """
        interactions = []
        
        if not self.interactions_file.exists():
            return interactions
        
        with self.lock:
            try:
                with self.interactions_file.open('r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            interactions.append(ChatInteraction(**data))
                
                # Return most recent first
                interactions.reverse()
                
                if limit:
                    interactions = interactions[:limit]
                
                return interactions
            except Exception as e:
                logger.error(f"Failed to load interactions: {e}")
                return []
    
    def get_training_data(self) -> List[Dict[str, Any]]:
        """
        Extract training data from interactions.
        
        Returns:
            List of training examples
        """
        interactions = self.get_interactions()
        training_data = []
        
        for interaction in interactions:
            # Extract log entries from analysis results
            failures = interaction.analysis_results.get('failures', [])
            stats = interaction.analysis_results.get('stats', {})
            
            # Create training examples from failures
            for failure in failures:
                training_example = {
                    'log_entry': failure.get('raw', failure.get('msg', '')),
                    'component': failure.get('component', ''),
                    'operation': failure.get('operation', ''),
                    'level': failure.get('level', 'INFO'),
                    'is_failure': failure.get('rule_fail', False),
                    'severity': failure.get('severity_score', 0.0),
                    'error_patterns': failure.get('error_patterns', {}),
                    'keywords': failure.get('keywords_found', []),
                    'timestamp': failure.get('ts', ''),
                    'transaction_id': failure.get('txn_id', ''),
                    'source': 'chat_interaction',
                    'interaction_timestamp': interaction.timestamp
                }
                training_data.append(training_example)
        
        logger.info(f"Extracted {len(training_data)} training examples from {len(interactions)} interactions")
        return training_data
    
    def export_for_training(self, output_file: Path) -> int:
        """
        Export training data to a file compatible with training scripts.
        
        Args:
            output_file: Path to output file
            
        Returns:
            Number of training examples exported
        """
        training_data = self.get_training_data()
        
        if not training_data:
            logger.warning("No training data to export")
            return 0
        
        # Convert to format compatible with existing training pipeline
        export_data = {
            'source': 'chat_interactions',
            'total_examples': len(training_data),
            'export_timestamp': datetime.now().isoformat(),
            'training_data': training_data
        }
        
        try:
            with output_file.open('w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Exported {len(training_data)} training examples to {output_file}")
            return len(training_data)
        except Exception as e:
            logger.error(f"Failed to export training data: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about collected interactions."""
        interactions = self.get_interactions()
        
        total_logs = sum(1 for i in interactions if i.log_content)
        total_failures = sum(
            len(i.analysis_results.get('failures', []))
            for i in interactions
        )
        
        return {
            'total_interactions': len(interactions),
            'total_logs_analyzed': total_logs,
            'total_failures_detected': total_failures,
            'data_file': str(self.interactions_file),
            'data_file_size_mb': self.interactions_file.stat().st_size / (1024 * 1024) if self.interactions_file.exists() else 0
        }
    
    def clear_old_data(self, days: int = 90) -> int:
        """
        Clear interactions older than specified days.
        
        Args:
            days: Number of days to keep
            
        Returns:
            Number of interactions removed
        """
        interactions = self.get_interactions()
        cutoff_date = datetime.now().timestamp() - (days * 24 * 60 * 60)
        
        kept = []
        removed = 0
        
        for interaction in interactions:
            try:
                interaction_date = datetime.fromisoformat(interaction.timestamp).timestamp()
                if interaction_date > cutoff_date:
                    kept.append(interaction)
                else:
                    removed += 1
            except:
                # Keep if we can't parse date
                kept.append(interaction)
        
        # Rewrite file with kept interactions
        if removed > 0:
            with self.lock:
                try:
                    with self.interactions_file.open('w', encoding='utf-8') as f:
                        for interaction in reversed(kept):
                            f.write(json.dumps(interaction.to_dict(), ensure_ascii=False) + '\n')
                    
                    logger.info(f"Removed {removed} old interactions (kept {len(kept)})")
                except Exception as e:
                    logger.error(f"Failed to clear old data: {e}")
        
        return removed

