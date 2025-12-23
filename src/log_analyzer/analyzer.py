#!/usr/bin/env python3
"""
Enhanced Log Analyzer with Supervised Learning Training
========================================================
This version correctly identifies failures in your specific log format and supports
training supervised learning models to improve accuracy.

Your logs follow this pattern:
TIMESTAMP|NUMBER|NUMBER|UUID|COMPONENT|PJMS|failure/success|UUID|OPERATION|TRANSACTION_ID|MESSAGE

Features:
* Rule-based failure detection (looks for "failure" in the 7th field)
* Supervised learning with trainable classifiers (Random Forest, Logistic Regression, SVM, Naive Bayes)
* Model persistence (save/load trained models)
* Evaluation metrics (accuracy, precision, recall, F1 score)
* Unsupervised clustering-based detection (KMeans, DBSCAN)

Usage Examples:
    # Basic analysis with rule-based detection
    python python_logs_analyser.py
    
    # Train a model on your log file
    python python_logs_analyser.py --train --logfile server.log --model trained_model.pkl
    
    # Use trained model for analysis (improved accuracy)
    python python_logs_analyser.py --model trained_model.pkl --logfile server.log
    
    # Train with specific model type
    python python_logs_analyser.py --train --model-type random_forest --logfile server.log
    
    # Unsupervised clustering (backward compatibility)
    python python_logs_analyser.py --ml 5 --threshold 0.25
"""

from __future__ import annotations
import argparse
import json
import logging
import pickle
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterator, Dict, Any, List, Optional, Set, Tuple, Union
from datetime import datetime
from collections import defaultdict
import warnings

# Import enhanced pattern analysis
try:
    from .pattern_analysis import PatternAnalyzer, THRESHOLDS, FLOW_STATES
except ImportError:
    # Fallback if pattern_analysis is not available
    PatternAnalyzer = None
    THRESHOLDS = {}
    FLOW_STATES = {}

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==================== FIXED REGEX PATTERNS ====================
class LogPatterns:
    """Updated regex patterns for your specific log format."""
    
    # Your log format: TIMESTAMP|NUM|NUM|UUID|COMPONENT|PJMS|STATUS|UUID|OPERATION|TXN_ID|MESSAGE
    YOUR_LOG_FORMAT = re.compile(
        r"(?P<ts>\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2},\d{3})\|"
        r"(?P<num1>[^|]*)\|(?P<num2>[^|]*)\|(?P<uuid1>[^|]*)\|"
        r"(?P<component>[^|]*)\|(?P<system>[^|]*)\|(?P<status>[^|]*)\|"
        r"(?P<uuid2>[^|]*)\|(?P<operation>[^|]*)\|(?P<txn_id>[^|]*)\|"
        r"(?P<message>.*)",
        re.IGNORECASE
    )
    
    # New log format: TIMESTAMP|QUEUE_TIME_MS|PROCESS_TIME_MS|PROCESSOR_UUID|PROCESSOR_NAME|COMPONENT|RELATIONSHIP|PG_UUID|PG_NAME|IDENTIFIER|LOG
    NEW_LOG_FORMAT = re.compile(
        r"(?P<ts>\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2},\d{3})\|"
        r"(?P<queue_time>[^|]*)\|(?P<process_time>[^|]*)\|(?P<processor_uuid>[^|]*)\|"
        r"(?P<processor_name>[^|]*)\|(?P<component>[^|]*)\|(?P<relationship>[^|]*)\|"
        r"(?P<pg_uuid>[^|]*)\|(?P<pg_name>[^|]*)\|(?P<identifier>[^|]*)\|?"
        r"(?P<log>.*)?$",
        re.IGNORECASE
    )
    
    # Fallback patterns for other formats
    FREE_FORMAT = re.compile(
        r"(?P<ts>\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2},\d{3})\s\|\s"
        r"(?P<lvl>\w+)\s\|\sreqId:(?P<req>[A-F0-9]+)\s\|\s(?P<msg>.+)",
        re.IGNORECASE
    )
    
    STACK_FORMAT = re.compile(
        r"(?P<ts>\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2},\d{3})\s\|\s"
        r"(?P<lvl>\w+)\s\|\s(?P<logger>[^|]+)\s\|\s(?P<msg>.+)",
        re.IGNORECASE
    )
    
    # Enhanced error keywords with severity levels
    ERROR_KEYWORDS = {
        'critical': {'fatal', 'critical', 'panic', 'emergency', 'abort', 'crash', 'down'},
        'error': {'error', 'exception', 'failed', 'failure', 'fault', 'timeout', 'refused', 'rejected', 
                 'denied', 'unauthorized', 'forbidden', 'not found', 'notfound', 'missing', 'invalid',
                 'abort', 'cancel', 'terminate', 'aborted', 'cancelled'},
        'warning': {'warn', 'warning', 'deprecated', 'invalid', 'retry', 'retries', 'slow', 'degraded'}
    }
    
    # HTTP/Status code patterns (4xx, 5xx errors)
    STATUS_CODE_PATTERNS = [
        re.compile(r'\b(4\d{2}|5\d{2})\b'),  # HTTP status codes 400-499, 500-599
        re.compile(r'\bstatus[:\s]+(4\d{2}|5\d{2})\b', re.IGNORECASE),
        re.compile(r'\bcode[:\s]+(4\d{2}|5\d{2})\b', re.IGNORECASE),
        re.compile(r'\brespCode[:\s]+(4\d{2}|5\d{2})\b', re.IGNORECASE),
        re.compile(r'\berror[:\s]*code[:\s]+(\d{3,5})\b', re.IGNORECASE),
        re.compile(r'\b(EC\d+|ERR\d+|ERROR\d+)\b', re.IGNORECASE),  # Error codes like EC123, ERR456
    ]
    
    # Compiled regex for common error patterns in your logs
    ERROR_PATTERNS = [
        re.compile(r'\bfailure\b', re.IGNORECASE),  # Direct failure status
        re.compile(r'\b(connection\s*(refused|timeout|reset|closed|lost))\b', re.IGNORECASE),
        re.compile(r'\b(timeout|timed\s*out|expired)\b', re.IGNORECASE),
        re.compile(r'\b(exception|error|throwable)\b', re.IGNORECASE),
        re.compile(r'\b(null\s*pointer|npe|nullpointer)\b', re.IGNORECASE),
        re.compile(r'\b(out\s*of\s*(memory|disk|space|bounds))\b', re.IGNORECASE),
        re.compile(r'\b(no\s*record\s*found|record\s*not\s*found|not\s*found)\b', re.IGNORECASE),
        re.compile(r'\b(unable|unavailable|unreachable|offline)\b', re.IGNORECASE),
        re.compile(r'\b(rejected|denied|unauthorized|forbidden)\b', re.IGNORECASE),
        re.compile(r'\b(abort|cancel|terminate|stop|halt)\b', re.IGNORECASE),
    ]

# ==================== DATA MODELS ====================
@dataclass
class LogEntry:
    """Enhanced log entry with comprehensive metadata."""
    ts: str
    msg: str
    level: str = "INFO"
    component: str = ""
    operation: str = ""
    txn_id: str = ""
    raw: str = ""
    rule_fail: bool = False
    severity_score: float = 0.0
    keywords_found: Set[str] = field(default_factory=set)
    # Enhanced fields for comprehensive analysis
    queue_time_ms: int = 0
    process_time_ms: int = 0
    processor_uuid: str = ""
    processor_name: str = ""
    relationship: str = ""
    pg_uuid: str = ""
    pg_name: str = ""
    identifier: str = ""
    http_status: Optional[int] = None
    external_endpoint: Optional[str] = None
    error_message: Optional[str] = None
    filename: Optional[str] = None
    detected_state: Optional[str] = None
    error_patterns: Dict[str, Any] = field(default_factory=dict)
    # Root cause analysis fields
    is_root_cause: bool = False
    is_cascade: bool = False
    root_cause_type: str = ""
    caused_by: str = ""
    cascade_count: int = 0
    impact_count: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['keywords_found'] = list(data['keywords_found'])
        # Convert None to empty string for JSON serialization
        for key, value in data.items():
            if value is None:
                data[key] = ""
        return data

@dataclass
class AnalysisStats:
    """Comprehensive analysis statistics."""
    total_lines: int = 0
    parsed_entries: int = 0
    rule_failures: int = 0
    ml_failures: int = 0
    critical_errors: int = 0
    warnings: int = 0
    processing_time: float = 0.0
    file_size_mb: float = 0.0
    # Enhanced metrics
    connection_issues: int = 0
    http_errors: int = 0
    timeout_issues: int = 0
    data_issues: int = 0
    flow_breaks: int = 0
    success_count: int = 0
    failure_count: int = 0
    total_duration_ms: float = 0.0
    avg_process_time_ms: float = 0.0
    avg_queue_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

# ==================== ENHANCED PARSING ====================
class LogParser:
    """Enhanced log parser for your specific format."""
    
    def __init__(self):
        self.patterns = LogPatterns()
        self.pattern_analyzer = PatternAnalyzer() if PatternAnalyzer else None
        
    def detect_format(self, sample_lines: List[str]) -> str:
        """Detect log format from multiple sample lines."""
        format_scores = {'your_format': 0, 'new_format': 0, 'free': 0, 'stack': 0}
        
        for line in sample_lines[:10]:  # Check first 10 lines
            # Skip header lines
            if line.strip().startswith('TIMESTAMP|'):
                continue
            if self.patterns.NEW_LOG_FORMAT.match(line):
                format_scores['new_format'] += 1
            elif self.patterns.YOUR_LOG_FORMAT.match(line):
                format_scores['your_format'] += 1
            elif self.patterns.FREE_FORMAT.match(line):
                format_scores['free'] += 1
            elif self.patterns.STACK_FORMAT.match(line):
                format_scores['stack'] += 1
        
        detected = max(format_scores, key=format_scores.get) if any(format_scores.values()) else 'unknown'
        logger.info(f"Format detection scores: {format_scores} -> {detected}")
        return detected
    
    def _parse_json_in_message(self, text: str) -> Dict[str, Any]:
        """Extract JSON objects from log message and check for error indicators."""
        json_data = {}
        
        # Try to find JSON objects in the message
        # Look for patterns like { "code": "404", "status": "Failed" }
        import json
        import re
        
        # Find JSON objects (starting with { and ending with })
        json_pattern = re.compile(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}')
        json_matches = json_pattern.findall(text)
        
        for json_str in json_matches:
            try:
                # Try to parse as JSON
                data = json.loads(json_str)
                
                # Check for error indicators in JSON
                if isinstance(data, dict):
                    # Check for status code
                    code = data.get('code') or data.get('statusCode') or data.get('errorCode')
                    if code:
                        # Convert to string and check if it's numeric
                        code_str = str(code).strip('"\'')
                        if code_str.isdigit():
                            code_num = int(code_str)
                            if 400 <= code_num < 600:
                                json_data['has_error_code'] = True
                                json_data['error_code'] = code_str
                                json_data['code_value'] = code_num
                    
                    # Check for status field
                    status = data.get('status') or data.get('result') or data.get('resultCode')
                    if status:
                        status_str = str(status).lower().strip('"\'')
                        if 'fail' in status_str or status_str == 'error':
                            json_data['has_failed_status'] = True
                            json_data['status_value'] = status_str
                    
                    # Check for error message
                    error_msg = data.get('message') or data.get('error') or data.get('errorMessage')
                    if error_msg:
                        error_msg_lower = str(error_msg).lower()
                        if any(term in error_msg_lower for term in ['not found', 'error', 'failed', 'exception', 'invalid']):
                            json_data['has_error_message'] = True
                            json_data['error_message'] = str(error_msg)
                    
                    # Check for reason field
                    reason = data.get('reason')
                    if reason and str(reason).strip():
                        json_data['has_reason'] = True
                        
            except (json.JSONDecodeError, ValueError):
                # Not valid JSON, continue
                continue
        
        return json_data
    
    def calculate_severity(self, msg: str, level: str, status: str = "") -> Tuple[float, Set[str]]:
        """Calculate severity score and find matching keywords."""
        msg_lower = msg.lower()
        level_lower = level.lower()
        status_lower = status.lower()
        keywords_found = set()
        score = 0.0
        
        # Combine message and status for comprehensive checking
        full_text = f"{msg} {status}".lower()
        
        # Parse JSON in message for error indicators
        json_info = self._parse_json_in_message(msg)
        if json_info:
            # JSON error code found
            if json_info.get('has_error_code'):
                code = json_info.get('error_code', '')
                code_num = json_info.get('code_value', 0)
                score += 0.9
                keywords_found.add(f'json_code_{code}')
                if 500 <= code_num < 600:
                    score += 0.1  # Extra for server errors
            
            # JSON status "Failed" found
            if json_info.get('has_failed_status'):
                score += 0.9
                keywords_found.add('json_status_failed')
                status_val = json_info.get('status_value', '')
                if status_val:
                    keywords_found.add(f'json_status_{status_val}')
            
            # JSON error message found
            if json_info.get('has_error_message'):
                score += 0.7
                keywords_found.add('json_error_message')
                error_msg = json_info.get('error_message', '')
                if 'not found' in error_msg.lower():
                    keywords_found.add('not_found')
        
        # Check status field first (most reliable for your format)
        if status_lower == 'failure':
            score += 0.9
            keywords_found.add('failure')
        elif 'no record found' in status_lower:
            score += 0.7
            keywords_found.add('no record found')
        elif status_lower == 'error':
            score += 0.8
            keywords_found.add('error')
        elif any(term in status_lower for term in ['failed', 'exception', 'abort', 'reject']):
            score += 0.7
            keywords_found.add(status_lower)
        
        # Check for HTTP/Status codes (4xx, 5xx are failures) - also check in JSON context
        for pattern in self.patterns.STATUS_CODE_PATTERNS:
            matches = pattern.findall(full_text)
            for match in matches:
                if isinstance(match, tuple):
                    code = match[0] if match else None
                else:
                    code = match
                
                if code:
                    code_num = int(code) if code.isdigit() else 0
                    if 400 <= code_num < 600:  # 4xx or 5xx
                        score += 0.9
                        keywords_found.add(f'status_{code}')
                        if 500 <= code_num < 600:
                            score += 0.1  # Extra for server errors
                    elif code_num >= 1000:  # Custom error codes
                        score += 0.6
                        keywords_found.add(f'error_code_{code}')
        
        # Check level-based scoring
        if level_lower in ['fatal', 'critical']:
            score += 1.0
        elif level_lower == 'error':
            score += 0.8
        elif level_lower in ['warn', 'warning']:
            score += 0.4
        
        # Check keyword categories in message
        msg_words = set(msg_lower.split())
        for category, keywords in self.patterns.ERROR_KEYWORDS.items():
            found = keywords.intersection(msg_words)
            if found:
                keywords_found.update(found)
                if category == 'critical':
                    score += 0.8
                elif category == 'error':
                    score += 0.6
                elif category == 'warning':
                    score += 0.3
        
        # Check regex patterns in both message and status
        for pattern in self.patterns.ERROR_PATTERNS:
            if pattern.search(msg) or pattern.search(status):
                score += 0.4
                break
        
        return min(score, 1.0), keywords_found
    
    def parse_line(self, line: str, fmt: str) -> Optional[LogEntry]:
        """Parse a single log line with enhanced error detection."""
        line = line.rstrip('\n\r')
        if not line.strip():
            return None
            
        try:
            # Skip header lines
            if line.strip().startswith('TIMESTAMP|'):
                return None
            if fmt == 'new_format':
                return self._parse_new_format(line)
            elif fmt == 'your_format':
                return self._parse_your_format(line)
            elif fmt == 'free':
                return self._parse_free(line)
            elif fmt == 'stack':
                return self._parse_stack(line)
        except Exception as e:
            logger.debug(f"Parse error for line: {line[:100]}... Error: {e}")
        
        return None
    
    def _parse_your_format(self, line: str) -> Optional[LogEntry]:
        """Parse your specific log format."""
        if match := self.patterns.YOUR_LOG_FORMAT.match(line):
            groups = match.groupdict()
            
            # Extract key fields
            ts = groups.get('ts', '')
            component = groups.get('component', '')
            status = groups.get('status', '').lower()
            operation = groups.get('operation', '')
            txn_id = groups.get('txn_id', '')
            message = groups.get('message', '')
            
            # Determine if this is a failure - check status, message, and JSON
            full_text = f"{status} {message}".lower()
            is_fail = status == 'failure'
            
            # Check for JSON error indicators in message
            json_info = self._parse_json_in_message(message)
            if json_info:
                if json_info.get('has_error_code') or json_info.get('has_failed_status'):
                    is_fail = True
            
            # Check for HTTP status codes (4xx, 5xx) in message or status
            if not is_fail:
                for pattern in self.patterns.STATUS_CODE_PATTERNS:
                    matches = pattern.findall(full_text)
                    for match in matches:
                        code = match[0] if isinstance(match, tuple) and match else match
                        if code and code.isdigit():
                            code_num = int(code)
                            if 400 <= code_num < 600:  # 4xx or 5xx are failures
                                is_fail = True
                                break
                    if is_fail:
                        break
            
            # Also check for other failure indicators
            if not is_fail:
                is_fail = any(term in status for term in ['error', 'failed', 'exception', 'abort', 'reject'])
            
            level = 'ERROR' if is_fail else 'INFO'
            
            # Calculate severity
            severity, keywords = self.calculate_severity(message, level, status)
            
            # For failures, always capture the full line
            raw_line = line if is_fail or severity > 0.5 else ''
            
            return LogEntry(
                ts=ts,
                msg=message,
                level=level,
                component=component,
                operation=operation,
                txn_id=txn_id,
                raw=raw_line,
                rule_fail=is_fail or severity > 0.5,
                severity_score=max(severity, 0.9 if is_fail else 0.0),  # Boost failure status
                keywords_found=keywords
            )
        return None
    
    def _parse_new_format(self, line: str) -> Optional[LogEntry]:
        """Parse new log format: TIMESTAMP|QUEUE_TIME_MS|PROCESS_TIME_MS|PROCESSOR_UUID|PROCESSOR_NAME|COMPONENT|RELATIONSHIP|PG_UUID|PG_NAME|IDENTIFIER|LOG"""
        if match := self.patterns.NEW_LOG_FORMAT.match(line):
            groups = match.groupdict()
            
            # Extract key fields
            ts = groups.get('ts', '')
            queue_time_str = groups.get('queue_time', '0')
            process_time_str = groups.get('process_time', '0')
            processor_uuid = groups.get('processor_uuid', '')
            processor_name = groups.get('processor_name', '')
            component = groups.get('component', '')
            relationship = groups.get('relationship', '')
            pg_uuid = groups.get('pg_uuid', '')
            pg_name = groups.get('pg_name', '')
            identifier = groups.get('identifier', '')
            log_message = groups.get('log', '')
            
            # Parse numeric fields
            try:
                queue_time_ms = int(queue_time_str) if queue_time_str.isdigit() else 0
            except:
                queue_time_ms = 0
            try:
                process_time_ms = int(process_time_str) if process_time_str.isdigit() else 0
            except:
                process_time_ms = 0
            
            # Extract HTTP status code and endpoint from log message
            http_status = None
            external_endpoint = None
            filename = None
            error_message = None
            
            # Extract filename
            filename_match = re.search(r'Filename:\s*([^\s|]+)', log_message)
            if filename_match:
                filename = filename_match.group(1)
            
            # Extract HTTP status code
            status_match = re.search(r'(?:3P\s+)?Status\s+Code:\s*(\d{3})', log_message, re.IGNORECASE)
            if status_match:
                http_status = int(status_match.group(1))
            else:
                # Check JSON for status code
                json_info = self._parse_json_in_message(log_message)
                if json_info and json_info.get('code_value'):
                    http_status = json_info.get('code_value')
            
            # Extract endpoint/URL
            url_match = re.search(r"URL:\s*(?:GET|POST|PUT|DELETE)\s+'([^']+)'", log_message)
            if url_match:
                external_endpoint = url_match.group(1)
            
            # Extract error message from JSON
            json_info = self._parse_json_in_message(log_message)
            if json_info and json_info.get('error_message'):
                error_message = json_info.get('error_message')
            
            # Determine if this is a failure
            relationship_lower = relationship.lower()
            full_text = f"{relationship} {log_message} {processor_name}".lower()
            
            # Check for explicit failure status
            is_fail = (relationship_lower == 'failure' or 
                      'no record found' in relationship_lower or
                      'error' in relationship_lower or
                      'failed' in relationship_lower or
                      'exception' in relationship_lower or
                      'abort' in relationship_lower or
                      'reject' in relationship_lower or
                      'denied' in relationship_lower or
                      relationship_lower == 'no retry')
            
            # Check for JSON error indicators in log message
            if not is_fail and json_info:
                if json_info.get('has_error_code') or json_info.get('has_failed_status'):
                    is_fail = True
            
            # Check for HTTP status codes (4xx, 5xx)
            if not is_fail and http_status:
                if 400 <= http_status < 600:
                    is_fail = True
            
            # Use pattern analyzer for enhanced detection
            error_patterns = {}
            detected_state = None
            if self.pattern_analyzer:
                error_patterns = self.pattern_analyzer.detect_error_patterns(log_message, processor_name)
                detected_state = self.pattern_analyzer.detect_state(processor_name)
                if error_patterns.get('categories'):
                    is_fail = True
            
            level = 'ERROR' if is_fail else 'INFO'
            
            # Calculate severity
            severity, keywords = self.calculate_severity(log_message or processor_name, level, relationship)
            
            # Boost severity based on error patterns
            if error_patterns.get('severity', 0) > severity:
                severity = error_patterns.get('severity', severity)
            
            # For failures, always capture the full line
            raw_line = line if is_fail or severity > 0.3 else ''
            
            return LogEntry(
                ts=ts,
                msg=log_message or processor_name,
                level=level,
                component=component,
                operation=processor_name,
                txn_id=identifier,
                raw=raw_line,
                rule_fail=is_fail or severity > 0.5,
                severity_score=max(severity, 0.9 if is_fail else 0.0),
                keywords_found=keywords,
                queue_time_ms=queue_time_ms,
                process_time_ms=process_time_ms,
                processor_uuid=processor_uuid,
                processor_name=processor_name,
                relationship=relationship,
                pg_uuid=pg_uuid,
                pg_name=pg_name,
                identifier=identifier,
                http_status=http_status,
                external_endpoint=external_endpoint,
                error_message=error_message,
                filename=filename,
                detected_state=detected_state,
                error_patterns=error_patterns
            )
        return None
    
    def _parse_free(self, line: str) -> Optional[LogEntry]:
        """Parse free format logs."""
        if match := self.patterns.FREE_FORMAT.match(line):
            groups = match.groupdict()
            level = groups.get('lvl', 'INFO')
            msg = groups.get('msg', '')
            
            severity, keywords = self.calculate_severity(msg, level)
            is_fail = level.lower() == 'error' or severity > 0.5
            
            return LogEntry(
                ts=groups.get('ts', ''),
                msg=msg,
                level=level,
                raw=line if is_fail else '',
                rule_fail=is_fail,
                severity_score=severity,
                keywords_found=keywords
            )
        return None
    
    def _parse_stack(self, line: str) -> Optional[LogEntry]:
        """Parse stack trace format logs."""
        if match := self.patterns.STACK_FORMAT.match(line):
            groups = match.groupdict()
            level = groups.get('lvl', 'INFO')
            msg = groups.get('msg', '')
            
            severity, keywords = self.calculate_severity(msg, level)
            is_fail = level.lower() == 'error' or severity > 0.5
            
            return LogEntry(
                ts=groups.get('ts', ''),
                msg=msg,
                level=level,
                component=groups.get('logger', ''),
                raw=line if is_fail else '',
                rule_fail=is_fail,
                severity_score=severity,
                keywords_found=keywords
            )
        return None

# ==================== SIMPLIFIED ML DETECTION ====================
class MLErrorDetector:
    """Simplified ML-based error detection."""
    
    def __init__(self, algorithm: str = 'kmeans', threshold: float = 0.3, max_features: int = 1000):
        self.algorithm = algorithm.lower()
        self.threshold = threshold
        self.max_features = max_features
        
    def detect_failures(self, entries: List[LogEntry], rule_failed_indices: Set[int], 
                       n_clusters: int) -> Tuple[Set[int], Dict[str, Any]]:
        """Detect additional failures using ML clustering."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.cluster import MiniBatchKMeans, DBSCAN
            import numpy as np
        except ImportError as exc:
            logger.warning("scikit-learn not available for ML detection")
            return set(), {'error': 'scikit-learn not installed'}
        
        if len(entries) < max(n_clusters, 5):
            logger.warning(f"Too few entries ({len(entries)}) for clustering")
            return set(), {'error': f'Insufficient data: {len(entries)} entries'}
        
        # Extract and vectorize messages
        texts = [f"{entry.component} {entry.operation} {entry.msg}" for entry in entries]
        
        try:
            vectorizer = TfidfVectorizer(
                stop_words='english',
                max_features=self.max_features,
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95
            )
            X = vectorizer.fit_transform(texts)
        except ValueError as e:
            logger.error(f"Vectorization failed: {e}")
            return set(), {'error': f'Vectorization failed: {str(e)}'}
        
        # Apply clustering
        try:
            if self.algorithm == 'kmeans':
                clusterer = MiniBatchKMeans(
                    n_clusters=n_clusters, 
                    random_state=42, 
                    batch_size=min(500, len(entries))
                )
                labels = clusterer.fit_predict(X)
            elif self.algorithm == 'dbscan':
                clusterer = DBSCAN(eps=0.3, min_samples=3)
                labels = clusterer.fit_predict(X.toarray())
            else:
                raise ValueError(f"Unsupported algorithm: {self.algorithm}")
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            return set(), {'error': f'Clustering failed: {str(e)}'}
        
        # Analyze clusters
        additional_failures = self._analyze_clusters(labels, rule_failed_indices)
        
        return additional_failures, {
            'algorithm': self.algorithm,
            'clusters': len(set(labels)) - (1 if -1 in labels else 0),
            'threshold': self.threshold,
            'features_used': X.shape[1],
            'additional_found': len(additional_failures)
        }
    
    def _analyze_clusters(self, labels, rule_failed_indices: Set[int]) -> Set[int]:
        """Analyze cluster composition to find additional failures."""
        import numpy as np
        
        cluster_stats = {}
        
        # Count failures per cluster
        for idx, label in enumerate(labels):
            if label == -1:  # DBSCAN noise
                continue
            if label not in cluster_stats:
                cluster_stats[label] = {'total': 0, 'failures': 0}
            cluster_stats[label]['total'] += 1
            if idx in rule_failed_indices:
                cluster_stats[label]['failures'] += 1
        
        # Find clusters with high failure rates
        additional_failures = set()
        for label, stats in cluster_stats.items():
            if stats['total'] > 0:
                failure_rate = stats['failures'] / stats['total']
                logger.debug(f"Cluster {label}: {stats['failures']}/{stats['total']} = {failure_rate:.2f}")
                
                if failure_rate >= self.threshold:
                    # Add all non-failed entries from this cluster
                    cluster_indices = {i for i, l in enumerate(labels) 
                                     if l == label and i not in rule_failed_indices}
                    additional_failures.update(cluster_indices)
                    logger.info(f"Cluster {label} flagged as failure-prone: {len(cluster_indices)} additional entries")
        
        return additional_failures

# ==================== SUPERVISED LEARNING CLASSIFIER ====================
class SupervisedErrorClassifier:
    """Trainable supervised learning classifier for error detection."""
    
    def __init__(self, model_type: str = 'random_forest', max_features: int = 2000):
        self.model_type = model_type.lower()
        self.max_features = max_features
        self.vectorizer = None
        self.classifier = None
        self.is_trained = False
        self.training_metrics = {}
        
    def _create_classifier(self):
        """Create the appropriate classifier based on model_type."""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.svm import SVC
            from sklearn.naive_bayes import MultinomialNB
            from sklearn.model_selection import cross_val_score
        except ImportError:
            raise ImportError("scikit-learn is required for supervised learning")
        
        if self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
        elif self.model_type == 'logistic_regression':
            return LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight='balanced',
                solver='liblinear'
            )
        elif self.model_type == 'svm':
            return SVC(
                kernel='rbf',
                probability=True,
                random_state=42,
                class_weight='balanced',
                gamma='scale'
            )
        elif self.model_type == 'naive_bayes':
            return MultinomialNB(alpha=1.0)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def extract_features(self, entries: List[LogEntry]) -> List[Dict[str, Any]]:
        """Extract features from log entries for training/prediction."""
        features = []
        for entry in entries:
            # Combine text features
            text = f"{entry.component} {entry.operation} {entry.msg}".lower()
            
            # Extract structured features
            feature_dict = {
                'text': text,
                'component': entry.component.lower(),
                'operation': entry.operation.lower(),
                'level': entry.level.lower(),
                'has_keywords': len(entry.keywords_found) > 0,
                'keyword_count': len(entry.keywords_found),
                'severity_score': entry.severity_score,
                'message_length': len(entry.msg),
                'has_uuid': bool(entry.txn_id),
            }
            features.append(feature_dict)
        return features
    
    def prepare_training_data(self, entries: List[LogEntry], labels: List[bool]) -> Tuple[Any, Any]:
        """Prepare training data with feature extraction and vectorization."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.preprocessing import StandardScaler
            import numpy as np
        except ImportError:
            raise ImportError("scikit-learn is required for training")
        
        if len(entries) != len(labels):
            raise ValueError(f"Mismatch: {len(entries)} entries but {len(labels)} labels")
        
        # Extract features
        feature_dicts = self.extract_features(entries)
        
        # Text vectorization
        texts = [fd['text'] for fd in feature_dicts]
        if not self.vectorizer:
            self.vectorizer = TfidfVectorizer(
                stop_words='english',
                max_features=self.max_features,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                lowercase=True
            )
            X_text = self.vectorizer.fit_transform(texts)
        else:
            X_text = self.vectorizer.transform(texts)
        
        # Extract numerical features
        numerical_features = np.array([
            [
                fd['severity_score'],
                fd['keyword_count'],
                fd['message_length'],
                1.0 if fd['has_keywords'] else 0.0,
                1.0 if fd['has_uuid'] else 0.0,
            ]
            for fd in feature_dicts
        ])
        
        # Combine text and numerical features
        from scipy.sparse import hstack
        X = hstack([X_text, numerical_features])
        
        y = np.array([1 if label else 0 for label in labels])
        
        return X, y
    
    def train(self, entries: List[LogEntry], labels: List[bool], 
              test_size: float = 0.2, cross_validate: bool = True) -> Dict[str, Any]:
        """Train the classifier on labeled data."""
        if len(entries) < 10:
            raise ValueError(f"Insufficient training data: {len(entries)} entries (minimum 10 required)")
        
        try:
            from sklearn.model_selection import train_test_split, cross_val_score
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
            import numpy as np
        except ImportError:
            raise ImportError("scikit-learn is required for training")
        
        logger.info(f"Training {self.model_type} classifier on {len(entries)} entries...")
        
        # Prepare training data
        X, y = self.prepare_training_data(entries, labels)
        
        # Check if we have enough samples for stratified split
        from collections import Counter
        class_counts = Counter(y)
        min_class_count = min(class_counts.values())
        
        # Use stratified split only if we have at least 2 samples in each class
        use_stratify = min_class_count >= 2 and test_size > 0
        
        # Split into train/test
        if use_stratify:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
        else:
            # Use non-stratified split or adjust test size
            if min_class_count == 1:
                # If only 1 sample in minority class, use very small test set or skip split
                logger.warning(f"Very imbalanced dataset: {class_counts}. Using smaller test set.")
                test_size_adj = min(0.1, 1.0 / len(y))  # Use at most 10% or 1 sample
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size_adj, random_state=42
                )
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
        
        # Create and train classifier
        self.classifier = self._create_classifier()
        self.classifier.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate on test set
        y_pred = self.classifier.predict(X_test)
        y_pred_proba = self.classifier.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # Cross-validation if requested
        cv_scores = None
        if cross_validate and len(entries) >= 20:
            try:
                # Check if we have enough samples for CV
                from collections import Counter
                train_class_counts = Counter(y_train)
                min_train_class = min(train_class_counts.values())
                
                if min_train_class >= 2:
                    cv_folds = min(5, len(y_train)//4, min_train_class)
                    if cv_folds >= 2:
                        cv_scores = cross_val_score(self.classifier, X_train, y_train, cv=cv_folds, scoring='f1')
                        logger.info(f"Cross-validation F1: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
                    else:
                        logger.warning("Skipping cross-validation: insufficient samples per class")
                else:
                    logger.warning("Skipping cross-validation: insufficient samples in minority class")
            except Exception as e:
                logger.warning(f"Cross-validation failed: {e}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        self.training_metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'test_samples': len(y_test),
            'train_samples': len(y_train),
            'cv_mean': float(cv_scores.mean()) if cv_scores is not None else None,
            'cv_std': float(cv_scores.std()) if cv_scores is not None else None,
            'confusion_matrix': cm.tolist(),
            'model_type': self.model_type,
            'features_used': X.shape[1]
        }
        
        logger.info(f"Training complete! Accuracy: {accuracy:.3f}, F1: {f1:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}")
        
        return self.training_metrics
    
    def predict(self, entries: List[LogEntry]) -> Tuple[List[bool], List[float]]:
        """Predict failures for log entries."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        if not entries:
            return [], []
        
        # Prepare features
        X, _ = self.prepare_training_data(entries, [False] * len(entries))
        
        # Predict
        predictions = self.classifier.predict(X)
        probabilities = self.classifier.predict_proba(X)[:, 1]
        
        return [bool(p) for p in predictions], [float(prob) for prob in probabilities]
    
    def save_model(self, model_path: Path):
        """Save the trained model to disk."""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            'vectorizer': self.vectorizer,
            'classifier': self.classifier,
            'model_type': self.model_type,
            'max_features': self.max_features,
            'training_metrics': self.training_metrics,
            'is_trained': self.is_trained
        }
        
        with model_path.open('wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: Path):
        """Load a trained model from disk."""
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        with model_path.open('rb') as f:
            model_data = pickle.load(f)
        
        self.vectorizer = model_data['vectorizer']
        self.classifier = model_data['classifier']
        self.model_type = model_data['model_type']
        self.max_features = model_data.get('max_features', 2000)
        self.training_metrics = model_data.get('training_metrics', {})
        self.is_trained = model_data.get('is_trained', True)
        
        logger.info(f"Model loaded from {model_path}")
        if self.training_metrics:
            logger.info(f"Model accuracy: {self.training_metrics.get('accuracy', 'N/A'):.3f}")

# ==================== MAIN PROCESSOR ====================
class LogAnalyzer:
    """Main log analysis orchestrator."""
    
    def __init__(self, batch_size: int = 10000, model_path: Optional[Path] = None,
                 ml_models_config: Optional[Dict[str, Any]] = None):
        self.parser = LogParser()
        self.batch_size = batch_size
        self.supervised_classifier = None
        
        # ML models (optional)
        self.ml_models_config = ml_models_config or {}
        self.classification_model = None
        self.anomaly_detector = None
        self.ensemble_detector = None
        
        # Load trained model if provided
        if model_path and model_path.exists():
            try:
                self.supervised_classifier = SupervisedErrorClassifier()
                self.supervised_classifier.load_model(model_path)
                logger.info(f"Loaded trained model from {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load model from {model_path}: {e}")
        
        # Load ML models if configured
        self._load_ml_models()
    
    def _load_ml_models(self):
        """Load ML models if configured."""
        try:
            from .ml_models import ClassificationModel, AnomalyDetector, EnsembleDetector
            
            # Helper to resolve paths (relative to project root or absolute)
            def resolve_path(path_str):
                if not path_str:
                    return None
                path = Path(path_str)
                if path.is_absolute():
                    return path
                # Try relative to project root (assuming analyzer.py is in src/log_analyzer/)
                # Go up: src/log_analyzer -> src -> project_root
                project_root = Path(__file__).parent.parent.parent
                resolved = project_root / path
                if resolved.exists():
                    return resolved
                # If not found, return original path
                return path
            
            # Classification model
            if self.ml_models_config.get('classification_model_path'):
                cls_path = resolve_path(self.ml_models_config['classification_model_path'])
                if cls_path and cls_path.exists():
                    self.classification_model = ClassificationModel(
                        model_type=self.ml_models_config.get('classification_type', 'xgboost')
                    )
                    self.classification_model.load(cls_path)
                    logger.info(f"Loaded classification model from {cls_path}")
            
            # Anomaly detector
            if self.ml_models_config.get('anomaly_detector_path'):
                anomaly_path = resolve_path(self.ml_models_config['anomaly_detector_path'])
                if anomaly_path and anomaly_path.exists():
                    self.anomaly_detector = AnomalyDetector(
                        contamination=self.ml_models_config.get('anomaly_contamination', 0.1)
                    )
                    self.anomaly_detector.load(anomaly_path)
                    logger.info(f"Loaded anomaly detector from {anomaly_path}")
            
            # Ensemble detector
            if self.classification_model or self.anomaly_detector:
                self.ensemble_detector = EnsembleDetector()
                if self.classification_model:
                    self.ensemble_detector.add_classification_model(self.classification_model)
                if self.anomaly_detector:
                    self.ensemble_detector.add_anomaly_detector(self.anomaly_detector)
                
                # Set custom weights if provided
                if self.ml_models_config.get('ensemble_weights'):
                    self.ensemble_detector.set_weights(self.ml_models_config['ensemble_weights'])
                
                logger.info("Ensemble detector initialized")
        except ImportError as e:
            logger.debug(f"ML models not available: {e}")
        except Exception as e:
            logger.warning(f"Failed to load ML models: {e}")
        
    def analyze(self, log_file: Path, max_failures: Optional[int] = None,
                ml_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Main analysis function."""
        start_time = time.time()
        
        # Initialize stats
        stats = AnalysisStats()
        stats.file_size_mb = log_file.stat().st_size / (1024 * 1024)
        
        logger.info(f"Analyzing {log_file} ({stats.file_size_mb:.1f} MB)")
        
        # Stream and parse log file
        all_entries, rule_failures = self._process_log_file(log_file, stats, max_failures)
        
        logger.info(f"Found {len(rule_failures)} rule-based failures out of {len(all_entries)} entries")
        
        # Apply supervised learning if model is loaded
        supervised_failures = []
        supervised_info = {}
        if self.supervised_classifier and self.supervised_classifier.is_trained and all_entries:
            logger.info("Applying trained supervised learning model...")
            try:
                predictions, probabilities = self.supervised_classifier.predict(all_entries)
                supervised_indices = {i for i, (pred, prob) in enumerate(zip(predictions, probabilities)) 
                                    if pred or prob > 0.5}
                supervised_failures = [all_entries[i] for i in supervised_indices 
                                      if not all_entries[i].rule_fail]  # Only new ones
                stats.ml_failures = len(supervised_failures)
                supervised_info = {
                    'model_type': self.supervised_classifier.model_type,
                    'accuracy': self.supervised_classifier.training_metrics.get('accuracy', 0),
                    'f1_score': self.supervised_classifier.training_metrics.get('f1_score', 0),
                    'predictions_made': len(predictions),
                    'high_confidence': sum(1 for p in probabilities if p > 0.7),
                    'additional_found': len(supervised_failures)
                }
                logger.info(f"Supervised model found {len(supervised_failures)} additional potential failures")
            except Exception as e:
                logger.error(f"Supervised prediction failed: {e}")
        
        # Apply ML detection if requested (clustering-based)
        ml_failures = []
        ml_info = {}
        if ml_config and all_entries and not self.supervised_classifier:
            logger.info("Starting ML-based detection (clustering)...")
            ml_detector = MLErrorDetector(
                algorithm=ml_config.get('algorithm', 'kmeans'),
                threshold=ml_config.get('threshold', 0.3),
                max_features=ml_config.get('max_features', 1000)
            )
            
            rule_failed_indices = {i for i, entry in enumerate(all_entries) if entry.rule_fail}
            additional_indices, ml_info = ml_detector.detect_failures(
                all_entries, rule_failed_indices, ml_config.get('clusters', 5)
            )
            
            ml_failures = [all_entries[i] for i in additional_indices]
            stats.ml_failures = len(ml_failures)
            logger.info(f"ML detection found {len(ml_failures)} additional potential failures")
        
        # Combine supervised and clustering failures
        if supervised_failures:
            ml_failures.extend(supervised_failures)
            ml_info.update(supervised_info)
        
        # Apply ML models if available
        ml_model_predictions = {}
        if self.ensemble_detector and self.ensemble_detector.is_ready:
            try:
                logger.info("Applying ensemble ML models...")
                
                # Log which models are being used
                models_used = []
                if self.ensemble_detector.classification_model and self.ensemble_detector.classification_model.is_trained:
                    models_used.append(f"Classification ({self.ensemble_detector.classification_model.model_type})")
                if self.ensemble_detector.anomaly_detector and self.ensemble_detector.anomaly_detector.is_trained:
                    models_used.append("Anomaly Detection")
                if self.ensemble_detector.nlp_model and self.ensemble_detector.nlp_model.is_loaded:
                    models_used.append("NLP (Sentence Transformers)")
                models_used.append("Rule-Based Analysis")
                
                logger.info(f"Using models: {', '.join(models_used)}")
                
                # Import numpy early
                import numpy as np
                
                rule_scores = [e.severity_score for e in all_entries]
                ensemble_results = self.ensemble_detector.predict(all_entries, rule_scores)
                
                # Log individual model contributions
                individual_preds = ensemble_results.get('individual_predictions', {})
                if individual_preds:
                    for model_name, preds in individual_preds.items():
                        try:
                            if isinstance(preds, np.ndarray):
                                count = int(np.sum(preds))
                                logger.debug(f"  {model_name}: {count} predictions")
                        except:
                            pass
                
                # Find additional failures from ML models
                ml_predicted_indices = {
                    i for i, pred in enumerate(ensemble_results['predictions'])
                    if pred == 1 and not all_entries[i].rule_fail
                }
                
                ml_model_failures = [all_entries[i] for i in ml_predicted_indices]
                ml_failures.extend(ml_model_failures)
                
                # Convert numpy arrays to lists for JSON serialization
                try:
                    ensemble_preds_list = ensemble_results['predictions'].tolist() if isinstance(ensemble_results['predictions'], np.ndarray) else list(ensemble_results['predictions'])
                    ensemble_scores_list = ensemble_results['scores'].tolist() if isinstance(ensemble_results['scores'], np.ndarray) else list(ensemble_results['scores'])
                    individual_preds_dict = {}
                    for k, v in individual_preds.items():
                        if isinstance(v, np.ndarray):
                            individual_preds_dict[k] = v.tolist()
                        else:
                            individual_preds_dict[k] = v
                except (AttributeError, TypeError):
                    # Fallback if not numpy arrays
                    ensemble_preds_list = list(ensemble_results['predictions'])
                    ensemble_scores_list = list(ensemble_results['scores'])
                    individual_preds_dict = individual_preds
                
                ml_model_predictions = {
                    'ensemble_predictions': ensemble_preds_list,
                    'ensemble_scores': ensemble_scores_list,
                    'additional_found': len(ml_model_failures),
                    'models_used': models_used,
                    'individual_predictions': individual_preds_dict
                }
                
                logger.info(f"Ensemble ML models found {len(ml_model_failures)} additional potential failures")
            except Exception as e:
                logger.warning(f"Ensemble ML prediction failed: {e}")
                import traceback
                logger.debug(traceback.format_exc())
        
        # Combine and deduplicate failures
        all_failures = self._combine_failures(rule_failures, ml_failures, max_failures)
        
        # Root cause analysis - identify primary failures
        root_cause_analysis = {}
        primary_failures = []
        if all_failures:
            logger.info("Performing root cause analysis...")
            from .root_cause_analyzer import RootCauseAnalyzer
            
            root_analyzer = RootCauseAnalyzer()
            
            # Convert failures to dict format for analysis
            failure_dicts = []
            for failure in all_failures:
                failure_dict = {
                    'ts': failure.ts,
                    'msg': failure.msg,
                    'level': failure.level,
                    'component': failure.component,
                    'operation': failure.operation,
                    'txn_id': failure.txn_id or failure.identifier,
                    'identifier': failure.identifier,
                    'severity_score': failure.severity_score,
                    'http_status': failure.http_status,
                    'processor_name': failure.processor_name,
                    'relationship': failure.relationship
                }
                failure_dicts.append(failure_dict)
            
            # Perform root cause analysis
            root_cause_analysis = root_analyzer.identify_root_causes(failure_dicts, all_entries)
            
            # Enhance failures with root cause information
            enhanced_failures = root_analyzer.enhance_failures_with_root_cause(failure_dicts, root_cause_analysis)
            
            # Update failures with root cause info
            for i, enhanced in enumerate(enhanced_failures):
                if i < len(all_failures):
                    all_failures[i].is_root_cause = enhanced.get('is_root_cause', False)
                    all_failures[i].root_cause_type = enhanced.get('root_cause_type', '')
                    all_failures[i].is_cascade = enhanced.get('is_cascade', False)
                    all_failures[i].caused_by = enhanced.get('caused_by', '')
                    all_failures[i].cascade_count = enhanced.get('cascade_count', 0)
                    all_failures[i].impact_count = enhanced.get('impact_count', 1)
            
            # Get primary failures (root causes)
            primary_failures = root_cause_analysis.get('primary_failures', [])
            logger.info(f"Identified {len(primary_failures)} primary failures (root causes)")
        
        # Enhanced pattern analysis
        flow_analysis = {}
        issues = {'critical': [], 'warning': [], 'info': []}
        recommendations = []
        
        if self.parser.pattern_analyzer and all_entries:
            logger.info("Running enhanced pattern analysis...")
            
            # Group entries by transaction ID for flow analysis
            entries_by_txn = defaultdict(list)
            for entry in all_entries:
                if entry.txn_id:
                    entries_by_txn[entry.txn_id].append(entry)
            
            # Analyze each transaction flow
            connection_issues_list = []
            http_errors_list = []
            timeout_issues_list = []
            data_issues_list = []
            flow_breaks_list = []
            
            for txn_id, txn_entries in entries_by_txn.items():
                # Sort by timestamp
                txn_entries.sort(key=lambda x: x.ts)
                
                # Convert to dict format for pattern analyzer
                entry_dicts = []
                for entry in txn_entries:
                    entry_dict = {
                        'timestamp': entry.ts,
                        'processor_name': entry.processor_name,
                        'relationship': entry.relationship,
                        'process_time_ms': entry.process_time_ms,
                        'queue_time_ms': entry.queue_time_ms,
                        'log_message': entry.msg,
                        'component': entry.component
                    }
                    entry_dicts.append(entry_dict)
                
                # Detect connection issues
                for entry_dict in entry_dicts:
                    conn_issue = self.parser.pattern_analyzer.detect_connection_issues(entry_dict)
                    if conn_issue:
                        connection_issues_list.append({
                            'transaction_id': txn_id,
                            'issue': conn_issue
                        })
                
                # Detect HTTP errors
                for entry_dict in entry_dicts:
                    http_error = self.parser.pattern_analyzer.detect_http_errors(entry_dict)
                    if http_error:
                        http_errors_list.append({
                            'transaction_id': txn_id,
                            'error': http_error
                        })
                
                # Detect timeouts
                timeouts = self.parser.pattern_analyzer.detect_timeouts(entry_dicts)
                if timeouts:
                    timeout_issues_list.extend([{
                        'transaction_id': txn_id,
                        'timeout': t
                    } for t in timeouts])
                
                # Detect data issues
                data_issues = self.parser.pattern_analyzer.detect_data_issues(entry_dicts)
                if data_issues:
                    data_issues_list.extend([{
                        'transaction_id': txn_id,
                        'issue': d
                    } for d in data_issues])
                
                # Detect flow breaks
                breaks = self.parser.pattern_analyzer.detect_flow_breaks(entry_dicts)
                if breaks:
                    flow_breaks_list.extend([{
                        'transaction_id': txn_id,
                        'break': b
                    } for b in breaks])
            
            # Update statistics
            stats.connection_issues = len(connection_issues_list)
            stats.http_errors = len(http_errors_list)
            stats.timeout_issues = len(timeout_issues_list)
            stats.data_issues = len(data_issues_list)
            stats.flow_breaks = len(flow_breaks_list)
            
            # Categorize issues
            for conn in connection_issues_list:
                if conn['issue']['severity'] >= 0.8:
                    issues['critical'].append(conn)
                else:
                    issues['warning'].append(conn)
            
            for http_err in http_errors_list:
                if http_err['error']['severity'] >= 0.8:
                    issues['critical'].append(http_err)
                else:
                    issues['warning'].append(http_err)
            
            for timeout in timeout_issues_list:
                if timeout['timeout']['severity'] >= 0.8:
                    issues['critical'].append(timeout)
                else:
                    issues['warning'].append(timeout)
            
            for data_issue in data_issues_list:
                if data_issue['issue']['severity'] >= 0.7:
                    issues['warning'].append(data_issue)
                else:
                    issues['info'].append(data_issue)
            
            for flow_break in flow_breaks_list:
                if flow_break['break']['severity'] >= 0.7:
                    issues['warning'].append(flow_break)
                else:
                    issues['info'].append(flow_break)
            
            # Generate recommendations
            if connection_issues_list:
                recommendations.append({
                    'type': 'connection',
                    'priority': 'high',
                    'message': f"Found {len(connection_issues_list)} connection issues. Check network connectivity and service availability."
                })
            
            if http_errors_list:
                http_4xx = sum(1 for e in http_errors_list if '404' in str(e.get('error', {}).get('errors', [])))
                http_5xx = sum(1 for e in http_errors_list if any('500' in str(err) or '503' in str(err) for err in e.get('error', {}).get('errors', [])))
                recommendations.append({
                    'type': 'http_errors',
                    'priority': 'high' if http_5xx > 0 else 'medium',
                    'message': f"Found {len(http_errors_list)} HTTP errors ({http_4xx} 4xx, {http_5xx} 5xx). Review API endpoints and service health."
                })
            
            if timeout_issues_list:
                recommendations.append({
                    'type': 'timeout',
                    'priority': 'high',
                    'message': f"Found {len(timeout_issues_list)} timeout issues. Consider optimizing slow operations or increasing timeout thresholds."
                })
            
            if flow_breaks_list:
                recommendations.append({
                    'type': 'flow',
                    'priority': 'medium',
                    'message': f"Found {len(flow_breaks_list)} flow breaks. Review transaction flow completeness."
                })
            
            # Flow analysis summary
            flow_analysis = {
                'total_transactions': len(entries_by_txn),
                'transactions_with_issues': len(set(
                    txn_id for issue_list in [
                        connection_issues_list, http_errors_list, timeout_issues_list,
                        data_issues_list, flow_breaks_list
                    ] for item in issue_list for txn_id in [item.get('transaction_id')]
                )),
                'connection_issues': len(connection_issues_list),
                'http_errors': len(http_errors_list),
                'timeouts': len(timeout_issues_list),
                'data_issues': len(data_issues_list),
                'flow_breaks': len(flow_breaks_list)
            }
        
        # Calculate enhanced metrics
        if all_entries:
            process_times = [e.process_time_ms for e in all_entries if e.process_time_ms > 0]
            queue_times = [e.queue_time_ms for e in all_entries if e.queue_time_ms > 0]
            
            if process_times:
                stats.avg_process_time_ms = sum(process_times) / len(process_times)
            if queue_times:
                stats.avg_queue_time_ms = sum(queue_times) / len(queue_times)
            
            stats.success_count = sum(1 for e in all_entries if e.relationship.lower() == 'success')
            stats.failure_count = sum(1 for e in all_entries if e.rule_fail)
            
            # Calculate total duration if we have timestamps
            try:
                timestamps = [datetime.strptime(e.ts.split(',')[0], '%Y-%m-%d %H:%M:%S') 
                            for e in all_entries if e.ts]
                if len(timestamps) >= 2:
                    stats.total_duration_ms = (timestamps[-1] - timestamps[0]).total_seconds() * 1000
            except:
                pass
        
        # Final statistics
        stats.processing_time = time.time() - start_time
        stats.critical_errors = sum(1 for f in all_failures if f.severity_score >= 0.8)
        stats.warnings = sum(1 for f in all_failures if 0.3 <= f.severity_score < 0.8)
        
        logger.info(f"Analysis complete: {stats.rule_failures} rule failures, "
                   f"{stats.ml_failures} ML failures, {stats.processing_time:.2f}s")
        logger.info(f"Enhanced analysis: {stats.connection_issues} connection issues, "
                   f"{stats.http_errors} HTTP errors, {stats.timeout_issues} timeouts")
        
        # Update ml_info with ensemble results
        if ml_model_predictions:
            ml_info.update(ml_model_predictions)
        
        # Add root cause information to failures
        failures_dict = []
        for i, f in enumerate(all_failures):
            failure_dict = f.to_dict()
            if root_cause_analysis:
                # Add root cause flags
                failure_dict['is_root_cause'] = getattr(f, 'is_root_cause', False)
                failure_dict['is_cascade'] = getattr(f, 'is_cascade', False)
                failure_dict['root_cause_type'] = getattr(f, 'root_cause_type', '')
                failure_dict['caused_by'] = getattr(f, 'caused_by', '')
                failure_dict['cascade_count'] = getattr(f, 'cascade_count', 0)
                failure_dict['impact_count'] = getattr(f, 'impact_count', 1)
            failures_dict.append(failure_dict)
        
        # Add primary failures summary
        primary_failures_summary = []
        if primary_failures:
            for pf in primary_failures:
                # Get full message, not truncated
                failure_msg = pf['failure'].get('msg', '')
                if isinstance(failure_msg, str) and len(failure_msg) > 500:
                    # Keep first 500 chars but try to preserve endpoint info
                    failure_msg = failure_msg[:500] + "..."
                
                primary_failures_summary.append({
                    'index': pf['failure_index'],
                    'type': pf['type'],
                    'message': failure_msg,  # Full message, not truncated to 200
                    'component': pf['failure'].get('component', ''),
                    'operation': pf['failure'].get('operation', ''),
                    'severity': pf['severity'],
                    'impact_count': pf['impact_count'],
                    'cascade_count': pf['cascade_count'],
                    'transaction_id': pf['transaction_id'],
                    'root_cause_type': pf.get('root_cause_type', pf['type']),
                    'timestamp': pf['failure'].get('ts', ''),
                    'http_status': pf['failure'].get('http_status')
                })
        
        return {
            'stats': stats.to_dict(),
            'failures': failures_dict,
            'ml_info': ml_info,
            'format': getattr(self, '_detected_format', 'unknown'),
            'flow_analysis': flow_analysis,
            'issues': issues,
            'recommendations': recommendations,
            'root_cause_analysis': {
                'primary_failures': primary_failures_summary,
                'total_root_causes': len(root_cause_analysis.get('root_causes', [])),
                'total_cascades': sum(len(chain.cascading_failures) for chain in root_cause_analysis.get('failure_chains', []))
            }
        }
    
    def _process_log_file(self, log_file: Path, stats: AnalysisStats, 
                         max_failures: Optional[int]) -> Tuple[List[LogEntry], List[LogEntry]]:
        """Process log file in batches."""
        all_entries = []
        rule_failures = []
        sample_lines = []
        
        try:
            with log_file.open('r', encoding='utf-8', errors='replace') as f:
                # Read sample for format detection
                current_pos = f.tell()
                for _ in range(20):
                    line = f.readline()
                    if not line:
                        break
                    line_stripped = line.strip()
                    # Skip header lines
                    if line_stripped and not line_stripped.startswith('TIMESTAMP|'):
                        sample_lines.append(line_stripped)
                
                # Detect format
                self._detected_format = self.parser.detect_format(sample_lines)
                if self._detected_format == 'unknown':
                    logger.warning("Could not detect log format, trying all patterns")
                    # Try to detect by checking a sample line directly
                    if sample_lines:
                        test_line = sample_lines[0]
                        if self.parser.patterns.NEW_LOG_FORMAT.match(test_line):
                            self._detected_format = 'new_format'
                        elif self.parser.patterns.YOUR_LOG_FORMAT.match(test_line):
                            self._detected_format = 'your_format'
                        else:
                            self._detected_format = 'new_format'  # Default to new format
                    else:
                        self._detected_format = 'new_format'  # Default to new format
                
                logger.info(f"Using format: {self._detected_format}")
                
                # Reset file pointer and process
                f.seek(current_pos)
                batch = []
                
                for line_num, line in enumerate(f, 1):
                    stats.total_lines += 1
                    batch.append(line)
                    
                    if len(batch) >= self.batch_size:
                        entries, failures = self._process_batch(batch, self._detected_format)
                        all_entries.extend(entries)
                        rule_failures.extend(failures)
                        stats.parsed_entries += len(entries)
                        stats.rule_failures += len(failures)
                        batch = []
                        
                        # Progress reporting
                        if line_num % 10000 == 0:
                            logger.info(f"Processed {line_num:,} lines, "
                                      f"found {stats.rule_failures} failures")
                        
                        # Early termination if max failures reached
                        if max_failures and len(rule_failures) >= max_failures:
                            logger.info(f"Reached max failures limit: {max_failures}")
                            break
                
                # Process remaining batch
                if batch:
                    entries, failures = self._process_batch(batch, self._detected_format)
                    all_entries.extend(entries)
                    rule_failures.extend(failures)
                    stats.parsed_entries += len(entries)
                    stats.rule_failures += len(failures)
                    
        except Exception as e:
            logger.error(f"Error processing log file: {e}")
            raise
        
        return all_entries, rule_failures
    
    def prepare_training_data_from_log(self, log_file: Path, 
                                      use_rule_labels: bool = True) -> Tuple[List[LogEntry], List[bool]]:
        """Prepare training data from log file, using rule-based detection as labels."""
        logger.info(f"Preparing training data from {log_file}...")
        
        all_entries, _ = self._process_log_file(log_file, AnalysisStats(), None)
        
        if use_rule_labels:
            # Use rule-based detection as ground truth labels
            labels = [entry.rule_fail for entry in all_entries]
            logger.info(f"Using rule-based labels: {sum(labels)} failures, {len(labels) - sum(labels)} successes")
        else:
            # All entries marked as non-failure (user should provide labels)
            labels = [False] * len(all_entries)
            logger.warning("No labels provided - all entries marked as non-failure")
        
        return all_entries, labels
    
    def _process_batch(self, batch: List[str], format_type: str) -> Tuple[List[LogEntry], List[LogEntry]]:
        """Process a batch of log lines."""
        entries = []
        failures = []
        
        for line in batch:
            entry = self.parser.parse_line(line, format_type)
            if entry:
                entries.append(entry)
                if entry.rule_fail:
                    failures.append(entry)
        
        return entries, failures
    
    def _combine_failures(self, rule_failures: List[LogEntry], ml_failures: List[LogEntry],
                         max_failures: Optional[int]) -> List[LogEntry]:
        """Combine and deduplicate failures."""
        # Use timestamp + message as unique key
        seen = set()
        combined = []
        
        # Add rule failures first (higher priority)
        for failure in rule_failures:
            key = (failure.ts, failure.msg[:100])  # First 100 chars to handle long messages
            if key not in seen:
                seen.add(key)
                combined.append(failure)
        
        # Add ML failures that aren't duplicates
        for failure in ml_failures:
            key = (failure.ts, failure.msg[:100])
            if key not in seen:
                seen.add(key)
                combined.append(failure)
        
        # Sort by severity score (descending) then by timestamp
        combined.sort(key=lambda x: (-x.severity_score, x.ts))
        
        # Apply max failures limit
        if max_failures:
            combined = combined[:max_failures]
        
        return combined

# ==================== CLI INTERFACE ====================
def train_model(log_file: Path, model_path: Path, model_type: str, 
                use_rule_labels: bool, test_size: float, max_features: int) -> int:
    """Train a supervised learning model on log data."""
    try:
        import sys
        import io
        
        # Set UTF-8 encoding for Windows console
        if sys.platform == 'win32':
            try:
                sys.stdout.reconfigure(encoding='utf-8')
                sys.stderr.reconfigure(encoding='utf-8')
            except:
                pass
        
        logger.info("=" * 80)
        logger.info("TRAINING SUPERVISED LEARNING MODEL")
        logger.info("=" * 80)
        
        # Initialize analyzer
        analyzer = LogAnalyzer()
        
        # Prepare training data
        entries, labels = analyzer.prepare_training_data_from_log(log_file, use_rule_labels)
        
        if len(entries) < 10:
            print(f"\n[ERROR] Insufficient data for training: {len(entries)} entries (minimum 10 required)")
            return 1
        
        # Check label distribution
        failure_count = sum(labels)
        success_count = len(labels) - failure_count
        
        print(f"\nTraining Data Summary:")
        print(f"   Total entries: {len(entries):,}")
        print(f"   Failures: {failure_count:,} ({failure_count/len(entries)*100:.1f}%)")
        print(f"   Successes: {success_count:,} ({success_count/len(entries)*100:.1f}%)")
        
        if failure_count == 0:
            print("\n[WARNING] No failures found in training data!")
            print("   The model may not learn to detect failures effectively.")
            print("   Continuing with training anyway...")
        
        if success_count == 0:
            print("\n[WARNING] No successes found in training data!")
            print("   The model may not learn to distinguish failures from successes.")
            print("   Continuing with training anyway...")
        
        # Create and train classifier
        classifier = SupervisedErrorClassifier(model_type=model_type, max_features=max_features)
        metrics = classifier.train(entries, labels, test_size=test_size, cross_validate=True)
        
        # Save model
        classifier.save_model(model_path)
        
        # Print results
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE!")
        print("=" * 80)
        print(f"Model saved to: {model_path}")
        print(f"\nModel Performance Metrics:")
        print(f"   Accuracy:  {metrics['accuracy']:.3f} ({metrics['accuracy']*100:.1f}%)")
        print(f"   Precision: {metrics['precision']:.3f}")
        print(f"   Recall:    {metrics['recall']:.3f}")
        print(f"   F1 Score:  {metrics['f1_score']:.3f}")
        
        if metrics.get('cv_mean'):
            print(f"   CV F1:     {metrics['cv_mean']:.3f} (+/- {metrics['cv_std']:.3f})")
        
        print(f"\nTraining Details:")
        print(f"   Model Type: {model_type}")
        print(f"   Training samples: {metrics['train_samples']:,}")
        print(f"   Test samples: {metrics['test_samples']:,}")
        print(f"   Features used: {metrics['features_used']:,}")
        
        print(f"\nUsage:")
        print(f"   python python_logs_analyser.py --model {model_path} --logfile <your_log_file>")
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        if logging.getLogger().level == logging.DEBUG:
            import traceback
            traceback.print_exc()
        return 1

def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Enhanced log analyzer with supervised learning training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis
  python python_logs_analyser.py
  
  # Train a model
  python python_logs_analyser.py --train --logfile server.log --model trained_model.pkl
  
  # Use trained model for analysis
  python python_logs_analyser.py --model trained_model.pkl --logfile server.log
  
  # With ML clustering (unsupervised)
  python python_logs_analyser.py --ml 5 --threshold 0.25
  
Your log format:
  TIMESTAMP|NUM|NUM|UUID|COMPONENT|PJMS|failure/success|UUID|OPERATION|TXN_ID|MESSAGE
        """
    )
    
    parser.add_argument('--logfile', type=Path, default=Path('server.log'), 
                       help='Log file to analyze (default: server.log)')
    parser.add_argument('-o', '--output', type=Path, default=Path('analysis_report.json'),
                       help='Output JSON report file (default: analysis_report.json)')
    parser.add_argument('--max-failures', type=int, 
                       help='Maximum number of failures to include in output')
    parser.add_argument('--batch-size', type=int, default=5000,
                       help='Batch size for processing (default: 5000)')
    
    # Model options
    parser.add_argument('--model', type=Path, 
                       help='Path to trained model file (for prediction)')
    parser.add_argument('--train', action='store_true',
                       help='Train a new model on the log file')
    parser.add_argument('--model-type', choices=['random_forest', 'logistic_regression', 'svm', 'naive_bayes'],
                       default='random_forest',
                       help='Type of model to train (default: random_forest)')
    parser.add_argument('--train-test-split', type=float, default=0.2,
                       help='Test set size for training (default: 0.2)')
    parser.add_argument('--train-max-features', type=int, default=2000,
                       help='Max features for training (default: 2000)')
    parser.add_argument('--use-rule-labels', action='store_true', default=True,
                       help='Use rule-based detection as training labels (default: True)')
    
    # ML options (clustering-based, for backward compatibility)
    ml_group = parser.add_argument_group('ML Clustering Options (Unsupervised)')
    ml_group.add_argument('--ml', type=int, metavar='K', dest='ml_clusters',
                         help='Enable ML clustering detection with K clusters')
    ml_group.add_argument('--ml-algo', choices=['kmeans', 'dbscan'], default='kmeans',
                         help='ML clustering algorithm (default: kmeans)')
    ml_group.add_argument('--threshold', type=float, default=0.3,
                         help='ML failure detection threshold (default: 0.3)')
    ml_group.add_argument('--max-features', type=int, default=1000,
                         help='Maximum features for text vectorization (default: 1000)')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args(argv)
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Handle training mode
    if args.train:
        if not args.logfile.exists():
            print(f"\n[ERROR] Training log file '{args.logfile}' not found!")
            return 1
        
        model_path = args.model if args.model else Path('trained_model.pkl')
        return train_model(
            log_file=args.logfile,
            model_path=model_path,
            model_type=args.model_type,
            use_rule_labels=args.use_rule_labels,
            test_size=args.train_test_split,
            max_features=args.train_max_features
        )
    
    # Check if server.log exists
    if not args.logfile.exists():
        print(f"\n[WARNING] Log file '{args.logfile}' not found!")
        print("Please create the file and copy-paste your log errors into it.")
        print(f"   You can create it with: touch {args.logfile}")
        print(f"\nYour log format should be:")
        print("   TIMESTAMP|NUM|NUM|UUID|COMPONENT|PJMS|failure|UUID|OPERATION|TXN_ID|MESSAGE")
        return 1
    
    if not args.logfile.is_file():
        logger.error(f"Path is not a file: {args.logfile}")
        return 1
    
    # Check if file is empty
    if args.logfile.stat().st_size == 0:
        print(f"\nThe file '{args.logfile}' is empty!")
        print("Please copy-paste your log errors into the file and run again.")
        return 1
    
    try:
        # Initialize analyzer with optional trained model
        analyzer = LogAnalyzer(batch_size=args.batch_size, model_path=args.model)
        
        # Configure ML detection
        ml_config = None
        if args.ml_clusters:
            ml_config = {
                'clusters': args.ml_clusters,
                'algorithm': args.ml_algo,
                'threshold': args.threshold,
                'max_features': args.max_features
            }
        
        # Run analysis
        results = analyzer.analyze(args.logfile, args.max_failures, ml_config)
        
        # Save results
        with args.output.open('w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Generate HTML report
        html_output = args.output.with_suffix('.html')
        try:
            import sys as sys_module
            script_dir = Path(__file__).parent
            sys_module.path.insert(0, str(script_dir))
            from generate_html_report import generate_html_report
            generate_html_report(args.output, html_output)
            print(f"\n[INFO] HTML report generated: {html_output}")
        except ImportError:
            # If generate_html_report is not available, skip HTML generation
            logger.debug("HTML report generator not available")
        except Exception as e:
            logger.debug(f"HTML report generation failed: {e}")
        
        # Print summary
        stats = results['stats']
        failures = results['failures']
        
        print(f"\n[SUCCESS] Analysis complete! Results saved to {args.output}")
        print("=" * 80)
        print(f"LOG ANALYSIS SUMMARY")
        print("=" * 80)
        print(f"Analyzed file: {args.logfile}")
        print(f"File size: {stats['file_size_mb']:.1f} MB")
        print(f"Detected format: {results['format']}")
        print(f"Total lines processed: {stats['total_lines']:,}")
        print(f"Parsed entries: {stats['parsed_entries']:,}")
        print(f"Rule-based failures: {stats['rule_failures']:,}")
        print(f"ML-detected failures: {stats['ml_failures']:,}")
        print(f"Critical errors: {stats['critical_errors']:,}")
        print(f"Warnings: {stats['warnings']:,}")
        print(f"Processing time: {stats['processing_time']:.2f}s")
        
        # Show model info if using trained model
        if results.get('ml_info') and 'accuracy' in results['ml_info']:
            ml_info = results['ml_info']
            print(f"\nTrained Model Info:")
            print(f"   Model Type: {ml_info.get('model_type', 'N/A')}")
            print(f"   Accuracy: {ml_info.get('accuracy', 0):.3f} ({ml_info.get('accuracy', 0)*100:.1f}%)")
            print(f"   F1 Score: {ml_info.get('f1_score', 0):.3f}")
            print(f"   High Confidence Predictions: {ml_info.get('high_confidence', 0):,}")
        
        if stats['total_lines'] > 0:
            print(f"Throughput: {stats['total_lines']/stats['processing_time']:.0f} lines/sec")
        
        # Show sample failures
        if failures:
            print(f"\nSAMPLE FAILURES (showing first 3 of {len(failures)}):")
            print("-" * 80)
            for i, failure in enumerate(failures[:3]):
                print(f"\n{i+1}. [{failure['level']}] {failure['ts']}")
                print(f"   Component: {failure['component']}")
                print(f"   Operation: {failure['operation']}")
                print(f"   TXN ID: {failure['txn_id']}")
                print(f"   Severity: {failure['severity_score']:.2f}")
                if failure['keywords_found']:
                    print(f"   Keywords: {', '.join(failure['keywords_found'])}")
                print(f"   Message: {failure['msg'][:200]}{'...' if len(failure['msg']) > 200 else ''}")
        
        print("=" * 80)
        print(f"Full detailed report saved to: {args.output}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())



# # Basic analysis
# python log_analyzer.py

# # With ML detection
# python log_analyzer.py --ml 6

# # Advanced ML with custom settings
# python log_analyzer.py --ml 8 --threshold 0.25 --ml-algo dbscan