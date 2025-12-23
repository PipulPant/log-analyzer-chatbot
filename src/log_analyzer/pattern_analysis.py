#!/usr/bin/env python3
"""
Enhanced Pattern Analysis Module
=================================
Comprehensive error pattern recognition, state machine analysis, timing analysis,
and data consistency checks for log analysis.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple, Any
from datetime import datetime
from collections import defaultdict

# ==================== ERROR PATTERN RECOGNITION ====================

ERROR_KEYWORDS = {
    'connection': [
        'Failed to connect', 'Connection refused', 'Connection timeout',
        'ConnectException', 'SocketException', 'Network is unreachable',
        'No route to host', 'Connection reset', 'Connection closed',
        'Connection lost', 'IHTTP.*Failure'
    ],
    'http': [
        '404', '500', '503', 'Timeout', 'Gateway', 'Not Found',
        'Service Unavailable', 'Bad Gateway', 'Internal Server Error',
        '3P Status Code', 'Status Code'
    ],
    'database': [
        'SQLException', 'ORA-', 'PSQLException', 'JDBC', 'Connection pool',
        'No Record Found', 'Record not found', 'Database error',
        'Query failed', 'Transaction failed'
    ],
    'service': [
        'No such service', 'Endpoint not found', 'WSDL', 'SOAPFault',
        'Service unavailable', 'Service error'
    ],
    'authentication': [
        'Unauthorized', 'Forbidden', 'Authentication failed',
        'Invalid credentials', 'Access denied', 'Token expired',
        'Invalid token'
    ],
    'timeout': [
        'Read timed out', 'Timeout', 'Took too long', 'exceeded threshold',
        'Timeout reached', 'Request timeout', 'Connection timeout'
    ]
}

# ==================== STATE MACHINE ANALYSIS ====================

FLOW_STATES = {
    'START': [
        'Pull Check Max Request', 'Receive Request', 'Pull Request',
        'Fetch Pending', 'Receive request'
    ],
    'PROCESSING': [
        'Validate', 'Transform', 'Convert', 'Prepare', 'Map',
        'Evaluate', 'Inspect', 'Store', 'Extract', 'Set',
        'Separate', 'Update', 'Prepare SQL', 'Map status'
    ],
    'EXTERNAL_CALL': [
        'Call to', 'Invoke', 'HTTP', 'SOAP', 'API', 'Call BSS API',
        'Call MM', 'Call Ext', 'IHTTP', 'Fetch Access Token',
        'Call Result Code Mapping'
    ],
    'RESPONSE': [
        'Response', 'Handle Response', 'Send Response', 'Publish Response',
        'Respond with', 'Prepare Response', 'Handle Result'
    ],
    'ERROR_HANDLING': [
        'Handle Error', 'Save failed', 'Log Error', 'Handle Internal Error',
        'No Retry', 'Handle Result'
    ],
    'END': [
        'Publish Response', 'Send Response', 'Respond with', 'Log Response'
    ]
}

# ==================== TIMING THRESHOLDS ====================

THRESHOLDS = {
    'total_timeout': 30000,  # 30 seconds
    'external_call_timeout': 10000,  # 10 seconds
    'db_query_slow': 1000,  # 1 second
    'processing_slow': 500,  # 500ms
    'queue_time_high': 100,  # 100ms queue time
    'connection_failure_threshold': 10  # Very short process time + failure indicates connection issue
}

# ==================== DATA VALIDATION RULES ====================

DATA_VALIDATION_RULES = {
    'id_consistency': {
        'pattern': r'id[=:]\s*["\']?([A-F0-9-]+)["\']?',
        'should_match_across': ['request', 'response', 'external_call']
    },
    'data_transformation': {
        'fields_to_track': ['IDNUMBER', 'IDTYPE', 'TRXTYPEID', 'MSISDN', 'trxId', 'instanceId'],
        'validate_transforms': True
    },
    'mandatory_fields': ['id', 'name', 'instanceId', 'trxId']
}

# ==================== PATTERN ANALYSIS CLASS ====================

class PatternAnalyzer:
    """Comprehensive pattern analysis for log entries."""
    
    def __init__(self):
        self.error_patterns = self._compile_error_patterns()
        self.state_patterns = self._compile_state_patterns()
        
    def _compile_error_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Compile regex patterns for error detection."""
        patterns = {}
        for category, keywords in ERROR_KEYWORDS.items():
            compiled = []
            for kw in keywords:
                # Check if keyword contains regex special characters (like .*)
                if any(char in kw for char in ['.*', '.*?', '+', '*', '?', '(', ')', '[', ']']):
                    # It's already a regex pattern
                    compiled.append(re.compile(kw, re.IGNORECASE))
                else:
                    # Escape and compile as literal
                    compiled.append(re.compile(rf'\b{re.escape(kw)}\b', re.IGNORECASE))
            patterns[category] = compiled
        return patterns
    
    def _compile_state_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Compile regex patterns for state detection."""
        patterns = {}
        for state, keywords in FLOW_STATES.items():
            patterns[state] = [
                re.compile(rf'\b{re.escape(kw)}\b', re.IGNORECASE)
                for kw in keywords
            ]
        return patterns
    
    def detect_error_patterns(self, text: str, processor_name: str = "") -> Dict[str, Any]:
        """Detect error patterns in text."""
        full_text = f"{text} {processor_name}".lower()
        detected = {
            'categories': [],
            'keywords_found': [],
            'severity': 0.0
        }
        
        for category, patterns in self.error_patterns.items():
            for pattern in patterns:
                if pattern.search(full_text):
                    if category not in detected['categories']:
                        detected['categories'].append(category)
                    match = pattern.search(full_text)
                    if match:
                        detected['keywords_found'].append(match.group(0))
        
        # Calculate severity based on categories
        if 'connection' in detected['categories']:
            detected['severity'] = max(detected['severity'], 0.9)
        if 'http' in detected['categories']:
            detected['severity'] = max(detected['severity'], 0.8)
        if 'database' in detected['categories']:
            detected['severity'] = max(detected['severity'], 0.7)
        if 'timeout' in detected['categories']:
            detected['severity'] = max(detected['severity'], 0.8)
        
        return detected
    
    def detect_state(self, processor_name: str) -> Optional[str]:
        """Detect the state/phase of processing."""
        processor_lower = processor_name.lower()
        
        for state, patterns in self.state_patterns.items():
            for pattern in patterns:
                if pattern.search(processor_lower):
                    return state
        
        return None
    
    def detect_connection_issues(self, log_entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Detect connection issues."""
        indicators = [
            r'Failed to connect to',
            r'Connection refused',
            r'Network is unreachable',
            r'IHTTP.*Failure',
            r'No route to host',
            r'Connection timeout',
            r'Connection reset'
        ]
        
        process_time = log_entry.get('process_time_ms', 0)
        relationship = log_entry.get('relationship', '').lower()
        processor_name = log_entry.get('processor_name', '')
        log_message = log_entry.get('log_message', '')
        
        full_text = f"{relationship} {processor_name} {log_message}".lower()
        
        # Check for connection failure indicators
        connection_keywords_found = []
        for indicator in indicators:
            if re.search(indicator, full_text, re.IGNORECASE):
                connection_keywords_found.append(indicator)
        
        # Very short process time + failure indicates connection issue
        is_connection_failure = (
            (process_time < THRESHOLDS['connection_failure_threshold'] and 
             relationship in ['failure', 'no retry']) or
            any('IHTTP' in processor_name and 'Failure' in relationship for _ in [True])
        )
        
        if connection_keywords_found or is_connection_failure:
            return {
                'type': 'connection_issue',
                'indicators': connection_keywords_found,
                'process_time_ms': process_time,
                'relationship': relationship,
                'severity': 0.9 if is_connection_failure else 0.7
            }
        
        return None
    
    def detect_http_errors(self, log_entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Detect HTTP error responses."""
        log_message = log_entry.get('log_message', '')
        processor_name = log_entry.get('processor_name', '')
        relationship = log_entry.get('relationship', '')
        
        full_text = f"{log_message} {processor_name} {relationship}"
        
        patterns = {
            '404': [
                r'404.*Not Found',
                r'404.*not available',
                r'Status Code:\s*404',
                r'3P Status Code:\s*404',
                r'"code"\s*:\s*"404"'
            ],
            '500': [
                r'500.*Internal Server Error',
                r'500.*trx failure',
                r'Status Code:\s*500',
                r'"code"\s*:\s*"500"'
            ],
            '503': [
                r'503.*Service Unavailable',
                r'Status Code:\s*503',
                r'"code"\s*:\s*"503"'
            ],
            'timeout': [
                r'Timeout',
                r'timed?\s*out',
                r'Request timeout'
            ],
            'tomcat_error': [
                r'Apache Tomcat.*Error report'
            ]
        }
        
        detected_errors = []
        for error_type, error_patterns in patterns.items():
            for pattern in error_patterns:
                if re.search(pattern, full_text, re.IGNORECASE):
                    detected_errors.append({
                        'type': error_type,
                        'pattern': pattern,
                        'severity': 0.9 if error_type in ['500', '503'] else 0.7
                    })
                    break
        
        if detected_errors:
            return {
                'type': 'http_error',
                'errors': detected_errors,
                'severity': max(e['severity'] for e in detected_errors)
            }
        
        return None
    
    def detect_timeouts(self, log_entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect timeout issues in log sequence."""
        timeouts = []
        
        if not log_entries:
            return timeouts
        
        # Parse timestamps and calculate durations
        timestamps = []
        for entry in log_entries:
            try:
                ts_str = entry.get('timestamp', '')
                if ts_str:
                    ts = datetime.strptime(ts_str.split(',')[0], '%Y-%m-%d %H:%M:%S')
                    timestamps.append((ts, entry))
            except:
                continue
        
        if len(timestamps) < 2:
            return timeouts
        
        # Check total duration
        total_duration_ms = (timestamps[-1][0] - timestamps[0][0]).total_seconds() * 1000
        
        if total_duration_ms > THRESHOLDS['total_timeout']:
            timeouts.append({
                'type': 'total_timeout',
                'duration_ms': total_duration_ms,
                'threshold_ms': THRESHOLDS['total_timeout'],
                'severity': 0.9,
                'start_time': timestamps[0][0].isoformat(),
                'end_time': timestamps[-1][0].isoformat()
            })
        
        # Check individual step timeouts
        for i in range(len(timestamps) - 1):
            duration_ms = (timestamps[i+1][0] - timestamps[i][0]).total_seconds() * 1000
            entry = timestamps[i][1]
            process_time = entry.get('process_time_ms', 0)
            
            processor_name = entry.get('processor_name', '')
            
            # Check external call timeout
            if any(state in processor_name for state in FLOW_STATES['EXTERNAL_CALL']):
                if duration_ms > THRESHOLDS['external_call_timeout']:
                    timeouts.append({
                        'type': 'external_call_timeout',
                        'processor': processor_name,
                        'duration_ms': duration_ms,
                        'threshold_ms': THRESHOLDS['external_call_timeout'],
                        'severity': 0.8
                    })
            
            # Check processing timeout
            if process_time > THRESHOLDS['processing_slow']:
                timeouts.append({
                    'type': 'processing_slow',
                    'processor': processor_name,
                    'process_time_ms': process_time,
                    'threshold_ms': THRESHOLDS['processing_slow'],
                    'severity': 0.6
                })
            
            # Check queue time
            queue_time = entry.get('queue_time_ms', 0)
            if queue_time > THRESHOLDS['queue_time_high']:
                timeouts.append({
                    'type': 'queue_time_high',
                    'processor': processor_name,
                    'queue_time_ms': queue_time,
                    'threshold_ms': THRESHOLDS['queue_time_high'],
                    'severity': 0.5
                })
        
        return timeouts
    
    def detect_data_issues(self, log_entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect data transformation and consistency issues."""
        issues = []
        
        if not log_entries:
            return issues
        
        # Extract IDs from entries
        id_pattern = DATA_VALIDATION_RULES['id_consistency']['pattern']
        ids_found = []
        
        for entry in log_entries:
            log_message = entry.get('log_message', '')
            matches = re.findall(id_pattern, log_message, re.IGNORECASE)
            if matches:
                ids_found.extend(matches)
        
        # Check ID consistency
        if ids_found:
            unique_ids = set(ids_found)
            if len(unique_ids) > 1:
                # Check if IDs are related (e.g., one is prefix of another)
                id_issues = []
                id_list = list(unique_ids)
                for i, id1 in enumerate(id_list):
                    for id2 in id_list[i+1:]:
                        if id1 != id2 and not (id1 in id2 or id2 in id1):
                            id_issues.append(f"{id1} vs {id2}")
                
                if id_issues:
                    issues.append({
                        'type': 'id_inconsistency',
                        'ids': list(unique_ids),
                        'issues': id_issues,
                        'severity': 0.7
                    })
        
        # Check for missing mandatory fields
        for entry in log_entries:
            log_message = entry.get('log_message', '')
            missing_fields = []
            
            for field in DATA_VALIDATION_RULES['mandatory_fields']:
                pattern = rf'\b{field}\s*[=:]\s*["\']?([^,"\']+)["\']?'
                if not re.search(pattern, log_message, re.IGNORECASE):
                    # Check if field exists but is empty
                    empty_pattern = rf'\b{field}\s*[=:]\s*["\']?\s*["\']?[,}}]'
                    if re.search(empty_pattern, log_message, re.IGNORECASE):
                        missing_fields.append(f"{field} (empty)")
                    else:
                        missing_fields.append(field)
            
            if missing_fields:
                issues.append({
                    'type': 'missing_mandatory_fields',
                    'processor': entry.get('processor_name', ''),
                    'missing_fields': missing_fields,
                    'severity': 0.6
                })
        
        return issues
    
    def detect_flow_breaks(self, log_entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect breaks in expected transaction flow."""
        breaks = []
        
        if not log_entries:
            return breaks
        
        # Expected sequence for a complete transaction
        expected_sequence = [
            ('START', FLOW_STATES['START']),
            ('PROCESSING', FLOW_STATES['PROCESSING']),
            ('EXTERNAL_CALL', FLOW_STATES['EXTERNAL_CALL']),
            ('RESPONSE', FLOW_STATES['RESPONSE']),
            ('END', FLOW_STATES['END'])
        ]
        
        # Extract actual sequence
        actual_states = []
        for entry in log_entries:
            processor_name = entry.get('processor_name', '')
            state = self.detect_state(processor_name)
            if state:
                actual_states.append(state)
        
        # Check for missing critical steps
        found_states = set(actual_states)
        
        for state_name, state_keywords in expected_sequence:
            # Check if any keyword from this state appears
            found = False
            for entry in log_entries:
                processor_name = entry.get('processor_name', '').lower()
                for keyword in state_keywords:
                    if keyword.lower() in processor_name:
                        found = True
                        break
                if found:
                    break
            
            if not found and state_name in ['START', 'EXTERNAL_CALL', 'RESPONSE']:
                breaks.append({
                    'type': 'missing_step',
                    'missing_state': state_name,
                    'severity': 0.7 if state_name == 'EXTERNAL_CALL' else 0.5
                })
        
        # Check for error handling without proper response
        has_error_handling = any(
            self.detect_state(e.get('processor_name', '')) == 'ERROR_HANDLING'
            for e in log_entries
        )
        has_response = any(
            self.detect_state(e.get('processor_name', '')) == 'RESPONSE'
            for e in log_entries
        )
        
        if has_error_handling and not has_response:
            breaks.append({
                'type': 'error_without_response',
                'severity': 0.8
            })
        
        return breaks

