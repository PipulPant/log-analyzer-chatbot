#!/usr/bin/env python3
"""
Root Cause Analysis Module
=========================
Identifies primary failures and traces error cascades to find root causes.
"""

from typing import Dict, List, Set, Optional, Tuple, Any
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass

@dataclass
class FailureChain:
    """Represents a chain of failures with root cause."""
    root_failure_index: int
    cascading_failures: List[int]
    transaction_id: str
    root_cause_type: str
    severity: float
    impact_count: int


class RootCauseAnalyzer:
    """Analyzes failures to identify root causes and error cascades."""
    
    def __init__(self):
        self.root_cause_patterns = {
            'external_api_failure': {
                'indicators': ['404', '500', '503', 'Connection refused', 'Timeout'],
                'severity_multiplier': 1.5,
                'priority': 1
            },
            'connection_failure': {
                'indicators': ['Connection refused', 'Connection timeout', 'Failed to connect'],
                'severity_multiplier': 1.4,
                'priority': 2
            },
            'authentication_failure': {
                'indicators': ['401', '403', 'Unauthorized', 'Authentication failed'],
                'severity_multiplier': 1.3,
                'priority': 3
            },
            'data_validation_failure': {
                'indicators': ['Validation failed', 'Invalid data', 'Missing required'],
                'severity_multiplier': 1.2,
                'priority': 4
            },
            'service_unavailable': {
                'indicators': ['503', 'Service unavailable', 'Endpoint not found'],
                'severity_multiplier': 1.3,
                'priority': 2
            }
        }
    
    def identify_root_causes(self, failures: List[Dict], all_entries: List) -> Dict[str, Any]:
        """
        Identify root causes and trace error cascades.
        
        Returns:
            Dict with:
            - root_causes: List of primary failures
            - failure_chains: List of failure chains
            - cascade_map: Map of failure index to root cause index
        """
        if not failures:
            return {
                'root_causes': [],
                'failure_chains': [],
                'cascade_map': {},
                'primary_failures': []
            }
        
        # Group failures by transaction
        failures_by_txn = defaultdict(list)
        failure_indices = {}
        
        for idx, failure in enumerate(failures):
            txn_id = failure.get('txn_id', '') or failure.get('identifier', '')
            failures_by_txn[txn_id].append((idx, failure))
            failure_indices[idx] = failure
        
        root_causes = []
        failure_chains = []
        cascade_map = {}  # Maps failure index to root cause index
        
        # Analyze each transaction
        for txn_id, txn_failures in failures_by_txn.items():
            if not txn_failures:
                continue
            
            # Sort by timestamp to find earliest failure
            txn_failures_sorted = sorted(
                txn_failures,
                key=lambda x: self._parse_timestamp(x[1].get('ts', ''))
            )
            
            # Identify root cause (earliest failure with highest severity)
            root_idx, root_failure = self._identify_root_failure(txn_failures_sorted)
            
            if root_idx is not None:
                # Find cascading failures (occurring after root)
                root_time = self._parse_timestamp(root_failure.get('ts', ''))
                cascading = []
                
                for idx, failure in txn_failures_sorted:
                    if idx == root_idx:
                        continue
                    fail_time = self._parse_timestamp(failure.get('ts', ''))
                    if fail_time >= root_time:
                        cascading.append(idx)
                        cascade_map[idx] = root_idx
                
                # Determine root cause type
                root_cause_type = self._classify_root_cause(root_failure)
                
                # Calculate impact
                impact_count = len(cascading) + 1
                
                # Create failure chain
                chain = FailureChain(
                    root_failure_index=root_idx,
                    cascading_failures=cascading,
                    transaction_id=txn_id,
                    root_cause_type=root_cause_type,
                    severity=root_failure.get('severity_score', 1.0),
                    impact_count=impact_count
                )
                
                failure_chains.append(chain)
                
                # Mark as root cause
                root_cause_info = {
                    'failure_index': root_idx,
                    'failure': root_failure,
                    'type': root_cause_type,
                    'cascade_count': len(cascading),
                    'impact_count': impact_count,
                    'transaction_id': txn_id,
                    'severity': root_failure.get('severity_score', 1.0)
                }
                root_causes.append(root_cause_info)
        
        # Sort root causes by severity and impact
        root_causes.sort(key=lambda x: (
            -x['severity'],
            -x['impact_count'],
            x['type']
        ))
        
        # Get primary failures (top root causes)
        primary_failures = root_causes[:min(5, len(root_causes))]
        
        return {
            'root_causes': root_causes,
            'failure_chains': failure_chains,
            'cascade_map': cascade_map,
            'primary_failures': primary_failures
        }
    
    def _identify_root_failure(self, txn_failures: List[Tuple[int, Dict]]) -> Tuple[Optional[int], Optional[Dict]]:
        """Identify the root failure in a transaction."""
        if not txn_failures:
            return None, None
        
        # Score each failure for root cause potential
        scored_failures = []
        
        for idx, failure in txn_failures:
            score = failure.get('severity_score', 0.5)
            msg = failure.get('msg', '').lower()
            component = failure.get('component', '').lower()
            operation = failure.get('operation', '').lower()
            
            # Boost score for external calls (often root causes)
            if 'external' in component or 'call' in operation or 'invoke' in operation:
                score *= 1.3
            
            # Boost for HTTP errors (common root causes)
            if failure.get('http_status'):
                http_status = failure.get('http_status')
                if 400 <= http_status < 500:
                    score *= 1.2
                elif 500 <= http_status < 600:
                    score *= 1.4
            
            # Boost for connection issues
            if any(keyword in msg for keyword in ['connection', 'timeout', 'refused']):
                score *= 1.2
            
            # Boost for earliest failures
            position_boost = 1.0 - (txn_failures.index((idx, failure)) * 0.1)
            score *= position_boost
            
            scored_failures.append((score, idx, failure))
        
        # Return highest scoring failure
        scored_failures.sort(key=lambda x: -x[0])
        if scored_failures:
            return scored_failures[0][1], scored_failures[0][2]
        
        return None, None
    
    def _classify_root_cause(self, failure: Dict) -> str:
        """Classify the type of root cause."""
        msg = failure.get('msg', '').lower()
        component = failure.get('component', '').lower()
        http_status = failure.get('http_status')
        
        # Check patterns
        for cause_type, pattern_info in self.root_cause_patterns.items():
            for indicator in pattern_info['indicators']:
                if indicator.lower() in msg or indicator.lower() in component:
                    return cause_type
        
        # Check HTTP status
        if http_status:
            if http_status == 404:
                return 'external_api_failure'
            elif http_status == 500 or http_status == 503:
                return 'service_unavailable'
            elif http_status == 401 or http_status == 403:
                return 'authentication_failure'
        
        # Default classification
        if 'timeout' in msg:
            return 'connection_failure'
        elif 'validation' in msg or 'invalid' in msg:
            return 'data_validation_failure'
        else:
            return 'unknown'
    
    def _parse_timestamp(self, ts_str: str) -> datetime:
        """Parse timestamp string to datetime."""
        if not ts_str:
            return datetime.min
        
        try:
            # Try common formats
            formats = [
                '%Y-%m-%d %H:%M:%S,%f',
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%dT%H:%M:%S',
                '%Y-%m-%dT%H:%M:%S.%f'
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(ts_str, fmt)
                except ValueError:
                    continue
            
            return datetime.min
        except Exception:
            return datetime.min
    
    def enhance_failures_with_root_cause(self, failures: List[Dict], root_cause_analysis: Dict) -> List[Dict]:
        """Enhance failure entries with root cause information."""
        cascade_map = root_cause_analysis.get('cascade_map', {})
        root_causes = {rc['failure_index']: rc for rc in root_cause_analysis.get('root_causes', [])}
        
        enhanced_failures = []
        
        for idx, failure in enumerate(failures):
            enhanced_failure = failure.copy()
            
            # Check if this is a root cause
            if idx in root_causes:
                rc_info = root_causes[idx]
                enhanced_failure['is_root_cause'] = True
                enhanced_failure['root_cause_type'] = rc_info['type']
                enhanced_failure['cascade_count'] = rc_info['cascade_count']
                enhanced_failure['impact_count'] = rc_info['impact_count']
                enhanced_failure['priority'] = 'critical'
            else:
                enhanced_failure['is_root_cause'] = False
            
            # Check if this is a cascading failure
            if idx in cascade_map:
                root_idx = cascade_map[idx]
                if root_idx in root_causes:
                    rc_info = root_causes[root_idx]
                    enhanced_failure['is_cascade'] = True
                    enhanced_failure['root_cause_index'] = root_idx
                    enhanced_failure['caused_by'] = rc_info['type']
                    enhanced_failure['priority'] = 'high' if failure.get('severity_score', 0) > 0.7 else 'medium'
                else:
                    enhanced_failure['is_cascade'] = False
            else:
                enhanced_failure['is_cascade'] = False
            
            enhanced_failures.append(enhanced_failure)
        
        return enhanced_failures

