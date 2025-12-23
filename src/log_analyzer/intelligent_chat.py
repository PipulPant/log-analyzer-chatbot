#!/usr/bin/env python3
"""
Intelligent Chat Handler
=======================
Provides intelligent conversation handling similar to ChatGPT/DeepSeek.
"""

import re
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class IntelligentChatHandler:
    """Handles intelligent conversation and query understanding."""
    
    def __init__(self, llm_enhancer=None):
        """
        Initialize intelligent chat handler.
        
        Args:
            llm_enhancer: Optional LLMEnhancer instance for AI-powered responses
        """
        self.llm_enhancer = llm_enhancer
        self.query_patterns = {
            'question': [
                r'^(what|why|how|when|where|who|which|can|could|should|would|is|are|was|were|do|does|did)\s+',
                r'\?$',
                r'^(explain|describe|tell me|show me|help me|give me)',
                r'^(what\'s|whats|what is|what was|what are)',
            ],
            'follow_up': [
                r'^(and|also|what about|how about|tell me more|more details|explain|why|how|what)',
                r'^(what|which|where|when)\s+(about|is|was|are)',
                r'^(can you|could you|please)',
                r'^(this|that|it|the error|the failure|the issue)',
            ],
            'reference': [
                r'\b(this|that|it|the error|the failure|the issue|the problem|the endpoint|the component)\b',
            ],
            'comparison': [
                r'compare|difference|same|different|similar|versus|vs',
            ],
            'analysis_request': [
                r'analyze|check|review|examine|investigate|look at',
            ],
            'log_content': [
                r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}',  # Timestamp pattern
                r'\|\s*\d+\s*\|',  # Pipe-separated log format
                r'(ERROR|WARN|INFO|DEBUG|FATAL)',
                r'Status Code:|3P Status Code:|HTTP Status',
            ],
        }
    
    def classify_query(self, query: str, conversation_context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Classify user query to understand intent with improved intelligence.
        
        Returns:
            {
                'type': 'question' | 'log_analysis' | 'follow_up' | 'comparison',
                'intent': specific intent,
                'entities': extracted entities,
                'needs_analysis': bool
            }
        """
        query_lower = query.lower().strip()
        query_original = query.strip()
        
        # Check if it's log content (more strict check)
        is_log_content = any(
            re.search(pattern, query_original) 
            for pattern in self.query_patterns['log_content']
        ) and len(query_original) > 50  # Must be substantial log content
        
        if is_log_content:
            return {
                'type': 'log_analysis',
                'intent': 'analyze_logs',
                'entities': {},
                'needs_analysis': True
            }
        
        # Check for references (this, that, it, the error, etc.) - indicates follow-up
        has_reference = any(
            re.search(pattern, query_lower, re.IGNORECASE)
            for pattern in self.query_patterns['reference']
        )
        
        # If has reference and has context, it's definitely a follow-up
        if has_reference and conversation_context and conversation_context.get('previous_failures'):
            return {
                'type': 'follow_up',
                'intent': 'reference_question',
                'entities': self._extract_entities(query_lower, conversation_context),
                'needs_analysis': False
            }
        
        # Check for questions (improved detection)
        is_question = any(
            re.search(pattern, query_lower, re.IGNORECASE)
            for pattern in self.query_patterns['question']
        ) or query_original.endswith('?')
        
        if is_question:
            # Extract question type
            intent = self._extract_question_intent(query_lower)
            entities = self._extract_entities(query_lower, conversation_context)
            
            # If there's context, treat as follow-up question
            if conversation_context and conversation_context.get('previous_failures'):
                return {
                    'type': 'follow_up',
                    'intent': intent,
                    'entities': entities,
                    'needs_analysis': False
                }
            
            return {
                'type': 'question',
                'intent': intent,
                'entities': entities,
                'needs_analysis': False,
                'is_follow_up': conversation_context is not None
            }
        
        # Check for follow-up patterns
        is_follow_up = any(
            re.search(pattern, query_lower, re.IGNORECASE)
            for pattern in self.query_patterns['follow_up']
        )
        
        if is_follow_up and conversation_context:
            return {
                'type': 'follow_up',
                'intent': 'follow_up_question',
                'entities': self._extract_entities(query_lower, conversation_context),
                'needs_analysis': False
            }
        
        # Check for comparison
        is_comparison = any(
            re.search(pattern, query_lower, re.IGNORECASE)
            for pattern in self.query_patterns['comparison']
        )
        
        if is_comparison:
            return {
                'type': 'comparison',
                'intent': 'compare_analyses',
                'entities': {},
                'needs_analysis': False
            }
        
        # If short query with context, likely a question
        if len(query_original) < 100 and conversation_context and conversation_context.get('previous_failures'):
            return {
                'type': 'follow_up',
                'intent': 'general_question',
                'entities': self._extract_entities(query_lower, conversation_context),
                'needs_analysis': False
            }
        
        # Default: treat as log analysis or general query
        return {
            'type': 'general',
            'intent': 'general_query',
            'entities': {},
            'needs_analysis': len(query) > 200  # Long text likely log content
        }
    
    def _extract_question_intent(self, query: str) -> str:
        """Extract specific intent from question."""
        if re.search(r'\b(what|which)\s+(is|was|are|were|does|did)', query):
            return 'what_is'
        elif re.search(r'\bwhy\b', query):
            return 'why'
        elif re.search(r'\bhow\b', query):
            return 'how'
        elif re.search(r'\bwhen\b', query):
            return 'when'
        elif re.search(r'\bwhere\b', query):
            return 'where'
        elif re.search(r'\b(can|could|should|would)\b', query):
            return 'capability'
        elif re.search(r'\b(explain|describe|tell me|show me)\b', query):
            return 'explain'
        return 'general_question'
    
    def _extract_entities(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Extract entities from query with context awareness."""
        entities = {}
        
        # Extract component names
        component_patterns = [
            r'\b(IHTTP|HTTP|DB|PSQL|ROA|UA|EGS|CJSONTSQL|LC)\b',
            r'\b(component|service|endpoint|api)\s+([A-Za-z0-9_-]+)',
        ]
        for pattern in component_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                entities['component'] = match.group(1) if len(match.groups()) == 1 else match.group(2)
        
        # Extract endpoint/URL
        url_match = re.search(r'(http[s]?://[^\s]+|/[^\s/]+)', query)
        if url_match:
            entities['endpoint'] = url_match.group(1)
        
        # Extract error codes
        error_code_match = re.search(r'\b(40[0-9]|50[0-9])\b', query)
        if error_code_match:
            entities['error_code'] = error_code_match.group(1)
        
        # Extract transaction ID
        txn_match = re.search(r'\b(transaction|txn)[\s_-]*(id|ID)?[\s:]*([A-Za-z0-9-]+)', query, re.IGNORECASE)
        if txn_match:
            entities['transaction_id'] = txn_match.group(3)
        
        # If context available, try to resolve references
        if context:
            # Resolve "this", "that", "it", "the error", "the failure"
            if re.search(r'\b(this|that|it|the error|the failure|the issue|the problem)\b', query.lower()):
                prev_failures = context.get('previous_failures', [])
                if prev_failures:
                    # Use the most recent failure
                    pf = prev_failures[0]
                    if not entities.get('component'):
                        entities['component'] = pf.get('component', '')
                    if not entities.get('endpoint'):
                        # Try to extract endpoint from previous failure message
                        msg = str(pf.get('message', ''))
                        url_match = re.search(r'(http[s]?://[^\s]+|/[^\s/]+)', msg)
                        if url_match:
                            entities['endpoint'] = url_match.group(1)
                    if not entities.get('error_code'):
                        entities['error_code'] = str(pf.get('http_status', ''))
        
        return entities
    
    def generate_intelligent_response(
        self, 
        query: str, 
        query_type: Dict[str, Any],
        analysis_results: Optional[Dict[str, Any]] = None,
        conversation_context: Optional[Dict] = None
    ) -> str:
        """Generate intelligent response based on query type and context."""
        
        if query_type['type'] == 'log_analysis' or query_type['needs_analysis']:
            # This will be handled by format_analysis_response
            return None  # Signal to use normal analysis
        
        if query_type['type'] == 'question':
            return self._answer_question(query, query_type, analysis_results, conversation_context)
        
        if query_type['type'] == 'follow_up':
            return self._handle_follow_up(query, query_type, analysis_results, conversation_context)
        
        if query_type['type'] == 'comparison':
            return self._handle_comparison(query, query_type, conversation_context)
        
        return self._handle_general_query(query, query_type, conversation_context)
    
    def _answer_question(
        self, 
        query: str, 
        query_type: Dict[str, Any],
        analysis_results: Optional[Dict[str, Any]],
        conversation_context: Optional[Dict]
    ) -> str:
        """Answer questions intelligently with full context."""
        intent = query_type['intent']
        entities = query_type['entities']
        query_lower = query.lower()
        
        # Use conversation context if available (even without current analysis_results)
        if conversation_context:
            prev_failures = conversation_context.get('previous_failures', [])
            prev_endpoints = conversation_context.get('previous_endpoints', set())
            prev_components = conversation_context.get('previous_components', set())
            prev_errors = conversation_context.get('previous_errors', [])
            last_analysis = conversation_context.get('last_analysis')
            
            # Use last_analysis if available, otherwise use analysis_results
            if not analysis_results and last_analysis:
                analysis_results = last_analysis
            
            # Answer based on intent
            if intent == 'what_is':
                if 'endpoint' in entities:
                    endpoint = entities['endpoint']
                    if endpoint in prev_endpoints:
                        # Find details about this endpoint
                        endpoint_info = []
                        for pf in prev_failures:
                            msg = str(pf.get('message', ''))
                            if endpoint in msg or endpoint in str(pf.get('component', '')):
                                endpoint_info.append(f"- **Status:** Failed with HTTP {pf.get('http_status', '404')}")
                                endpoint_info.append(f"- **Component:** {pf.get('component', 'Unknown')}")
                                endpoint_info.append(f"- **Time:** {pf.get('timestamp', 'Unknown')}")
                                endpoint_info.append(f"- **Impact:** {pf.get('impact_count', 0)} operations affected")
                                break
                        if endpoint_info:
                            return f"**Endpoint `{endpoint}`** was identified as failing in the previous analysis:\n\n" + "\n".join(endpoint_info)
                        return f"**Endpoint `{endpoint}`** was identified as failing in the previous analysis. It returned a 404 error, indicating the resource is not available on the server."
                    return f"I don't have information about endpoint `{endpoint}` in the current conversation context. Please analyze logs first."
                
                if 'component' in entities:
                    component = entities['component']
                    if component in prev_components:
                        # Find component details
                        component_failures = [pf for pf in prev_failures if pf.get('component') == component]
                        if component_failures:
                            pf = component_failures[0]
                            return f"**Component `{component}`** was involved in failures:\n\n- **Failure Type:** {pf.get('root_cause_type', pf.get('type', 'Unknown'))}\n- **Operation:** {pf.get('operation', 'Unknown')}\n- **Impact:** {pf.get('impact_count', 0)} operations affected\n- **Time:** {pf.get('timestamp', 'Unknown')}"
                        return f"**Component `{component}`** was involved in failures in the previous analysis. Check the previous analysis results for details."
                    return f"**Component `{component}`** is part of the system architecture. Based on the logs analyzed, it handles various operations including API calls and data processing."
            
            elif intent == 'why':
                if prev_failures:
                    pf = prev_failures[0]
                    root_type = pf.get('root_cause_type', pf.get('type', 'unknown'))
                    msg = str(pf.get('message', ''))
                    component = pf.get('component', 'Unknown')
                    operation = pf.get('operation', '')
                    
                    if 'timeout' in root_type.lower() or 'timeout' in msg.lower():
                        return f"**Why the timeout occurred:**\n\nThe service took longer than the configured timeout threshold to respond.\n\n**Component:** {component}\n**Operation:** {operation}\n**Time:** {pf.get('timestamp', 'Unknown')}\n\n**Possible causes:**\n- Service overload\n- Network latency\n- Database query performance issues\n- Resource constraints\n\nCheck the previous analysis for specific timeout details."
                    elif '404' in msg or 'not found' in msg.lower():
                        endpoint = pf.get('component', '')
                        # Extract endpoint from message if available
                        url_match = re.search(r'(http[s]?://[^\s]+|/[^\s/]+)', msg)
                        endpoint_url = url_match.group(1) if url_match else endpoint
                        return f"**Why the 404 error occurred:**\n\nThe endpoint does not exist on the server.\n\n**Endpoint:** {endpoint_url}\n**Component:** {component}\n**Operation:** {operation}\n**Time:** {pf.get('timestamp', 'Unknown')}\n\n**Possible reasons:**\n- Application not deployed\n- Wrong URL path\n- Service not started\n- Context path mismatch\n- Service endpoint not configured"
                    return f"**Why this failure occurred:**\n\nBased on root cause analysis, this is a **{root_type}** failure.\n\n**Details:**\n- **Component:** {component}\n- **Operation:** {operation}\n- **Impact:** {pf.get('impact_count', 0)} operations affected\n- **Time:** {pf.get('timestamp', 'Unknown')}\n- **Message:** {msg[:200] if len(msg) > 200 else msg}\n\nCheck the previous analysis results for complete details."
                return "I don't have previous failure information. Please analyze logs first to get root cause details."
            
            elif intent == 'how':
                if 'fix' in query_lower or 'resolve' in query_lower or 'solve' in query_lower:
                    if prev_failures:
                        pf = prev_failures[0]
                        root_type = pf.get('root_cause_type', '').lower()
                        msg = str(pf.get('message', ''))
                        component = pf.get('component', 'Unknown')
                        response = f"**How to fix:**\n\n**Root Cause:** {pf.get('root_cause_type', pf.get('type', 'Unknown'))}\n**Component:** {component}\n\n"
                        if '404' in root_type or 'not found' in msg.lower():
                            response += "**Steps to resolve:**\n1. Verify the endpoint/service is deployed and running\n2. Check the URL path matches the deployed application context\n3. Ensure the service is started and accessible\n4. Review deployment logs for errors\n5. Verify the service configuration matches the expected endpoint\n6. Check if the service needs to be restarted"
                        elif 'timeout' in root_type or 'timeout' in msg.lower():
                            response += "**Steps to resolve:**\n1. Check service performance and resource usage\n2. Review database query performance\n3. Increase timeout thresholds if appropriate\n4. Scale up resources if needed\n5. Check network connectivity\n6. Review service logs for performance bottlenecks"
                        elif 'auth' in root_type or '401' in msg or '403' in msg:
                            response += "**Steps to resolve:**\n1. Verify authentication credentials\n2. Check API keys and tokens\n3. Review access control lists\n4. Ensure proper permissions are configured\n5. Check authentication service status"
                        else:
                            response += "**Steps to resolve:**\n1. Review the 'Possible Issues' section from previous analysis\n2. Check the 'Recommended Actions' section for prioritized fixes\n3. Verify the endpoint/service is deployed and running\n4. Check network connectivity and firewall rules\n5. Review server logs for additional details\n6. Check service health and status"
                        return response
                    return "**How to fix:**\n1. Review the 'Possible Issues' section for specific troubleshooting steps\n2. Check the 'Recommended Actions' section for prioritized fixes\n3. Verify the endpoint/service is deployed and running\n4. Check network connectivity and firewall rules\n5. Review server logs for additional details"
                return "**How it works:** The log analysis system uses ML models and pattern recognition to identify failures, trace root causes, and provide actionable recommendations."
            
            elif intent == 'explain':
                if prev_failures:
                    pf = prev_failures[0]
                    msg = pf.get('message', '')
                    component = pf.get('component', '')
                    operation = pf.get('operation', '')
                    root_type = pf.get('root_cause_type', pf.get('type', 'Unknown'))
                    return f"**Explanation:**\n\n**Failure Type:** {root_type}\n**Component:** {component}\n**Operation:** {operation}\n**Impact:** {pf.get('impact_count', 0)} operations affected\n**Time:** {pf.get('timestamp', 'Unknown')}\n\n**Error Details:**\n{msg[:500] if len(msg) > 500 else msg}\n\nThis failure was identified as the root cause. It affected {pf.get('impact_count', 0)} operations and may have cascaded to other components."
                return "I don't have previous failure information to explain. Please analyze logs first."
        
        # Default response
        return "I can help you understand the log analysis results. Please ask specific questions about:\n- Components and endpoints\n- Error causes\n- How to fix issues\n- Comparison with previous logs\n\nOr upload/paste log content for analysis."
    
    def _handle_follow_up(
        self,
        query: str,
        query_type: Dict[str, Any],
        analysis_results: Optional[Dict[str, Any]],
        conversation_context: Optional[Dict]
    ) -> str:
        """Handle follow-up questions with intelligent context understanding."""
        if not conversation_context:
            return "I don't have previous context. Please provide log content for analysis first."
        
        # Use last_analysis if available
        if not analysis_results and conversation_context:
            analysis_results = conversation_context.get('last_analysis')
        
        query_lower = query.lower()
        prev_failures = conversation_context.get('previous_failures', [])
        
        # Handle references (this, that, it, the error, etc.)
        if re.search(r'\b(this|that|it|the error|the failure|the issue|the problem)\b', query_lower):
            if prev_failures:
                pf = prev_failures[0]
                msg = str(pf.get('message', ''))
                component = pf.get('component', 'Unknown')
                root_type = pf.get('root_cause_type', pf.get('type', 'Unknown'))
                
                # Determine what they're asking about
                if 'why' in query_lower:
                    return self._answer_question(query, query_type, analysis_results, conversation_context)
                elif 'what' in query_lower or 'explain' in query_lower:
                    return f"**About the failure:**\n\n**Type:** {root_type}\n**Component:** {component}\n**Operation:** {pf.get('operation', 'Unknown')}\n**Impact:** {pf.get('impact_count', 0)} operations affected\n**Time:** {pf.get('timestamp', 'Unknown')}\n\n**Error Message:**\n{msg[:400] if len(msg) > 400 else msg}"
                elif 'how' in query_lower and ('fix' in query_lower or 'resolve' in query_lower):
                    return self._answer_question(query, query_type, analysis_results, conversation_context)
                else:
                    return f"**The failure:**\n\n**Root Cause:** {root_type}\n**Component:** {component}\n**Message:** {msg[:300] if len(msg) > 300 else msg}\n\nWhat would you like to know more about? You can ask:\n- Why did this happen?\n- How to fix it?\n- What is the endpoint?\n- More details?"
        
        if 'more' in query_lower or 'details' in query_lower:
            if analysis_results:
                failures = analysis_results.get('failures', [])
                if failures:
                    pf = analysis_results.get('root_cause_analysis', {}).get('primary_failures', [{}])
                    impact = pf[0].get('impact_count', 0) if pf else 0
                    return f"**More Details:**\n\nThere are {len(failures)} total failures detected. The primary root cause affects {impact} operations. Check the detailed failure analysis section for complete information."
            elif prev_failures:
                pf = prev_failures[0]
                msg = str(pf.get('message', ''))
                return f"**More Details:**\n\n**Root Cause:** {pf.get('type', 'Unknown')}\n**Component:** {pf.get('component', 'Unknown')}\n**Operation:** {pf.get('operation', 'Unknown')}\n**Impact:** {pf.get('impact_count', 0)} operations affected\n**Time:** {pf.get('timestamp', 'Unknown')}\n\n**Full Error Message:**\n{msg[:500] if len(msg) > 500 else msg}"
            return "I don't have detailed analysis results. Please analyze logs first."
        
        if 'why' in query_lower:
            return self._answer_question(query, query_type, analysis_results, conversation_context)
        
        if 'what' in query_lower:
            return self._answer_question(query, query_type, analysis_results, conversation_context)
        
        if 'how' in query_lower:
            return self._answer_question(query, query_type, analysis_results, conversation_context)
        
        # Generic follow-up response
        if prev_failures:
            pf = prev_failures[0]
            return f"Based on the previous analysis, I can provide more information about:\n\n- **Root Cause:** {pf.get('root_cause_type', pf.get('type', 'Unknown'))}\n- **Component:** {pf.get('component', 'Unknown')}\n- **Impact:** {pf.get('impact_count', 0)} operations affected\n\nWhat specifically would you like to know more about? You can ask:\n- Why did this happen?\n- How to fix it?\n- What is the endpoint/component?\n- More details about the error?"
        
        return "Based on the previous analysis, I can provide more information. What specifically would you like to know more about?"
    
    def _handle_comparison(
        self,
        query: str,
        query_type: Dict[str, Any],
        conversation_context: Optional[Dict]
    ) -> str:
        """Handle comparison queries."""
        if not conversation_context:
            return "I need previous analysis context to make comparisons. Please analyze logs first."
        
        prev_failures = conversation_context.get('previous_failures', [])
        prev_endpoints = conversation_context.get('previous_endpoints', set())
        prev_components = conversation_context.get('previous_components', set())
        
        if not prev_failures:
            return "No previous failures to compare with."
        
        response = f"**Comparison Summary:**\n\n**Previous Analysis:**\n- **Root Causes:** {len(prev_failures)}\n"
        
        if prev_endpoints:
            endpoints = list(prev_endpoints)[:5]
            response += f"- **Failed Endpoints:** {', '.join(endpoints)}\n"
        
        if prev_components:
            components = list(prev_components)[:3]
            response += f"- **Affected Components:** {', '.join(components)}\n"
        
        response += "\nUpload new logs to compare with this analysis and identify patterns or changes."
        
        return response
    
    def _handle_general_query(
        self,
        query: str,
        query_type: Dict[str, Any],
        conversation_context: Optional[Dict]
    ) -> str:
        """Handle general queries."""
        if conversation_context and conversation_context.get('previous_failures'):
            return "I can help you analyze logs and answer questions about failures. Based on previous analysis, you can ask:\n- Why did the failure occur?\n- How to fix it?\n- What is the endpoint/component?\n- More details about the error?\n\nOr upload/paste new log content for analysis."
        return "I can help you analyze logs and answer questions about failures. Please:\n- Upload a log file\n- Paste log content\n- Ask specific questions about previous analysis\n\nWhat would you like to do?"
