#!/usr/bin/env python3
"""
Web Application for Log Analysis Chat Interface
==============================================
Flask-based web application with ChatGPT-style interface for log analysis.
"""

import os
import json
import logging
import socket
import threading
import re
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory, session
from flask_cors import CORS
from werkzeug.utils import secure_filename
import tempfile
import sys
from collections import defaultdict
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from log_analyzer import LogAnalyzer
from log_analyzer.chat_data_collector import ChatDataCollector
from log_analyzer.intelligent_chat import IntelligentChatHandler
from log_analyzer.llm_enhancer import LLMEnhancer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, 
            template_folder='web/templates',
            static_folder='web/static')
app.config['SECRET_KEY'] = os.urandom(24)  # For session management
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'txt', 'log', 'json'}

# Initialize analyzer - lazy loading to speed up startup
analyzer = None
analyzer_lock = threading.Lock()

# Initialize chat data collector for continuous learning
chat_collector = ChatDataCollector(project_root / "data" / "chat_interactions")

# Initialize LLM enhancer (optional - can be configured via environment or config file)
llm_config_path = project_root / "data" / "models" / "llm_config.json"
llm_enhancer = None
if llm_config_path.exists():
    try:
        with llm_config_path.open() as f:
            llm_config = json.load(f)
        llm_enhancer = LLMEnhancer(llm_config)
        logger.info(f"✓ LLM Enhancer initialized: {llm_config.get('provider', 'none')}")
    except Exception as e:
        logger.warning(f"Failed to load LLM config: {e}")

# Initialize intelligent chat handler with LLM enhancer
intelligent_chat = IntelligentChatHandler(llm_enhancer=llm_enhancer)

# Conversation context storage (in-memory, can be persisted)
conversation_contexts = {}
context_lock = threading.Lock()

def get_analyzer():
    """Get or initialize analyzer (lazy loading)."""
    global analyzer
    if analyzer is not None:
        return analyzer
    
    with analyzer_lock:
        if analyzer is not None:
            return analyzer
        
        try:
            ensemble_config_path = project_root / "data" / "models" / "ensemble_config.json"
            if ensemble_config_path.exists():
                with ensemble_config_path.open() as f:
                    ml_config = json.load(f)
                logger.info("Loading ML models (this may take a moment)...")
                analyzer = LogAnalyzer(ml_models_config=ml_config)
                logger.info("✓ Analyzer loaded with ML models")
            else:
                analyzer = LogAnalyzer()
                logger.info("✓ Analyzer loaded without ML models")
        except Exception as e:
            logger.warning(f"Failed to load ML models: {e}")
            analyzer = LogAnalyzer()
    
    return analyzer

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_conversation_context(conversation_id):
    """Get conversation context for a conversation."""
    with context_lock:
        return conversation_contexts.get(conversation_id, {
            'previous_failures': [],
            'previous_endpoints': set(),
            'previous_components': set(),
            'working_components': set(),
            'previous_errors': [],
            'conversation_history': [],
            'last_analysis': None
        })

def update_conversation_context(conversation_id, analysis_results):
    """Update conversation context with new analysis results."""
    with context_lock:
        if conversation_id not in conversation_contexts:
            conversation_contexts[conversation_id] = {
                'previous_failures': [],
                'previous_endpoints': set(),
                'previous_components': set(),
                'working_components': set(),
                'previous_errors': [],
                'conversation_history': [],
                'last_analysis': None  # Store last analysis results
            }
        
        ctx = conversation_contexts[conversation_id]
        
        # Store last analysis results for reference
        ctx['last_analysis'] = analysis_results
        
        # Extract endpoints from failures
        failures = analysis_results.get('failures', [])
        for failure in failures:
            msg = failure.get('msg', '')
            if msg:
                url_match = re.search(r'(http[s]?://[^\s]+|/[^\s/]+)', msg)
                if url_match:
                    ctx['previous_endpoints'].add(url_match.group(1))
            
            component = failure.get('component', '')
            if component:
                ctx['previous_components'].add(component)
        
        # Extract working components (successful operations)
        all_entries = analysis_results.get('all_entries', [])
        for entry in all_entries:
            if isinstance(entry, dict):
                relationship = entry.get('relationship', '').lower()
                component = entry.get('component', '')
                if relationship == 'success' and component:
                    ctx['working_components'].add(component)
        
        # Store primary failures
        primary_failures = analysis_results.get('root_cause_analysis', {}).get('primary_failures', [])
        if primary_failures:
            ctx['previous_failures'].extend(primary_failures[:3])  # Keep top 3
        
        # Store recent errors
        for failure in failures[:5]:
            ctx['previous_errors'].append({
                'component': failure.get('component', ''),
                'error': failure.get('msg', '')[:200],
                'timestamp': failure.get('ts', '')
            })
        
        # Add to conversation history
        ctx['conversation_history'].append({
            'timestamp': datetime.now().isoformat(),
            'type': 'analysis',
            'summary': f"Analyzed logs: {len(failures)} failures, {len(primary_failures)} root causes"
        })

def format_analysis_response(results, conversation_context=None):
    """Format analysis results in the new format: Key Discovery, Root Cause, Possible Issues, What's Working."""
    stats = results.get('stats', {})
    primary_failures = results.get('root_cause_analysis', {}).get('primary_failures', [])
    failures = results.get('failures', [])
    recommendations = results.get('recommendations', [])
    all_entries = results.get('all_entries', [])
    
    response_parts = []
    
    if not primary_failures and not failures:
        response_parts.append("## ✅ No Failures Detected\n")
        response_parts.append("Great news! No failures were detected in the analyzed logs.")
        return "\n".join(response_parts)
    
    # ========== KEY DISCOVERY ==========
    response_parts.append("## 🔍 Key Discovery\n")
    if primary_failures:
        pf = primary_failures[0]
        msg = pf.get('message', '')
        if msg and not isinstance(msg, str):
            msg = str(msg)
        
        # Extract the complete error message with full context
        core_error_parts = []
        
        # Extract HTTP status code
        http_status = None
        status_match = re.search(r'\b(40[0-9]|50[0-9])\b', msg)
        if status_match:
            http_status = status_match.group(1)
        
        # Extract full endpoint path - improved patterns
        endpoint = None
        full_url = None
        
        # Try to find full URL first (most reliable)
        url_patterns = [
            r'URL:\s*(?:GET|POST|PUT|DELETE)\s+[\'"]?(http[s]?://[^\s\'"]+)',  # URL: POST 'http://...'
            r'(http[s]?://[^\s\)\"]+)',  # Full URL anywhere
            r'[\'"]?(http[s]?://[^\s\'"]+)',  # Quoted URL
        ]
        
        for pattern in url_patterns:
            url_match = re.search(pattern, msg)
            if url_match:
                full_url = url_match.group(1) if len(url_match.groups()) > 0 and url_match.group(1) else url_match.group(0)
                full_url = full_url.strip('"\'')
                if full_url and 'http' in full_url and len(full_url) > 10:
                    break
        
        # If no full URL, try to extract path from error messages
        if not full_url:
            # Look for paths in specific error message formats
            path_patterns = [
                r'HTTP Status \d+[^\w]*[/]([^\s\)\"]+)',  # HTTP Status 404 - /path
                r'[/]([a-zA-Z0-9_-]+[/][a-zA-Z0-9_/-]+)',  # /app/service format (multi-segment)
                r'[/]([a-zA-Z0-9_-]+[a-zA-Z0-9_/-]*)',  # /service format
            ]
            
            for pattern in path_patterns:
                path_match = re.search(pattern, msg)
                if path_match:
                    potential_endpoint = '/' + path_match.group(1) if not path_match.group(1).startswith('/') else path_match.group(1)
                    potential_endpoint = potential_endpoint.strip('"\'')
                    # Filter out incomplete paths and ensure it has valid characters
                    if potential_endpoint and len(potential_endpoint) > 3:
                        # Must have alphanumeric characters, not just numbers/dots
                        if re.search(r'[a-zA-Z]', potential_endpoint):
                            endpoint = potential_endpoint
                            break
        
        # Get additional context from the failure entry
        component = pf.get('component', '')
        operation = pf.get('operation', '')
        txn_id = pf.get('transaction_id', '')
        timestamp = pf.get('timestamp', '')
        http_status_code = pf.get('http_status')
        
        # Try to get full message from original failure entry (not truncated)
        full_msg = msg
        if failures:
            # Find the corresponding failure entry by matching component, operation, timestamp
            for failure in failures:
                fail_component = failure.get('component', '')
                fail_operation = failure.get('operation', '')
                fail_ts = failure.get('ts', '')
                fail_msg = failure.get('msg', '')
                
                # Match by component and operation, or by timestamp
                if ((fail_component == component and fail_operation == operation) or
                    (fail_ts == timestamp and component)):
                    if fail_msg and len(fail_msg) > len(msg):
                        full_msg = fail_msg
                        msg = full_msg
                    break
        
        # Extract HTTP status code
        http_status = None
        if http_status_code:
            http_status = str(http_status_code)
        else:
            status_match = re.search(r'\b(40[0-9]|50[0-9])\b', msg)
            if status_match:
                http_status = status_match.group(1)
        
        # Extract HTTP method
        http_method = None
        method_match = re.search(r'URL:\s*(GET|POST|PUT|DELETE|PATCH)', msg, re.IGNORECASE)
        if method_match:
            http_method = method_match.group(1).upper()
        
        # Extract IP and port
        ip_port = None
        ip_match = re.search(r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}):(\d+)', msg)
        if ip_match:
            ip_port = f"{ip_match.group(1)}:{ip_match.group(2)}"
        
        # Build comprehensive Key Discovery
        # Status Code
        if '3P Status Code:' in msg:
            parts = msg.split('3P Status Code:')
            if len(parts) > 1:
                status_part = parts[1].split('Response:')[0].strip()
                core_error_parts.append(f"3P Status Code: {status_part}")
        elif 'Status Code:' in msg:
            parts = msg.split('Status Code:')
            if len(parts) > 1:
                status_part = parts[1].split('Response:')[0].strip()
                core_error_parts.append(f"Status Code: {status_part}")
        elif http_status:
            core_error_parts.append(f"HTTP Status {http_status}")
        
        # Endpoint information - show full URL
        if full_url:
            core_error_parts.append(f"Endpoint: {full_url}")
        elif endpoint and ip_port:
            core_error_parts.append(f"Endpoint: http://{ip_port}{endpoint}")
        elif endpoint:
            core_error_parts.append(f"Endpoint: {endpoint}")
        
        # Error description
        if 'not available' in msg.lower() or 'not found' in msg.lower():
            if full_url:
                core_error_parts.append(f"Error: HTTP Status {http_status or '404'} – {full_url}")
                core_error_parts.append("The requested resource is not available.")
            elif endpoint:
                core_error_parts.append(f"Error: HTTP Status {http_status or '404'} – {endpoint}")
                core_error_parts.append("The requested resource is not available.")
            else:
                if http_status:
                    core_error_parts.append(f"Error: HTTP Status {http_status}")
                core_error_parts.append("The requested resource is not available.")
        
        # Server information
        if 'Apache Tomcat' in msg:
            tomcat_match = re.search(r'Apache Tomcat/([^\s\)]+)', msg)
            if tomcat_match:
                core_error_parts.append(f"Server: Apache Tomcat/{tomcat_match.group(1)}")
        
        # Component and Operation context
        if component:
            core_error_parts.append(f"Component: {component}")
        if operation:
            core_error_parts.append(f"Operation: {operation}")
        if timestamp:
            core_error_parts.append(f"Timestamp: {timestamp}")
        if txn_id and not txn_id.startswith('Filename: '):
            core_error_parts.append(f"Transaction ID: {txn_id}")
        
        # If we don't have structured parts, use the full message (truncated)
        if not core_error_parts:
            # Try to extract the most relevant part
            if len(msg) > 500:
                # Find the error part
                error_keywords = ['error', 'failed', 'exception', 'status code', '404', '500']
                for keyword in error_keywords:
                    idx = msg.lower().find(keyword.lower())
                    if idx != -1:
                        start = max(0, idx - 50)
                        end = min(len(msg), idx + 300)
                        core_error_parts.append(msg[start:end])
                        break
                if not core_error_parts:
                    core_error_parts.append(msg[:400] + "...")
            else:
                core_error_parts.append(msg)
        
        # Format the discovery
        discovery_text = "\n".join(core_error_parts)
        response_parts.append(f"```\n{discovery_text}\n```")
    else:
        # If no primary failures, show first failure message
        if failures:
            first_failure = failures[0]
            failure_msg = first_failure.get('msg', '')
            if failure_msg and not isinstance(failure_msg, str):
                failure_msg = str(failure_msg)
            if failure_msg:
                response_parts.append(f"```\n{failure_msg[:400]}\n```")
            else:
                response_parts.append("```\nError detected in log analysis\n```")
        else:
            response_parts.append("```\nError detected in log analysis\n```")
    
    response_parts.append("")
    
    # ========== ROOT CAUSE IDENTIFIED ==========
    response_parts.append("## 🎯 Root Cause Identified\n")
    if primary_failures:
        pf = primary_failures[0]
        root_type = pf.get('root_cause_type', pf.get('type', 'Unknown')).replace('_', ' ').title()
        component = pf.get('component', 'Unknown Component')
        operation = pf.get('operation', '')
        msg = pf.get('message', '')
        if msg and not isinstance(msg, str):
            msg = str(msg)
        
        # Extract complete endpoint/URL with comprehensive pattern matching
        endpoint = None
        full_url = None
        
        # Try to find full URL first - multiple patterns
        url_patterns = [
            r'URL:\s*(?:GET|POST|PUT|DELETE)\s+[\'"]?(http[s]?://[^\s\'"]+)',  # URL: GET 'http://...'
            r'(http[s]?://[^\s\)\"]+)',  # Full URL
            r'[\'"]?(http[s]?://[^\s\'"]+)',  # Quoted URL
            r'http[s]?://[^\s]+',  # Simple URL pattern
        ]
        
        for pattern in url_patterns:
            url_match = re.search(pattern, msg)
            if url_match:
                full_url = url_match.group(1) if len(url_match.groups()) > 0 and url_match.group(1) else url_match.group(0)
                full_url = full_url.strip('"\'')
                if full_url and 'http' in full_url and len(full_url) > 10:
                    break
        
        # If no full URL, try to find endpoint path - look for complete paths
        if not full_url:
            # Look for paths in error messages (like "/mtna-ability/EaiEnvelopeSoapQSService")
            path_patterns = [
                r'HTTP Status \d+[^\w]*([/][^\s\)\"]+)',  # HTTP Status 404 - /path
                r'[/]([a-zA-Z0-9_-]+[/][^\s\)\"]+)',  # /app/service format
                r'[/]([a-zA-Z0-9_-]+[^\s\)\"]*)',  # /service format
                r'[\'"]?([/][^\s\'"]+)',  # Quoted path starting with /
            ]
            
            for pattern in path_patterns:
                path_match = re.search(pattern, msg)
                if path_match:
                    potential_endpoint = path_match.group(1) if len(path_match.groups()) > 0 else path_match.group(0)
                    potential_endpoint = potential_endpoint.strip('"\'')
                    # Filter out incomplete paths (like "/7." or single characters)
                    if potential_endpoint and len(potential_endpoint) > 3:
                        # Check if it looks like a valid path (has alphanumeric chars, not just numbers/dots)
                        if re.search(r'[a-zA-Z]', potential_endpoint) or len(potential_endpoint.split('/')) > 2:
                            endpoint = '/' + potential_endpoint if not potential_endpoint.startswith('/') else potential_endpoint
                            break
        
        # Extract IP and port
        ip_port = None
        ip_match = re.search(r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}):(\d+)', msg)
        if ip_match:
            ip_port = f"{ip_match.group(1)}:{ip_match.group(2)}"
        
        # Extract server info
        server_info = None
        if 'Apache Tomcat' in msg:
            tomcat_match = re.search(r'Apache Tomcat/([^\s\)]+)', msg)
            if tomcat_match:
                server_info = f"Apache Tomcat/{tomcat_match.group(1)}"
        
        # Extract HTTP status
        http_status = None
        status_match = re.search(r'\b(40[0-9]|50[0-9])\b', msg)
        if status_match:
            http_status = status_match.group(1)
        
        # Build comprehensive root cause statement
        root_cause_parts = []
        
        if full_url:
            root_cause_parts.append(f"The endpoint `{full_url}` does not exist on the Tomcat server.")
        elif endpoint and ip_port:
            root_cause_parts.append(f"The endpoint `http://{ip_port}{endpoint}` does not exist on the Tomcat server.")
        elif endpoint:
            root_cause_parts.append(f"The endpoint `{endpoint}` does not exist or is not available.")
        elif root_type.lower() == 'external api failure':
            if operation:
                root_cause_parts.append(f"External API call failed in component **{component}** during operation **{operation}**.")
            else:
                root_cause_parts.append(f"External API call failed in component **{component}**.")
            root_cause_parts.append("The API endpoint is not responding or does not exist.")
            if http_status:
                root_cause_parts.append(f"HTTP Status Code: {http_status}")
        elif root_type.lower() == 'authentication failure':
            root_cause_parts.append(f"Authentication failed in component **{component}**.")
            root_cause_parts.append("Invalid credentials or insufficient permissions.")
            if http_status:
                root_cause_parts.append(f"HTTP Status Code: {http_status}")
        elif root_type.lower() == 'connection failure':
            root_cause_parts.append(f"Connection to external service failed in component **{component}**.")
            root_cause_parts.append("Service is unreachable or not running.")
        elif root_type.lower() == 'timeout':
            root_cause_parts.append(f"Request timeout occurred in component **{component}**.")
            root_cause_parts.append("Service took too long to respond.")
        else:
            root_cause_parts.append(f"A {root_type.lower()} occurred in component **{component}**.")
            if operation:
                root_cause_parts.append(f"Operation: **{operation}**")
        
        # Add server info
        if server_info:
            root_cause_parts.append(f"\n**Server:** {server_info}")
        
        # Add transaction ID if available
        txn_id = pf.get('transaction_id', '')
        if txn_id and not isinstance(txn_id, str):
            txn_id = str(txn_id)
        if txn_id and not txn_id.startswith('Filename: '):
            root_cause_parts.append(f"**Transaction ID:** `{txn_id}`")
        
        # Add timestamp if available
        timestamp = pf.get('timestamp') or pf.get('ts', '')
        if timestamp:
            root_cause_parts.append(f"**Timestamp:** {timestamp}")
        
        response_parts.append("\n".join(root_cause_parts))
    else:
        if failures:
            first_failure = failures[0]
            component = first_failure.get('component', 'Unknown')
            operation = first_failure.get('operation', '')
            response_parts.append(f"Root cause analysis identified a failure in component **{component}**")
            if operation:
                response_parts.append(f"during operation **{operation}**.")
            else:
                response_parts.append(".")
            response_parts.append("Check detailed failures below for more information.")
        else:
            response_parts.append("Root cause analysis is in progress. Check detailed failures below.")
    
    response_parts.append("")
    
    # ========== POSSIBLE ISSUES ==========
    response_parts.append("## ⚠️ Possible Issues\n")
    if primary_failures:
        pf = primary_failures[0]
        root_type = pf.get('root_cause_type', '').lower()
        msg = pf.get('message', '')
        if msg and not isinstance(msg, str):
            msg = str(msg)
        
        # Extract complete endpoint/URL
        endpoint = None
        full_url = None
        
        # Try full URL first
        url_patterns = [
            r'URL:\s*(?:GET|POST|PUT|DELETE)\s+[\'"]?(http[s]?://[^\s\'"]+)',
            r'(http[s]?://[^\s\)\"]+)',
        ]
        for pattern in url_patterns:
            url_match = re.search(pattern, msg)
            if url_match:
                full_url = url_match.group(1) if len(url_match.groups()) > 0 and url_match.group(1) else url_match.group(0)
                full_url = full_url.strip('"\'')
                if full_url and 'http' in full_url:
                    break
        
        # If no full URL, try path
        if not full_url:
            path_patterns = [
                r'HTTP Status \d+[^\w]*[/]([^\s\)\"]+)',
                r'[/]([a-zA-Z0-9_-]+[/][a-zA-Z0-9_/-]+)',
                r'[/]([a-zA-Z0-9_-]+[a-zA-Z0-9_/-]*)',
            ]
            for pattern in path_patterns:
                path_match = re.search(pattern, msg)
                if path_match:
                    potential = '/' + path_match.group(1) if not path_match.group(1).startswith('/') else path_match.group(1)
                    potential = potential.strip('"\'')
                    if potential and len(potential) > 3 and re.search(r'[a-zA-Z]', potential):
                        endpoint = potential
                        break
        
        issues = []
        if root_type == 'external_api_failure' or '404' in msg or 'not available' in msg.lower() or 'not found' in msg.lower():
            if full_url:
                issues.append(f"**Wrong URL Path:** The web service might be deployed at a different path than `{full_url}`")
                issues.append(f"**Application Not Deployed:** The application might not be deployed on the server")
                issues.append(f"**Wrong Context Path:** The web service context path might be different")
                issues.append(f"**Service Not Started:** The specific SOAP service might not be running")
            elif endpoint:
                issues.append(f"**Wrong URL Path:** The web service might be deployed at a different path than `{endpoint}`")
                issues.append(f"**Application Not Deployed:** The application might not be deployed on the server")
                issues.append(f"**Wrong Context Path:** The web service context path might be different")
                issues.append(f"**Service Not Started:** The specific SOAP service might not be running")
            else:
                issues.append("**Wrong URL Path:** The web service might be deployed at a different path")
                issues.append("**Application Not Deployed:** The application might not be deployed")
                issues.append("**Service Not Started:** The service might not be running")
                issues.append("**Network Issues:** Connectivity problems to the service")
        elif root_type == 'authentication_failure':
            issues.append("**Invalid Credentials:** Authentication credentials might be incorrect or expired")
            issues.append("**Token Expired:** Authentication tokens might have expired")
            issues.append("**Permission Issues:** User might not have required permissions")
            issues.append("**Auth Service Down:** Authentication service might be unavailable")
        elif root_type == 'connection_failure':
            issues.append("**Service Not Running:** The target service might not be started")
            issues.append("**Network Connectivity:** Network issues preventing connection")
            issues.append("**Firewall Blocking:** Firewall rules might be blocking the connection")
            issues.append("**DNS Resolution:** DNS might not be resolving the service hostname")
        elif root_type == 'timeout':
            issues.append("**Service Overloaded:** The service might be overloaded and slow to respond")
            issues.append("**Network Latency:** High network latency causing timeouts")
            issues.append("**Timeout Too Short:** Configured timeout might be too short")
            issues.append("**Database Slow:** Database queries might be taking too long")
        else:
            issues.append("**Service Configuration:** Service might be misconfigured")
            issues.append("**Resource Constraints:** Service might be out of resources")
            issues.append("**Dependency Issues:** Required dependencies might be missing")
        
        for i, issue in enumerate(issues, 1):
            response_parts.append(f"{i}. {issue}")
    else:
        response_parts.append("1. Review the detailed failures below for specific issues")
    
    response_parts.append("")
    
    # ========== WHAT'S WORKING NOW ==========
    response_parts.append("## ✅ What's Working Now\n")
    
    # Identify working components
    working_items = []
    
    # Check successful operations
    success_count = stats.get('success_count', 0)
    if success_count > 0:
        working_items.append(f"**{success_count} successful operations** completed successfully")
    
    # Check for successful components
    successful_components = set()
    for entry in all_entries:
        if isinstance(entry, dict):
            relationship = entry.get('relationship', '').lower()
            component = entry.get('component', '')
            if relationship == 'success' and component:
                successful_components.add(component)
    
    if successful_components:
        working_items.append(f"**Components working:** {', '.join(list(successful_components)[:5])}")
    
    # Check network connectivity (if we have successful external calls)
    successful_external_calls = 0
    for entry in all_entries:
        if isinstance(entry, dict):
            component = entry.get('component', '').upper()
            relationship = entry.get('relationship', '').lower()
            if 'HTTP' in component or 'IHTTP' in component:
                if relationship == 'success':
                    successful_external_calls += 1
    
    # Extract server info from failures to show what's accessible
    if primary_failures:
        pf = primary_failures[0]
        msg = pf.get('message', '')
        if msg and not isinstance(msg, str):
            msg = str(msg)
        
        # Extract IP and port
        ip_match = re.search(r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}):(\d+)', msg)
        if ip_match:
            ip = ip_match.group(1)
            port = ip_match.group(2)
            working_items.append(f"**Network connectivity** to `{ip}:{port}` is restored")
        
        # Check for server info
        if 'Apache Tomcat' in msg:
            tomcat_match = re.search(r'Apache Tomcat/([^\s]+)', msg)
            if tomcat_match:
                version = tomcat_match.group(1)
                working_items.append(f"**Tomcat server** is running (Apache Tomcat/{version})")
                if ip_match:
                    working_items.append(f"**Port {port}** is accessible")
    
    # Check database operations
    db_success = 0
    for entry in all_entries:
        if isinstance(entry, dict):
            component = entry.get('component', '').upper()
            relationship = entry.get('relationship', '').lower()
            if ('SQL' in component or 'PSQL' in component or 'DB' in component) and relationship == 'success':
                db_success += 1
    
    if db_success > 0:
        working_items.append(f"**Database operations** ({db_success} successful queries)")
    
    if not working_items:
        working_items.append("**Basic system operations** are functioning")
        working_items.append("**Log processing** is working correctly")
    
    for item in working_items:
        response_parts.append(f"- ✅ {item}")
    
    response_parts.append("")
    
    # ========== ADDITIONAL CONTEXT FROM CONVERSATION ==========
    if conversation_context:
        prev_failures = conversation_context.get('previous_failures', [])
        prev_endpoints = conversation_context.get('previous_endpoints', set())
        prev_components = conversation_context.get('previous_components', set())
        prev_errors = conversation_context.get('previous_errors', [])
        working_components = conversation_context.get('working_components', set())
        
        if prev_failures or prev_endpoints or prev_errors:
            response_parts.append("## 📚 Context from Previous Analysis\n")
            response_parts.append("Based on previous logs analyzed in this conversation:\n")
            
            context_items = []
            
            # Compare current endpoint with previous
            if primary_failures:
                pf = primary_failures[0]
                msg = pf.get('message', '')
                if msg:
                    # Extract current endpoint
                    current_endpoint = None
                    url_patterns = [
                        r'(http[s]?://[^\s\)]+)',
                        r'/([^\s\)\"]+[^\s\)\"]*)',
                    ]
                    for pattern in url_patterns:
                        url_match = re.search(pattern, msg)
                        if url_match:
                            current_endpoint = url_match.group(1) if url_match.group(1) else url_match.group(0)
                            current_endpoint = current_endpoint.strip('"\'')
                            if current_endpoint and len(current_endpoint) > 2:
                                break
                    
                    if current_endpoint and prev_endpoints:
                        if current_endpoint in prev_endpoints:
                            context_items.append(f"⚠️ **Recurring Issue:** The endpoint `{current_endpoint}` was also failing in previous logs. This suggests a persistent problem.")
                        else:
                            context_items.append(f"ℹ️ **Different Endpoint:** Previous failures involved different endpoints: {', '.join(list(prev_endpoints)[:3])}")
                    
                    # Compare components
                    current_component = pf.get('component', '')
                    if current_component and prev_components:
                        if current_component in prev_components:
                            context_items.append(f"⚠️ **Same Component:** Component **{current_component}** has been failing in previous logs.")
                        else:
                            context_items.append(f"ℹ️ **Different Component:** Previous failures were in: {', '.join(list(prev_components)[:3])}")
            
            # Show previous error patterns
            if prev_errors:
                error_types = {}
                for err in prev_errors[:5]:
                    err_msg = err.get('error', '').lower()
                    if '404' in err_msg or 'not found' in err_msg:
                        error_types['404'] = error_types.get('404', 0) + 1
                    elif '500' in err_msg or 'internal' in err_msg:
                        error_types['500'] = error_types.get('500', 0) + 1
                    elif 'timeout' in err_msg:
                        error_types['timeout'] = error_types.get('timeout', 0) + 1
                    elif 'auth' in err_msg or '401' in err_msg:
                        error_types['auth'] = error_types.get('auth', 0) + 1
                
                if error_types:
                    error_summary = []
                    for err_type, count in error_types.items():
                        error_summary.append(f"{err_type.upper()} errors ({count} occurrences)")
                    context_items.append(f"📊 **Error Patterns:** Previous logs showed: {', '.join(error_summary)}")
            
            # Show what was working before
            if working_components:
                context_items.append(f"✅ **Previously Working:** These components were functioning: {', '.join(list(working_components)[:5])}")
            
            # Show failure count trend
            if prev_failures:
                context_items.append(f"📈 **Failure History:** {len(prev_failures)} previous root cause(s) identified in this conversation")
            
            if context_items:
                for item in context_items:
                    response_parts.append(f"- {item}")
            else:
                response_parts.append("- No significant patterns detected from previous analysis")
            
            response_parts.append("")
    
    # ========== DETAILED STATISTICS ==========
    response_parts.append("## 📊 Detailed Statistics\n")
    response_parts.append(f"- **Total Log Entries**: {stats.get('parsed_entries', 0):,}")
    response_parts.append(f"- **Failures Detected**: {stats.get('rule_failures', 0):,}")
    response_parts.append(f"- **Critical Errors**: {stats.get('critical_errors', 0):,}")
    response_parts.append(f"- **HTTP Errors**: {stats.get('http_errors', 0):,}")
    if stats.get('timeout_issues', 0) > 0:
        response_parts.append(f"- **Timeout Issues**: {stats.get('timeout_issues', 0):,}")
    response_parts.append("")
    
    # ========== RECOMMENDATIONS ==========
    if recommendations:
        response_parts.append("## 💡 Recommended Actions\n")
        for rec in recommendations[:5]:
            priority = rec.get('priority', 'info').upper()
            message = rec.get('message', '')
            response_parts.append(f"- **[{priority}]** {message}")
        response_parts.append("")
    
    return "\n".join(response_parts)

@app.route('/')
def index():
    """Serve the main chat interface."""
    # Pre-load analyzer in background to speed up first request
    if analyzer is None:
        def load_analyzer():
            get_analyzer()
        threading.Thread(target=load_analyzer, daemon=True).start()
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_logs():
    """API endpoint for log analysis with intelligent conversation handling."""
    try:
        data = request.json or {}
        log_content = data.get('log_content', '')
        log_file_path = data.get('log_file_path', None)
        user_query = log_content  # Treat log content as query
        
        if not log_content and not log_file_path:
            return jsonify({
                'success': False,
                'error': 'No log content or file provided'
            }), 400
        
        # Get or create conversation ID
        conversation_id = session.get('conversation_id')
        if not conversation_id:
            from datetime import datetime
            conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            session['conversation_id'] = conversation_id
        
        # Get conversation context
        conversation_context = get_conversation_context(conversation_id)
        
        # Classify query to understand intent
        query_type = intelligent_chat.classify_query(user_query, conversation_context)
        logger.info(f"Query classified as: {query_type['type']}, intent: {query_type['intent']}, needs_analysis: {query_type['needs_analysis']}")
        
        # If it's a question or follow-up without log content, handle intelligently
        # Also handle short queries that might be questions even if not explicitly detected
        is_intelligent_query = (
            query_type['type'] in ['question', 'follow_up', 'comparison'] and not query_type['needs_analysis']
        ) or (
            # Handle short queries with context as potential questions
            len(user_query.strip()) < 100 and 
            conversation_context and 
            conversation_context.get('previous_failures') and
            not query_type['needs_analysis']
        )
        
        if is_intelligent_query:
            # Get previous analysis results if available
            previous_results = conversation_context.get('last_analysis')
            if not previous_results:
                # Try to get last analysis from chat history
                try:
                    interactions = chat_collector.get_interactions(limit=1, conversation_id=conversation_id)
                    if interactions:
                        previous_results = interactions[0].analysis_results
                except:
                    pass
            
            # Generate intelligent response
            intelligent_response = intelligent_chat.generate_intelligent_response(
                user_query,
                query_type,
                previous_results,
                conversation_context
            )
            
            if intelligent_response:
                # Don't add context summary if it's already included in the response
                # The intelligent handler now includes context naturally
                
                return jsonify({
                    'success': True,
                    'response': intelligent_response,
                    'data': {},
                    'conversation_id': conversation_id,
                    'query_type': query_type['type']
                })
        
        # Otherwise, proceed with log analysis
        # Create temporary file if content provided
        log_file = None
        if log_content:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
                f.write(log_content)
                log_file = Path(f.name)
        elif log_file_path:
            log_file = Path(log_file_path)
        
        if not log_file or not log_file.exists():
            return jsonify({
                'success': False,
                'error': 'Log file not found'
            }), 400
        
        # Get analyzer (lazy load if needed)
        current_analyzer = get_analyzer()
        
        # Run analysis
        logger.info(f"Analyzing log file: {log_file}")
        results = current_analyzer.analyze(log_file)
        
        # Update conversation context
        update_conversation_context(conversation_id, results)
        
        # Get updated context for response
        conversation_context = get_conversation_context(conversation_id)
        
        # Format response with context
        formatted_response = format_analysis_response(results, conversation_context)
        
        # Enhance response with LLM if available
        if llm_enhancer and llm_enhancer.enabled:
            try:
                # Generate intelligent summary
                summary = llm_enhancer.generate_intelligent_summary(results, conversation_context)
                if summary:
                    formatted_response = f"## 🤖 AI Summary\n\n{summary}\n\n---\n\n{formatted_response}"
            except Exception as e:
                logger.warning(f"LLM summary generation failed: {e}")
        
        # Save interaction for training
        try:
            chat_collector.save_interaction(
                user_query=log_content[:500] if log_content else "File upload",
                log_content=log_content if log_content else "",
                log_filename=None,
                analysis_results=results,
                response_text=formatted_response,
                conversation_id=conversation_id
            )
        except Exception as e:
            logger.warning(f"Failed to save chat interaction: {e}")
        
        # Clean up temporary file
        if log_content and log_file.exists():
            try:
                log_file.unlink()
            except:
                pass
        
        return jsonify({
            'success': True,
            'response': formatted_response,
            'data': results,
            'conversation_id': conversation_id,
            'query_type': 'log_analysis'
        })
        
    except Exception as e:
        logger.error(f"Analysis error: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload."""
    try:
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file provided'
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = Path(app.config['UPLOAD_FOLDER']) / filename
            file.save(str(filepath))
            
            # Read file content
            with filepath.open('r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            
            # Get or create conversation ID
            conversation_id = session.get('conversation_id')
            if not conversation_id:
                from datetime import datetime
                conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                session['conversation_id'] = conversation_id
            
            # Get conversation context
            conversation_context = get_conversation_context(conversation_id)
            
            # Get analyzer (lazy load if needed)
            current_analyzer = get_analyzer()
            
            # Run analysis
            logger.info(f"Analyzing uploaded file: {filename}")
            results = current_analyzer.analyze(filepath)
            
            # Update conversation context
            update_conversation_context(conversation_id, results)
            
            # Get updated context for response
            conversation_context = get_conversation_context(conversation_id)
            
            # Format response with context
            formatted_response = format_analysis_response(results, conversation_context)
            
            # Enhance response with LLM if available
            if llm_enhancer and llm_enhancer.enabled:
                try:
                    # Generate intelligent summary
                    summary = llm_enhancer.generate_intelligent_summary(results, conversation_context)
                    if summary:
                        formatted_response = f"## 🤖 AI Summary\n\n{summary}\n\n---\n\n{formatted_response}"
                except Exception as e:
                    logger.warning(f"LLM summary generation failed: {e}")
            
            # Save interaction for training
            try:
                chat_collector.save_interaction(
                    user_query=f"Uploaded file: {filename}",
                    log_content=content,
                    log_filename=filename,
                    analysis_results=results,
                    response_text=formatted_response,
                    conversation_id=conversation_id
                )
            except Exception as e:
                logger.warning(f"Failed to save chat interaction: {e}")
            
            # Clean up uploaded file
            try:
                filepath.unlink()
            except:
                pass
            
            return jsonify({
                'success': True,
                'filename': filename,
                'response': formatted_response,
                'data': results,
                'conversation_id': conversation_id
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Invalid file type. Allowed: .txt, .log, .json'
            }), 400
            
    except Exception as e:
        logger.error(f"Upload error: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint - fast response without loading models."""
    return jsonify({
        'status': 'healthy',
        'analyzer_loaded': analyzer is not None,
        'message': 'Server is running'
    })

@app.route('/api/chat/stats', methods=['GET'])
def chat_stats():
    """Get statistics about collected chat interactions."""
    try:
        stats = chat_collector.get_stats()
        return jsonify({
            'success': True,
            'stats': stats
        })
    except Exception as e:
        logger.error(f"Failed to get chat stats: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/training/export', methods=['POST'])
def export_training_data():
    """Export chat interactions as training data."""
    try:
        output_file = project_root / "data" / "chat_training_data.json"
        count = chat_collector.export_for_training(output_file)
        
        return jsonify({
            'success': True,
            'exported_examples': count,
            'output_file': str(output_file),
            'message': f'Exported {count} training examples'
        })
    except Exception as e:
        logger.error(f"Failed to export training data: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/chat/conversations', methods=['GET'])
def get_conversations():
    """Get list of all conversations."""
    try:
        conversations = chat_collector.get_conversations()
        return jsonify({
            'success': True,
            'conversations': conversations
        })
    except Exception as e:
        logger.error(f"Failed to get conversations: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/chat/conversation/<conversation_id>', methods=['GET'])
def get_conversation(conversation_id):
    """Get messages for a specific conversation."""
    try:
        interactions = chat_collector.get_interactions(conversation_id=conversation_id)
        
        # Format for frontend
        messages = []
        for interaction in interactions:
            # Extract user message (query or filename)
            user_msg = interaction.user_query
            if interaction.log_filename:
                user_msg = f"📎 Uploaded file: {interaction.log_filename}"
            elif len(interaction.log_content) > 100:
                user_msg = interaction.log_content[:100] + "..."
            else:
                user_msg = interaction.log_content or interaction.user_query
            
            messages.append({
                'timestamp': interaction.timestamp,
                'user_message': user_msg,
                'assistant_response': interaction.response_text,
                'log_filename': interaction.log_filename
            })
        
        return jsonify({
            'success': True,
            'conversation_id': conversation_id,
            'messages': messages
        })
    except Exception as e:
        logger.error(f"Failed to get conversation: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/chat/new', methods=['POST'])
def new_conversation():
    """Start a new conversation."""
    try:
        from datetime import datetime
        conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        session['conversation_id'] = conversation_id
        
        # Clear conversation context
        with context_lock:
            if conversation_id in conversation_contexts:
                del conversation_contexts[conversation_id]
        
        return jsonify({
            'success': True,
            'conversation_id': conversation_id
        })
    except Exception as e:
        logger.error(f"Failed to create conversation: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/chat/history', methods=['GET'])
def get_chat_history():
    """Get chat history for current conversation."""
    try:
        conversation_id = request.args.get('conversation_id') or session.get('conversation_id')
        limit = request.args.get('limit', 50, type=int)
        interactions = chat_collector.get_interactions(limit=limit, conversation_id=conversation_id)
        
        # Format for frontend
        history = []
        for interaction in interactions:
            # Extract user message (query or filename)
            user_msg = interaction.user_query
            if interaction.log_filename:
                user_msg = f"📎 Uploaded file: {interaction.log_filename}"
            elif len(interaction.log_content) > 100:
                user_msg = interaction.log_content[:100] + "..."
            else:
                user_msg = interaction.log_content or interaction.user_query
            
            history.append({
                'timestamp': interaction.timestamp,
                'user_message': user_msg,
                'assistant_response': interaction.response_text,
                'log_filename': interaction.log_filename
            })
        
        return jsonify({
            'success': True,
            'history': history,
            'total': len(history),
            'conversation_id': conversation_id
        })
    except Exception as e:
        logger.error(f"Failed to get chat history: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def find_available_port(start_port=5000, max_attempts=10):
    """Find an available port starting from start_port."""
    for i in range(max_attempts):
        port = start_port + i
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(('localhost', port))
            sock.close()
            return port
        except OSError:
            continue
    return None

if __name__ == '__main__':
    # Get port from environment or use default
    requested_port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    # Find available port
    port = find_available_port(requested_port)
    
    if port is None:
        logger.error(f"Could not find an available port starting from {requested_port}")
        sys.exit(1)
    
    if port != requested_port:
        logger.info(f"Port {requested_port} is not available. Using port {port} instead.")
    
    print(f"\n{'='*60}")
    print(f"🚀 Log Analyzer Web Application")
    print(f"{'='*60}")
    print(f"🌐 Server running at: http://localhost:{port}")
    print(f"📝 Open your browser and navigate to the URL above")
    print(f"⚡ Models will load on first request (faster startup)")
    print(f"💬 Conversational AI with context memory enabled")
    print(f"{'='*60}\n")
    
    # Start loading models in background for faster first request
    def preload_models():
        try:
            get_analyzer()
        except Exception as e:
            logger.warning(f"Background model loading failed: {e}")
    
    threading.Thread(target=preload_models, daemon=True).start()
    
    app.run(host='0.0.0.0', port=port, debug=debug)
