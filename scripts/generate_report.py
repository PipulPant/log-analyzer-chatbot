#!/usr/bin/env python3
"""
Generate HTML report from analysis JSON
"""
import json
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def generate_html_report(json_file: Path, output_file: Path):
    """Generate HTML report from analysis JSON."""
    
    # Load JSON data
    with json_file.open('r', encoding='utf-8') as f:
        data = json.load(f)
    
    stats = data.get('stats', {})
    failures = data.get('failures', [])
    ml_info = data.get('ml_info', {})
    format_type = data.get('format', 'unknown')
    flow_analysis = data.get('flow_analysis', {})
    issues = data.get('issues', {'critical': [], 'warning': [], 'info': []})
    recommendations = data.get('recommendations', [])
    root_cause_analysis = data.get('root_cause_analysis', {})
    primary_failures = root_cause_analysis.get('primary_failures', [])
    
    # Generate HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Log Analysis Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        .header p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        .content {{
            padding: 30px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
            transition: transform 0.2s;
        }}
        .stat-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}
        .stat-card h3 {{
            color: #667eea;
            font-size: 0.9em;
            text-transform: uppercase;
            margin-bottom: 10px;
        }}
        .stat-card .value {{
            font-size: 2em;
            font-weight: bold;
            color: #333;
        }}
        .stat-card .label {{
            color: #666;
            font-size: 0.9em;
            margin-top: 5px;
        }}
        .failures-section {{
            margin-top: 30px;
        }}
        .failures-section h2 {{
            color: #333;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
        }}
        .failure-card {{
            background: #fff;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            transition: box-shadow 0.2s;
        }}
        .failure-card:hover {{
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}
        .failure-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            flex-wrap: wrap;
        }}
        .failure-title {{
            font-size: 1.2em;
            font-weight: bold;
            color: #333;
        }}
        .failure-level {{
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: bold;
        }}
        .level-error {{
            background: #fee;
            color: #c33;
        }}
        .level-warning {{
            background: #ffeaa7;
            color: #d63031;
        }}
        .level-info {{
            background: #e3f2fd;
            color: #1976d2;
        }}
        .failure-details {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 15px;
        }}
        .detail-item {{
            background: #f8f9fa;
            padding: 10px;
            border-radius: 5px;
        }}
        .detail-item label {{
            font-size: 0.85em;
            color: #666;
            display: block;
            margin-bottom: 5px;
        }}
        .detail-item value {{
            font-weight: bold;
            color: #333;
        }}
        .keywords {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 10px;
        }}
        .keyword {{
            background: #667eea;
            color: white;
            padding: 5px 12px;
            border-radius: 15px;
            font-size: 0.85em;
        }}
        .message {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin-top: 10px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            word-break: break-word;
            max-height: 200px;
            overflow-y: auto;
        }}
        .severity-bar {{
            width: 100%;
            height: 8px;
            background: #e0e0e0;
            border-radius: 4px;
            margin-top: 10px;
            overflow: hidden;
        }}
        .severity-fill {{
            height: 100%;
            background: linear-gradient(90deg, #4caf50, #ff9800, #f44336);
            transition: width 0.3s;
        }}
        .model-info {{
            background: #e8f5e9;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
            border-left: 4px solid #4caf50;
        }}
        .model-info h3 {{
            color: #2e7d32;
            margin-bottom: 10px;
        }}
        .no-failures {{
            text-align: center;
            padding: 40px;
            color: #4caf50;
            font-size: 1.2em;
        }}
        .timestamp {{
            color: #666;
            font-size: 0.9em;
            margin-top: 20px;
            text-align: center;
        }}
        @media print {{
            body {{
                background: white;
            }}
            .container {{
                box-shadow: none;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Log Analysis Report</h1>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        <div class="content">
"""
    
    # Add model info if available
    if ml_info and ml_info.get('accuracy'):
        html += f"""
            <div class="model-info">
                <h3>🤖 Trained Model Information</h3>
                <div class="stats-grid">
                    <div class="stat-card">
                        <h3>Model Type</h3>
                        <div class="value">{ml_info.get('model_type', 'N/A')}</div>
                    </div>
                    <div class="stat-card">
                        <h3>Accuracy</h3>
                        <div class="value">{ml_info.get('accuracy', 0)*100:.1f}%</div>
                    </div>
                    <div class="stat-card">
                        <h3>F1 Score</h3>
                        <div class="value">{ml_info.get('f1_score', 0):.3f}</div>
                    </div>
                    <div class="stat-card">
                        <h3>High Confidence</h3>
                        <div class="value">{ml_info.get('high_confidence', 0):,}</div>
                        <div class="label">Predictions</div>
                    </div>
                </div>
            </div>
"""
    
    # Statistics
    html += """
            <div class="stats-grid">
"""
    
    stats_items = [
        ('Total Lines', stats.get('total_lines', 0), 'lines', True),
        ('Parsed Entries', stats.get('parsed_entries', 0), 'entries', True),
        ('Rule Failures', stats.get('rule_failures', 0), 'failures', True),
        ('ML Failures', stats.get('ml_failures', 0), 'failures', True),
        ('Critical Errors', stats.get('critical_errors', 0), 'errors', True),
        ('Warnings', stats.get('warnings', 0), 'warnings', True),
        ('Connection Issues', stats.get('connection_issues', 0), 'issues', True),
        ('HTTP Errors', stats.get('http_errors', 0), 'errors', True),
        ('Timeout Issues', stats.get('timeout_issues', 0), 'issues', True),
        ('Data Issues', stats.get('data_issues', 0), 'issues', True),
        ('Flow Breaks', stats.get('flow_breaks', 0), 'breaks', True),
        ('Success Rate', f"{(stats.get('success_count', 0) / max(stats.get('parsed_entries', 1), 1) * 100):.1f}", '%', False),
        ('Avg Process Time', f"{stats.get('avg_process_time_ms', 0):.1f}", 'ms', False),
        ('Avg Queue Time', f"{stats.get('avg_queue_time_ms', 0):.1f}", 'ms', False),
        ('Processing Time', stats.get('processing_time', 0), 'seconds', False),
        ('File Size', stats.get('file_size_mb', 0), 'MB', False),
    ]
    
    for label, value, unit, is_int in stats_items:
        if is_int:
            value_str = f"{int(value):,}" if value else "0"
        else:
            try:
                if isinstance(value, str):
                    value_str = value
                else:
                    value_str = f"{float(value):.2f}"
            except (ValueError, TypeError):
                value_str = str(value) if value else "0"
        html += f"""
                <div class="stat-card">
                    <h3>{label}</h3>
                    <div class="value">{value_str}</div>
                    <div class="label">{unit}</div>
                </div>
"""
    
    html += """
            </div>
"""
    
    # Format info
    html += f"""
            <div style="background: #f0f0f0; padding: 15px; border-radius: 5px; margin-bottom: 30px;">
                <strong>Detected Format:</strong> {format_type}
            </div>
"""
    
    # Flow Analysis section - Minimized by default
    if flow_analysis:
        html += """
            <div class="failures-section">
                <details style="cursor: pointer;">
                    <summary style="font-size: 24px; font-weight: bold; padding: 10px; cursor: pointer;">
                        📊 Flow Analysis
                    </summary>
                    <div style="margin-top: 15px;">
"""
        html += f"""
                        <div class="stats-grid">
                            <div class="stat-card">
                                <h3>Total Transactions</h3>
                                <div class="value">{flow_analysis.get('total_transactions', 0):,}</div>
                            </div>
                            <div class="stat-card">
                                <h3>Transactions with Issues</h3>
                                <div class="value">{flow_analysis.get('transactions_with_issues', 0):,}</div>
                            </div>
                            <div class="stat-card">
                                <h3>Connection Issues</h3>
                                <div class="value">{flow_analysis.get('connection_issues', 0):,}</div>
                            </div>
                            <div class="stat-card">
                                <h3>HTTP Errors</h3>
                                <div class="value">{flow_analysis.get('http_errors', 0):,}</div>
                            </div>
                            <div class="stat-card">
                                <h3>Timeouts</h3>
                                <div class="value">{flow_analysis.get('timeouts', 0):,}</div>
                            </div>
                            <div class="stat-card">
                                <h3>Data Issues</h3>
                                <div class="value">{flow_analysis.get('data_issues', 0):,}</div>
                            </div>
                            <div class="stat-card">
                                <h3>Flow Breaks</h3>
                                <div class="value">{flow_analysis.get('flow_breaks', 0):,}</div>
                            </div>
                        </div>
                    </div>
                </details>
            </div>
"""
    
    # Primary Failures (Root Causes) section - Show at TOP (right after Flow Analysis)
    if primary_failures:
        html += """
            <div class="failures-section" style="border: 3px solid #f44336; background: #fff5f5;">
                <h2 style="color: #f44336; font-size: 28px;">🎯 PRIMARY FAILURES (ROOT CAUSES)</h2>
                <p style="font-size: 16px; color: #666; margin-bottom: 20px;">
                    These are the main failures that caused other errors. Fix these first to resolve cascading issues.
                </p>
"""
        for i, pf in enumerate(primary_failures, 1):
            failure_idx = pf.get('index', 0)
            if failure_idx < len(failures):
                failure = failures[failure_idx]
                root_type = pf.get('type', 'unknown')
                impact = pf.get('impact_count', 1)
                cascade = pf.get('cascade_count', 0)
                
                html += f"""
                <div class="failure-card" style="border-left: 6px solid #f44336; background: #fff5f5; margin-bottom: 20px;">
                    <div class="failure-header">
                        <div class="failure-title" style="font-size: 20px; font-weight: bold;">
                            🔴 PRIMARY FAILURE #{i}: {root_type.replace('_', ' ').title()}
                        </div>
                        <span class="failure-level level-error" style="background: #f44336; color: white; padding: 5px 15px; font-weight: bold;">ROOT CAUSE</span>
                    </div>
                    <div style="background: #ffe0e0; padding: 10px; margin: 10px 0; border-radius: 5px;">
                        <strong>Impact:</strong> {impact} total failures ({cascade} cascading failures caused by this root cause)
                    </div>
                    <div class="failure-details">
                        <div class="detail-item">
                            <label>Component</label>
                            <value>{failure.get('component', 'N/A')}</value>
                        </div>
                        <div class="detail-item">
                            <label>Operation</label>
                            <value>{failure.get('operation', 'N/A')}</value>
                        </div>
                        <div class="detail-item">
                            <label>Transaction ID</label>
                            <value>{pf.get('transaction_id', 'N/A')[:50]}</value>
                        </div>
                        <div class="detail-item">
                            <label>Severity</label>
                            <value>{pf.get('severity', 0):.2f}</value>
                        </div>
                    </div>
                    <div class="message" style="margin-top: 15px;">
                        <strong>Root Cause Message:</strong><br>
                        {pf.get('message', failure.get('msg', 'N/A'))[:500]}
                    </div>
                </div>
"""
        html += """
            </div>
"""
    
    # Failures section
    html += """
            <div class="failures-section">
                <h2>🚨 All Detected Failures</h2>
"""
    
    if failures:
        for i, failure in enumerate(failures, 1):
            level = failure.get('level', 'INFO')
            level_class = level.lower().replace('error', 'error').replace('warning', 'warning').replace('info', 'info')
            if 'error' not in level_class and 'warning' not in level_class:
                level_class = 'info'
            
            severity = failure.get('severity_score', 0)
            keywords = failure.get('keywords_found', [])
            msg = failure.get('msg', '')
            
            # Check if this is a root cause or cascade
            is_root = failure.get('is_root_cause', False)
            is_cascade = failure.get('is_cascade', False)
            root_type = failure.get('root_cause_type', '')
            caused_by = failure.get('caused_by', '')
            
            # Style root causes and cascades differently
            card_style = ""
            if is_root:
                card_style = "border-left: 6px solid #f44336; background: #fff5f5;"
            elif is_cascade:
                card_style = "border-left: 4px solid #ff9800; background: #fffbf0;"
            
            badge_html = ""
            if is_root:
                badge_html = '<span class="failure-level level-error" style="background: #f44336; color: white; margin-left: 10px; padding: 5px 10px; font-weight: bold;">ROOT CAUSE</span>'
            elif is_cascade:
                badge_html = f'<span class="failure-level level-warning" style="background: #ff9800; margin-left: 10px;">CASCADE (caused by: {caused_by})</span>'
            
            html += f"""
                <div class="failure-card" style="{card_style}">
                    <div class="failure-header">
                        <div class="failure-title">Failure #{i}{' - ROOT CAUSE' if is_root else ' - CASCADE' if is_cascade else ''}</div>
                        <span class="failure-level level-{level_class}">{level}</span>
                        {badge_html}
                    </div>
                    <div class="failure-details">
                        <div class="detail-item">
                            <label>Timestamp</label>
                            <value>{failure.get('ts', 'N/A')}</value>
                        </div>
                        <div class="detail-item">
                            <label>Component</label>
                            <value>{failure.get('component', 'N/A')}</value>
                        </div>
                        <div class="detail-item">
                            <label>Operation</label>
                            <value>{failure.get('operation', 'N/A')}</value>
                        </div>
                        <div class="detail-item">
                            <label>Transaction ID</label>
                            <value>{failure.get('txn_id', 'N/A')[:30]}{'...' if len(failure.get('txn_id', '')) > 30 else ''}</value>
                        </div>
                        <div class="detail-item">
                            <label>Severity Score</label>
                            <value>{severity:.2f}</value>
                            <div class="severity-bar">
                                <div class="severity-fill" style="width: {severity*100}%"></div>
                            </div>
                        </div>
"""
            if is_root:
                impact = failure.get('impact_count', 1)
                cascade = failure.get('cascade_count', 0)
                html += f"""
                        <div class="detail-item" style="background: #ffe0e0; padding: 10px; margin-top: 10px; border-radius: 5px;">
                            <label style="font-weight: bold; color: #d32f2f;">Impact:</label>
                            <value>{impact} total failures ({cascade} cascading)</value>
                        </div>
"""
            elif is_cascade:
                html += f"""
                        <div class="detail-item" style="background: #fff3e0; padding: 10px; margin-top: 10px; border-radius: 5px;">
                            <label style="font-weight: bold; color: #f57c00;">Caused By:</label>
                            <value>{caused_by.replace('_', ' ').title()}</value>
                        </div>
"""
            html += """
                    </div>
"""
            
            if keywords:
                html += """
                    <div style="margin-top: 10px;">
                        <strong>Keywords Found:</strong>
                        <div class="keywords">
"""
                for keyword in keywords:
                    html += f'<span class="keyword">{keyword}</span>'
                html += """
                        </div>
                    </div>
"""
            
            if msg:
                html += f"""
                    <div class="message">
                        <strong>Message:</strong><br>
                        {msg[:500]}{'...' if len(msg) > 500 else ''}
                    </div>
"""
            
            if failure.get('raw'):
                html += f"""
                    <details style="margin-top: 10px;">
                        <summary style="cursor: pointer; color: #667eea; font-weight: bold;">View Raw Log Line</summary>
                        <div class="message" style="margin-top: 10px;">
                            {failure.get('raw', '')}
                        </div>
                    </details>
"""
            
            html += """
                </div>
"""
    else:
        html += """
                <div class="no-failures">
                    ✅ No failures detected in this log file!
                </div>
"""
    
    html += """
            </div>
"""
    
    # Issues section - Show at BOTTOM (after all failures)
    if any(issues.values()):
        html += """
            <div class="failures-section">
                <h2>⚠️ Detailed Issues</h2>
"""
        for severity_level in ['critical', 'warning', 'info']:
            severity_issues = issues.get(severity_level, [])
            if severity_issues:
                severity_title = severity_level.upper()
                severity_color = '#f44336' if severity_level == 'critical' else '#ff9800' if severity_level == 'warning' else '#2196f3'
                html += f"""
                <h3 style="color: {severity_color}; margin-top: 20px;">{severity_title} Issues ({len(severity_issues)})</h3>
"""
                for issue in severity_issues[:10]:  # Show first 10 of each type
                    html += f"""
                <div class="failure-card">
                    <div class="failure-header">
                        <div class="failure-title">Transaction: {issue.get('transaction_id', 'N/A')[:30]}</div>
                    </div>
                    <div class="message">
                        {json.dumps(issue.get('issue') or issue.get('error') or issue.get('timeout') or issue.get('break'), indent=2)}
                    </div>
                </div>
"""
        html += """
            </div>
"""
    
    html += """
            <div class="timestamp">
                Report generated by Log Analyzer with Supervised Learning
            </div>
        </div>
    </div>
</body>
</html>
"""
    
    # Save HTML file
    with output_file.open('w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"HTML report generated: {output_file}")
    return output_file

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate HTML report from analysis JSON')
    parser.add_argument('--json', type=Path, default=Path('analysis_report.json'),
                       help='Input JSON file (default: analysis_report.json)')
    parser.add_argument('-o', '--output', type=Path, default=Path('analysis_report.html'),
                       help='Output HTML file (default: analysis_report.html)')
    
    args = parser.parse_args()
    
    if not args.json.exists():
        print(f"Error: JSON file '{args.json}' not found!")
        print("Please run analysis first:")
        print("  python python_logs_analyser.py --model trained_model.pkl --logfile server.log")
        return 1
    
    generate_html_report(args.json, args.output)
    print(f"\n[SUCCESS] HTML report saved to: {args.output}")
    print(f"   Open it in your browser to view the results!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

