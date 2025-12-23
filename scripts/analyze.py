#!/usr/bin/env python3
"""
Main analysis script - analyzes log files and generates reports
"""
import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from log_analyzer import LogAnalyzer
from scripts.generate_report import generate_html_report
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main analysis function."""
    import argparse
    
    project_root = Path(__file__).parent.parent
    
    parser = argparse.ArgumentParser(description='Analyze log files for failures')
    parser.add_argument('--logfile', type=Path, 
                       default=project_root / "data" / "server.log",
                       help='Log file to analyze')
    parser.add_argument('--model', type=Path,
                       default=project_root / "data" / "models" / "trained_model.pkl",
                       help='Path to trained model (optional)')
    parser.add_argument('-o', '--output', type=Path,
                       default=project_root / 'reports' / 'analysis_report.json',
                       help='Output JSON file (default: reports/analysis_report.json)')
    parser.add_argument('--max-failures', type=int,
                       help='Maximum number of failures to report')
    parser.add_argument('--no-html', action='store_true',
                       help='Skip HTML report generation')
    parser.add_argument('--ml-config', type=Path,
                       help='Path to ML models configuration JSON file')
    
    args = parser.parse_args()
    
    # Resolve paths relative to project root
    if not args.logfile.is_absolute():
        args.logfile = project_root / args.logfile
    if args.model and not args.model.is_absolute():
        args.model = project_root / args.model
    
    if not args.logfile.exists():
        logger.error(f"Log file not found: {args.logfile}")
        return 1
    
    # Load ML models configuration if provided
    ml_models_config = None
    if args.ml_config:
        ml_config_path = args.ml_config if args.ml_config.is_absolute() else project_root / args.ml_config
        if ml_config_path.exists():
            import json
            with ml_config_path.open() as f:
                ml_models_config = json.load(f)
            logger.info(f"Loaded ML models configuration from {ml_config_path}")
    else:
        # Try to load default ensemble config
        default_config = project_root / "data" / "models" / "ensemble_config.json"
        if default_config.exists():
            import json
            with default_config.open() as f:
                ml_models_config = json.load(f)
            logger.info(f"Using default ML models configuration from {default_config}")
    
    # Initialize analyzer
    model_path = args.model if args.model and args.model.exists() else None
    analyzer = LogAnalyzer(model_path=model_path, ml_models_config=ml_models_config)
    
    # Run analysis
    logger.info(f"Analyzing {args.logfile}...")
    results = analyzer.analyze(args.logfile, args.max_failures)
    
    # Save JSON report
    with args.output.open('w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Analysis complete! Results saved to {args.output}")
    
    # Generate HTML report
    if not args.no_html:
        html_output = args.output.with_suffix('.html')
        try:
            generate_html_report(args.output, html_output)
            logger.info(f"HTML report generated: {html_output}")
        except Exception as e:
            logger.warning(f"HTML report generation failed: {e}")
    
    # Print summary
    stats = results.get('stats', {})
    print("\n" + "=" * 80)
    print("LOG ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"Analyzed file: {args.logfile}")
    print(f"Total lines: {stats.get('total_lines', 0):,}")
    print(f"Parsed entries: {stats.get('parsed_entries', 0):,}")
    print(f"Rule-based failures: {stats.get('rule_failures', 0):,}")
    print(f"ML-detected failures: {stats.get('ml_failures', 0):,}")
    print(f"Critical errors: {stats.get('critical_errors', 0):,}")
    
    # Enhanced metrics if available
    if stats.get('connection_issues', 0) > 0:
        print(f"Connection issues: {stats.get('connection_issues', 0):,}")
    if stats.get('http_errors', 0) > 0:
        print(f"HTTP errors: {stats.get('http_errors', 0):,}")
    if stats.get('timeout_issues', 0) > 0:
        print(f"Timeout issues: {stats.get('timeout_issues', 0):,}")
    
    # Primary Failures (Root Causes)
    root_cause_analysis = results.get('root_cause_analysis', {})
    primary_failures = root_cause_analysis.get('primary_failures', [])
    if primary_failures:
        print(f"\n🎯 PRIMARY FAILURES (ROOT CAUSES): {len(primary_failures)}")
        print("   These are the main failures that caused other errors:")
        for i, pf in enumerate(primary_failures[:5], 1):  # Show top 5
            print(f"   {i}. [{pf.get('type', 'unknown').replace('_', ' ').title()}] "
                  f"Impact: {pf.get('impact_count', 1)} failures "
                  f"(Cascade: {pf.get('cascade_count', 0)})")
            print(f"      Component: {pf.get('component', 'N/A')} | "
                  f"Severity: {pf.get('severity', 0):.2f}")
    
    # Recommendations
    recommendations = results.get('recommendations', [])
    if recommendations:
        print(f"\nRecommendations: {len(recommendations)}")
        for rec in recommendations[:3]:  # Show first 3
            print(f"  [{rec.get('priority', 'info').upper()}] {rec.get('message', '')}")
    
    print(f"\nProcessing time: {stats.get('processing_time', 0):.2f}s")
    print("=" * 80)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

