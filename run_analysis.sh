#!/bin/bash
# macOS/Linux script to run log analysis
# Usage: ./run_analysis.sh [logfile]

cd "$(dirname "$0")"

LOG_FILE="${1:-data/server.log}"

echo "=========================================="
echo "Log Analyzer - Running Analysis"
echo "=========================================="
echo ""

# Run analysis
python3 scripts/analyze.py --logfile "$LOG_FILE"

# Open HTML report if it exists
if [ -f "reports/analysis_report.html" ]; then
    echo ""
    echo "Opening HTML report..."
    open reports/analysis_report.html
fi

