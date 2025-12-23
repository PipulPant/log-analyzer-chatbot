#!/bin/bash
# Run web application

cd "$(dirname "$0")"

echo "Starting Log Analyzer Web Application..."
echo "========================================"
echo ""
echo "🔍 Checking for available port (starting from 5000)..."
echo "📝 The server will display the actual URL when ready"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python3 app.py

