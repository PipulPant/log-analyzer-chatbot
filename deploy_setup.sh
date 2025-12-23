#!/bin/bash
# Deployment Setup Script
# This script prepares your project for deployment

echo "🚀 Preparing Log Analyzer Chatbot for Deployment..."
echo ""

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "📦 Initializing Git repository..."
    git init
    echo "✅ Git initialized"
else
    echo "✅ Git repository already exists"
fi

# Check if .gitignore includes llm_config.json
if ! grep -q "llm_config.json" .gitignore 2>/dev/null; then
    echo "🔒 Updating .gitignore to protect API keys..."
    echo "" >> .gitignore
    echo "# API keys and secrets" >> .gitignore
    echo "data/models/llm_config.json" >> .gitignore
    echo "✅ .gitignore updated"
else
    echo "✅ .gitignore already configured"
fi

# Create a README for deployment if it doesn't exist
if [ ! -f "README.md" ]; then
    echo "📝 Creating README.md..."
    cat > README.md << 'EOF'
# Log Analyzer Chatbot

AI-powered log analysis chatbot with intelligent error detection and root cause analysis.

## Features

- 🤖 Intelligent chatbot interface (ChatGPT-style)
- 🔍 Advanced log analysis with ML models
- 🎯 Root cause analysis
- 💬 Conversational AI with context memory
- 📊 Detailed HTML reports

## Quick Start

1. Install dependencies: `pip install -r requirements.txt`
2. Run: `python app.py`
3. Open: http://localhost:5000

## Deployment

See `DEPLOY_QUICK_START.md` for deployment instructions.

## Documentation

- `docs/DEPLOYMENT_GUIDE.md` - Full deployment guide
- `docs/LLM_INTEGRATION_GUIDE.md` - LLM setup guide
- `WEB_APP_GUIDE.md` - Web app usage guide
EOF
    echo "✅ README.md created"
else
    echo "✅ README.md exists"
fi

echo ""
echo "✅ Project is ready for deployment!"
echo ""
echo "📋 Next Steps:"
echo "1. Create a GitHub repository (if you haven't already)"
echo "2. Run these commands:"
echo "   git add ."
echo "   git commit -m 'Initial commit - ready for deployment'"
echo "   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git"
echo "   git push -u origin main"
echo ""
echo "3. Then follow DEPLOY_QUICK_START.md to deploy on Render"
echo ""

