# 🚀 Step-by-Step Deployment Guide

I'll walk you through deploying your chatbot. Follow these steps:

## Step 1: Prepare Your Code (I'll help with this)

Run this command in your project directory:
```bash
chmod +x deploy_setup.sh
./deploy_setup.sh
```

## Step 2: Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `log-analyzer-chatbot` (or any name you like)
3. Make it **Public** (required for free Render tier) or **Private** (if you have GitHub Pro)
4. **Don't** initialize with README (we already have one)
5. Click "Create repository"

## Step 3: Push Code to GitHub

Run these commands (replace YOUR_USERNAME and YOUR_REPO_NAME):

```bash
# Add all files
git add .

# Commit
git commit -m "Initial commit - Log Analyzer Chatbot ready for deployment"

# Add remote (replace with your actual repo URL)
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# Push to GitHub
git branch -M main
git push -u origin main
```

## Step 4: Deploy on Render (Easiest - 5 minutes)

1. **Sign up**: Go to https://render.com
   - Click "Get Started for Free"
   - Sign up with GitHub (recommended)

2. **Create Web Service**:
   - Click "New +" button (top right)
   - Select "Web Service"
   - Click "Connect account" next to GitHub (if not connected)
   - Authorize Render to access your repositories
   - Select your repository: `YOUR_USERNAME/YOUR_REPO_NAME`

3. **Configure Service**:
   - **Name**: `log-analyzer-chatbot` (or any name)
   - **Environment**: `Python 3`
   - **Region**: Choose closest to you
   - **Branch**: `main`
   - **Root Directory**: Leave empty
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python app.py`
   - **Plan**: Select **Free**

4. **Add Environment Variables** (for DeepSeek LLM):
   - Click "Advanced" → "Add Environment Variable"
   - **Key**: `DEEPSEEK_API_KEY`
   - **Value**: `sk-4628b1d9711f4c2cb3aa5fbbdb28b290`
   - Click "Add"

5. **Deploy**:
   - Scroll down and click "Create Web Service"
   - Wait ~5 minutes for deployment
   - You'll see build logs in real-time

6. **Get Your URL**:
   - Once deployed, you'll see: `https://your-app-name.onrender.com`
   - **Share this URL with your team!**

## Step 5: Test Your Deployment

1. Open the URL in your browser
2. Try uploading a log file or pasting log content
3. Test the chat functionality
4. Verify everything works

## ✅ Done!

Your chatbot is now live and accessible to your team!

---

## Alternative: Railway (Even Faster)

If Render doesn't work, try Railway:

1. Go to https://railway.app
2. Sign up with GitHub
3. Click "New Project"
4. Select "Deploy from GitHub repo"
5. Choose your repository
6. Railway auto-detects everything!
7. Add environment variable: `DEEPSEEK_API_KEY` = `sk-4628b1d9711f4c2cb3aa5fbbdb28b290`
8. Done! Get URL: `https://your-app.up.railway.app`

---

## Troubleshooting

**Build fails?**
- Check build logs in Render dashboard
- Make sure `requirements.txt` is correct
- Verify Python version (3.11 recommended)

**App won't start?**
- Check logs in Render dashboard
- Verify `PORT` environment variable (Render sets this automatically)
- Check start command: `python app.py`

**502 Error?**
- Free tier apps sleep after 15 min inactivity
- First request after sleep takes ~30 seconds
- This is normal for free tier

**Need help?**
- Check `docs/DEPLOYMENT_GUIDE.md` for detailed troubleshooting
- Render docs: https://render.com/docs

---

## 🎉 Success!

Once deployed, share the URL with your team. They can:
- Access the chatbot from anywhere
- Upload logs for analysis
- Get intelligent responses about failures
- Use it 24/7 (with free tier sleep limitation)

