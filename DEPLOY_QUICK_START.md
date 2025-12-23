# 🚀 Quick Deploy Guide - Share with Your Team

## Easiest Option: Render (5 minutes)

### Step 1: Push to GitHub
```bash
git init
git add .
git commit -m "Deploy chatbot"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

### Step 2: Deploy on Render
1. Go to https://render.com
2. Sign up with GitHub
3. Click "New +" → "Web Service"
4. Connect your repository
5. Configure:
   - **Name**: `log-analyzer-chatbot`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python app.py`
   - **Plan**: `Free`
6. Click "Create Web Service"
7. Wait ~5 minutes for deployment
8. **Done!** Share the URL with your team: `https://your-app.onrender.com`

### Step 3: Add API Key (Optional - for LLM)
1. In Render dashboard → "Environment"
2. Add: `DEEPSEEK_API_KEY` = `your-key`
3. Redeploy

---

## Alternative: Railway (3 minutes)

1. Go to https://railway.app
2. Sign up with GitHub
3. "New Project" → "Deploy from GitHub repo"
4. Select your repo
5. **Done!** Railway auto-detects and deploys
6. Share URL: `https://your-app.up.railway.app`

---

## What Your Team Gets

✅ **Web Interface**: ChatGPT-style chat UI  
✅ **Log Analysis**: Upload logs or paste content  
✅ **AI-Powered**: Intelligent responses (if LLM configured)  
✅ **Shareable URL**: Access from anywhere  
✅ **Free**: No cost for basic usage  

---

## Troubleshooting

**App sleeping?** (Free tier limitation)
- First request takes ~30 seconds to wake up
- Consider upgrading for always-on

**Need help?** Check `docs/DEPLOYMENT_GUIDE.md` for detailed instructions.

---

## 🎉 That's it!

Your team can now access the chatbot at the provided URL!

