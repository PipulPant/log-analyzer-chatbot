# Deployment Guide - Free Hosting Options

This guide covers multiple free hosting options to deploy your Log Analyzer Chatbot for team sharing.

## 🚀 Quick Comparison

| Platform | Free Tier | Ease | Best For |
|----------|-----------|------|----------|
| **Render** | ✅ Yes | ⭐⭐⭐⭐⭐ | Easiest, best free tier |
| **Railway** | ✅ Yes ($5 credit) | ⭐⭐⭐⭐ | Good balance |
| **Fly.io** | ✅ Yes | ⭐⭐⭐⭐ | Global edge deployment |
| **PythonAnywhere** | ✅ Yes | ⭐⭐⭐ | Simple Python hosting |
| **Replit** | ✅ Yes | ⭐⭐⭐ | Quick testing |

---

## Option 1: Render (Recommended - Easiest)

### Why Render?
- ✅ **Free tier**: 750 hours/month (enough for 24/7)
- ✅ **Automatic HTTPS**
- ✅ **Easy setup** from GitHub
- ✅ **Auto-deploy** on git push
- ✅ **No credit card required**

### Steps:

1. **Prepare your repository**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

2. **Sign up**: Go to https://render.com and sign up with GitHub

3. **Create New Web Service**:
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Select the repository

4. **Configure**:
   - **Name**: `log-analyzer-chatbot` (or any name)
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python app.py`
   - **Plan**: `Free`

5. **Environment Variables** (if using DeepSeek):
   - Add: `DEEPSEEK_API_KEY` = `your-api-key`
   - Or keep using `llm_config.json` (file will be in repo)

6. **Deploy**: Click "Create Web Service"

7. **Get URL**: Your app will be at `https://log-analyzer-chatbot.onrender.com`

### Notes:
- First deploy takes ~5 minutes
- Free tier sleeps after 15 min inactivity (wakes up on next request)
- Can upgrade to paid for always-on

---

## Option 2: Railway (Good Alternative)

### Why Railway?
- ✅ **$5 free credit/month**
- ✅ **Fast deployment**
- ✅ **Good performance**
- ⚠️ Requires credit card (but free tier available)

### Steps:

1. **Sign up**: https://railway.app (use GitHub)

2. **Create New Project**:
   - Click "New Project"
   - "Deploy from GitHub repo"
   - Select your repository

3. **Configure**:
   - Railway auto-detects Python
   - Uses `railway.json` if present
   - Or manually set:
     - **Start Command**: `python app.py`
     - **Health Check**: `/api/health`

4. **Environment Variables**:
   - Add `DEEPSEEK_API_KEY` if needed
   - Add `PORT` (Railway sets this automatically)

5. **Deploy**: Railway auto-deploys

6. **Get URL**: Railway provides a URL like `https://your-app.up.railway.app`

### Notes:
- Free tier: $5 credit/month (~100 hours)
- Can upgrade for more

---

## Option 3: Fly.io (Global Edge)

### Why Fly.io?
- ✅ **Free tier**: 3 shared VMs
- ✅ **Global edge deployment**
- ✅ **Fast worldwide**
- ⚠️ Requires credit card

### Steps:

1. **Install Fly CLI**:
   ```bash
   # macOS
   brew install flyctl
   
   # Windows (PowerShell)
   iwr https://fly.io/install.ps1 -useb | iex
   ```

2. **Login**:
   ```bash
   fly auth login
   ```

3. **Create Fly App**:
   ```bash
   fly launch
   ```
   - Follow prompts
   - Select region
   - Don't deploy yet

4. **Create `fly.toml`** (if not auto-generated):
   ```toml
   app = "your-app-name"
   primary_region = "iad"
   
   [build]
     dockerfile = "Dockerfile"
   
   [http_service]
     internal_port = 5000
     force_https = true
     auto_stop_machines = true
     auto_start_machines = true
     min_machines_running = 0
   
   [[services]]
     http_checks = []
     internal_port = 5000
     processes = ["app"]
     protocol = "tcp"
     script_checks = []
   ```

5. **Set Secrets** (API keys):
   ```bash
   fly secrets set DEEPSEEK_API_KEY=your-key
   ```

6. **Deploy**:
   ```bash
   fly deploy
   ```

7. **Get URL**: `https://your-app-name.fly.dev`

---

## Option 4: PythonAnywhere (Simplest)

### Why PythonAnywhere?
- ✅ **Free tier**: 1 web app
- ✅ **Simple Python hosting**
- ✅ **No Docker needed**
- ⚠️ Limited resources

### Steps:

1. **Sign up**: https://www.pythonanywhere.com

2. **Upload files**:
   - Go to "Files" tab
   - Upload your project files
   - Or use Git: `git clone <your-repo>`

3. **Install dependencies**:
   - Open "Bash" console
   ```bash
   pip3.10 install --user -r requirements.txt
   ```

4. **Create Web App**:
   - Go to "Web" tab
   - Click "Add a new web app"
   - Select "Manual configuration"
   - Python 3.10

5. **Configure WSGI**:
   - Edit WSGI file:
   ```python
   import sys
   path = '/home/yourusername/log-analyzer'
   if path not in sys.path:
       sys.path.append(path)
   
   from app import app as application
   ```

6. **Set environment variables**:
   - In WSGI file or via "Web" → "Environment variables"

7. **Reload**: Click "Reload" button

8. **Get URL**: `https://yourusername.pythonanywhere.com`

---

## Option 5: Replit (Quick Testing)

### Why Replit?
- ✅ **Free tier**
- ✅ **Instant deployment**
- ✅ **Built-in IDE**
- ⚠️ Limited for production

### Steps:

1. **Sign up**: https://replit.com

2. **Import from GitHub**:
   - Click "Create Repl"
   - "Import from GitHub"
   - Enter repo URL

3. **Install dependencies**:
   - Replit auto-installs from `requirements.txt`

4. **Run**:
   - Click "Run" button
   - Replit provides URL

5. **Deploy**:
   - Click "Deploy" button
   - Follow prompts

---

## 🔧 Common Configuration

### Update app.py for Production

Make sure `app.py` uses the PORT environment variable:

```python
if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
```

### Environment Variables to Set

- `PORT`: Usually auto-set by platform
- `DEEPSEEK_API_KEY`: If using DeepSeek LLM
- `FLASK_ENV`: Set to `production`

### Security Notes

1. **Don't commit API keys**: Use environment variables
2. **Update `.gitignore`**: Already includes `llm_config.json`
3. **Use HTTPS**: All platforms provide this automatically

---

## 📝 Pre-Deployment Checklist

- [ ] Test locally: `python app.py`
- [ ] Verify `requirements.txt` is complete
- [ ] Check `Procfile` or `Dockerfile` exists
- [ ] Remove debug mode in production
- [ ] Set environment variables
- [ ] Test health endpoint: `/api/health`
- [ ] Verify file uploads work
- [ ] Test chat functionality

---

## 🐛 Troubleshooting

### App won't start
- Check logs in platform dashboard
- Verify `requirements.txt` dependencies
- Check PORT environment variable

### 502 Bad Gateway
- App might be sleeping (free tier)
- Wait a moment and refresh
- Check app logs

### Import errors
- Verify all dependencies in `requirements.txt`
- Check Python version (3.11 recommended)

### File upload issues
- Check `MAX_CONTENT_LENGTH` setting
- Verify temp directory permissions

---

## 🎯 Recommended: Render

For most users, **Render** is the best choice:
- ✅ Easiest setup
- ✅ Good free tier
- ✅ Auto-deploy from GitHub
- ✅ No credit card needed
- ✅ Automatic HTTPS

---

## 📚 Additional Resources

- Render Docs: https://render.com/docs
- Railway Docs: https://docs.railway.app
- Fly.io Docs: https://fly.io/docs
- PythonAnywhere Docs: https://help.pythonanywhere.com

---

## 🚀 Quick Deploy Script

Save as `deploy.sh`:

```bash
#!/bin/bash
echo "🚀 Deploying to Render..."

# Commit changes
git add .
git commit -m "Deploy to production"

# Push to trigger Render deployment
git push origin main

echo "✅ Deployment triggered! Check Render dashboard."
```

Make executable: `chmod +x deploy.sh`

Then: `./deploy.sh`

