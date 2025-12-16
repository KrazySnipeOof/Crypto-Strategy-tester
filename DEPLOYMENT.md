# Web Deployment Guide

This guide will help you deploy your Crypto Quant Liquidity Simulator to the web.

## Option 1: Streamlit Cloud (Recommended - Free & Easy)

### Prerequisites
- ✅ Your code is already on GitHub: https://github.com/KrazySnipeOof/Crypto-Strategy-tester.git
- ✅ `requirements.txt` exists
- ✅ Main app file: `crypto_quant_liquidity_simulator.py`

### Steps

1. **Sign up for Streamlit Cloud**
   - Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
   - Click "Sign up" and use your GitHub account

2. **Deploy Your App**
   - Click "New app" button
   - Select repository: `KrazySnipeOof/Crypto-Strategy-tester`
   - Set main file path: `crypto_quant_liquidity_simulator.py`
   - Click "Deploy"

3. **Your App is Live!**
   - Streamlit Cloud will provide a URL like: `https://crypto-strategy-tester.streamlit.app`
   - Share this URL with anyone!

### Streamlit Cloud Features
- ✅ Free hosting
- ✅ Automatic deployments on git push
- ✅ Custom subdomain
- ✅ HTTPS enabled
- ✅ No credit card required

---

## Option 2: Railway

### Steps

1. **Sign up** at [Railway.app](https://railway.app)
2. **Create New Project** → "Deploy from GitHub repo"
3. **Select Repository**: `KrazySnipeOof/Crypto-Strategy-tester`
4. **Railway will automatically detect** the `Procfile` and deploy
5. **Your app will be live** at a Railway-provided URL

### Railway Features
- ✅ Free tier available
- ✅ Automatic deployments
- ✅ Custom domains supported

---

## Option 3: Render

### Steps

1. **Sign up** at [Render.com](https://render.com)
2. **Create New** → "Web Service"
3. **Connect GitHub** repository: `KrazySnipeOof/Crypto-Strategy-tester`
4. **Configure**:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run crypto_quant_liquidity_simulator.py --server.port $PORT --server.address 0.0.0.0`
5. **Deploy**

### Render Features
- ✅ Free tier available
- ✅ Automatic deployments
- ✅ Custom domains supported

---

## Important Notes

### Large Data Files
Your historical data files (BTC, ETH, SOL) are very large. For web deployment:

**Option A: Use GitHub LFS** (Recommended for keeping data)
```bash
git lfs install
git lfs track "*.txt"
git add .gitattributes
git commit -m "Add LFS tracking"
git push
```

**Option B: Exclude Data Files** (For faster deployment)
- Users can upload their own data via the UI
- Or host data separately and load via API

**Option C: Use Sample Data**
- Create smaller sample datasets for web deployment
- Full data can be loaded locally

### Environment Variables (Optional)
If you need to set environment variables:
- **Streamlit Cloud**: Settings → Secrets
- **Railway**: Variables tab
- **Render**: Environment section

### Custom Domain
All platforms support custom domains:
- Streamlit Cloud: Settings → Custom domain
- Railway: Settings → Custom domain
- Render: Settings → Custom domain

---

## Troubleshooting

### App Won't Deploy
- Check that `requirements.txt` has all dependencies
- Verify main file path is correct
- Check build logs for errors

### App Deploys But Shows Errors
- Check Streamlit Cloud logs
- Verify all imports are in requirements.txt
- Check file paths are correct

### Large Files Issue
- Use GitHub LFS for large data files
- Or exclude data files and use UI upload

---

## Quick Deploy Checklist

- [x] Code pushed to GitHub
- [x] `requirements.txt` exists
- [x] Main app file exists
- [x] `.streamlit/config.toml` created (optional)
- [x] `Procfile` created (for Railway/Render)
- [ ] Sign up for hosting platform
- [ ] Connect GitHub repository
- [ ] Deploy!

---

## Need Help?

- Streamlit Cloud Docs: https://docs.streamlit.io/streamlit-cloud
- Railway Docs: https://docs.railway.app
- Render Docs: https://render.com/docs

