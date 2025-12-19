# Daily Bias Calendar - Web-Based Workflow

## Why Web-Based is Better

✅ **No Jupyter widget issues** - No scoping errors, kernel restarts, or widget breaking  
✅ **Easy to share** - Just send a link (GitHub Pages) or run locally  
✅ **Better UX** - Modern, responsive web interface  
✅ **Stable** - Standard web technologies, no Jupyter quirks  
✅ **Easy maintenance** - Standard React/TypeScript stack  

## Quick Start

### Step 1: Generate Bias Data (One-time or when data updates)

Run the backtest notebook to generate bias data, then export to CSV:

```bash
# Option A: Run the export script
python export_bias_data.py

# Option B: Or manually in Jupyter:
# 1. Run cells 0-6 in backtest.ipynb (data loading through summary)
# 2. Then run:
#    bias_results.to_csv('calendar-app/public/bias-data.csv', index=False)
```

### Step 2: Run the Web App

```bash
cd calendar-app
npm install  # Only needed first time
npm run dev  # Starts local server at http://localhost:5173
```

Open your browser to `http://localhost:5173` - the calendar will work perfectly!

### Step 3: Build for Production (Optional)

```bash
cd calendar-app
npm run build  # Creates dist/ folder with production files
```

## Deploy to GitHub Pages (Optional)

1. **Build the app:**
   ```bash
   cd calendar-app
   npm run build
   ```

2. **Push to GitHub:**
   ```bash
   # From calendar-app/dist directory
   git add dist/
   git commit -m "Add calendar app build"
   git push origin main
   ```

3. **Enable GitHub Pages:**
   - Go to your repo settings
   - Enable GitHub Pages
   - Select `/dist` folder as source
   - Your calendar will be live at: `https://yourusername.github.io/Crypto-Strategy-tester/`

## Workflow Summary

```
┌─────────────────┐
│  Run Backtest   │  (Jupyter notebook - generates bias data)
│   Notebook      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Export to CSV   │  (python export_bias_data.py)
│  bias-data.csv  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Web Calendar   │  (npm run dev - no Jupyter needed!)
│     App         │
└─────────────────┘
```

## Benefits

- **Separation of concerns**: Data generation (Jupyter) vs. Visualization (Web)
- **No more widget errors**: Pure web app, no Jupyter widget issues
- **Easy sharing**: Just share the URL
- **Better performance**: Web app is faster than Jupyter widgets
- **Mobile friendly**: Works on phones/tablets

## Troubleshooting

**CSV not loading?**
- Make sure `bias-data.csv` is in `calendar-app/public/` folder
- Check browser console for errors
- Verify CSV has columns: `date`, `symbol`, `bias`, `scenario`

**App won't start?**
- Run `npm install` in `calendar-app/` directory
- Make sure Node.js is installed

**Data not updating?**
- Re-run `export_bias_data.py` after updating backtest results
- Refresh the browser

