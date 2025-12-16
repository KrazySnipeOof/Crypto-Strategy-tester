# Crypto Quant Liquidity Simulator

A Streamlit-based backtesting simulator for crypto trading strategies with funded account rules.

## Features

- **Multi-Asset Support**: BTC, ETH, XRP, SOL
- **Historical Data Import**: Load real BTC/ETH minute-by-minute data
- **Trading Window**: 7:30-11:00 AM Chicago time
- **Signal Analysis**: Price Action + VSA + CVD + Liquidity levels
- **Funded Account Rules**: 2% daily DD, 3% max DD, $500 target on $5k
- **Monte Carlo Simulation**: Run multiple simulations to estimate pass probability
- **Sensitivity Analysis**: Test parameter variations

## Quick Start

### Option 1: Launch with Chrome (Recommended)

Double-click `launch_app.py` or run:
```bash
python launch_app.py
```

This will:
- Start the Streamlit server
- Automatically open Chrome to `http://localhost:8501`

### Option 2: Manual Launch

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the app:
```bash
streamlit run crypto_quant_liquidity_simulator.py
```

3. Open Chrome and navigate to: `http://localhost:8501`

### Option 3: Batch File (Windows)

Double-click `run_chrome.bat` to launch the app.

## Data Sources

### 1. Synthetic Data (Default)
- Generates random trades for testing
- Configurable win rate, avg win/loss

### 2. CSV Files
- Upload custom trade data
- Required columns: `symbol`, `entry_price`, `exit_price`, `entry_time`
- See help section in app for full format

### 3. Historical Crypto Data
- Load BTC/ETH minute-by-minute data
- Supports multiple years (2019-2025)
- Automatically filters to trading window
- Default directories:
  - BTC: `BTCUSD DATA/`
  - ETH: `ETHUSD DATA/`

## Historical Data Format

Semicolon-separated files:
```
YYYYMMDD HHMMSS;Open;High;Low;Close;Volume
20250101 083000;93576.0;93610.93;93537.5;93610.93;8.21827
```

## Usage

1. **Select Data Source**: Choose Synthetic, CSV, or Historical Crypto Data
2. **Configure Parameters**: Set account size, DD limits, position sizes
3. **Run Backtest**: Click "Run Backtest" to see results
4. **Analyze**: View performance metrics, equity curve, trade log
5. **Monte Carlo**: Run multiple simulations to estimate pass rate
6. **Sensitivity**: Test how parameters affect results

## File Structure

```
Quant/
├── crypto_quant_liquidity_simulator.py  # Main app
├── launch_app.py                        # Chrome launcher
├── run_chrome.bat                       # Windows batch launcher
├── requirements.txt                     # Dependencies
├── example_trades.csv                   # Example CSV format
├── BTCUSD DATA/                         # BTC historical data
│   ├── BTCUSDT_2019_minute.Last.txt
│   ├── BTCUSDT_2020_minute.Last.txt
│   └── ...
└── ETHUSD DATA/                         # ETH historical data
    ├── ETHUSDT_2019_minute.Last.txt
    ├── ETHUSDT_2020_minute.Last.txt
    └── ...
```

## Requirements

- Python 3.8+
- Google Chrome (for auto-launch)
- See `requirements.txt` for Python packages

## Notes

- The app runs on `http://localhost:8501` by default
- Press `Ctrl+C` in the terminal to stop the server
- Historical data is automatically filtered to trading hours (7:30-11:00 AM Chicago time)
- Large datasets can be sampled using the "Sample Interval" setting

