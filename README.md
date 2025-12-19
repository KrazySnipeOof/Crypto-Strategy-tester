# Crypto Trading Strategy Monte Carlo Analysis

A comprehensive Jupyter notebook for backtesting crypto trading strategies with Monte Carlo simulation, using historical data and advanced strategy frameworks.

## Features

- **Multi-Asset Support**: BTC, ETH, SOL historical data analysis
- **Historical Data Import**: Load real minute-by-minute crypto data from CSV files
- **Daily Bias 4H Strategy Framework**: Advanced multi-timeframe strategy with:
  - 4H Session-based bias detection (Asia, London, NY sessions)
  - CISD (Change in Structure Direction) on 5m timeframe
  - Multi-timeframe confirmation (4H bias + 5m CISD signals)
  - Session reversal patterns (Asia Reversal, London Reversal, NY 6am/10am)
- **Monte Carlo Simulation**: Run multiple simulations to assess strategy robustness
- **Comprehensive Metrics**:
  - Sharpe Ratio (risk-adjusted returns)
  - Sortino Ratio (downside risk-adjusted returns)
  - Expected Value (average PnL per trade)
  - Pass Rate (funded account rules compliance)
- **Visual Analysis**: 10+ visualization charts including distributions, scatter plots, and CDFs
- **Trading Window**: 7:30-11:30 AM Chicago time, Monday-Saturday

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Open Jupyter Notebook

```bash
jupyter notebook notebook.ipynb
```

Or use JupyterLab:

```bash
jupyter lab notebook.ipynb
```

### 3. Run All Cells

Click "Run All" or execute cells sequentially to:
- Load historical crypto data
- Prepare trades from data
- Run Monte Carlo simulations
- View comprehensive visualizations

## Data Sources

### Historical Crypto Data

The notebook loads minute-by-minute data from `.Last.txt` files in:
- `BTCUSD DATA/` - Bitcoin historical data (2019-2025)
- `ETHUSD DATA/` - Ethereum historical data (2019-2025)
- `SOLUSD DATA/` - Solana historical data (2023-2025)

### Data Format

Semicolon-separated files:
```
YYYYMMDD HHMMSS;Open;High;Low;Close;Volume
20250101 083000;93576.0;93610.93;93537.5;93610.93;8.21827
```

## Notebook Structure

1. **Data Loading**: Loads historical crypto data from CSV files
2. **Backtesting Engine**: Custom `CryptoBacktester` class with funded account rules
3. **Strategy Framework**: Daily Bias 4H Strategy with multi-timeframe analysis
4. **Trade Preparation**: Converts price data to trade format with trading window filtering
5. **Monte Carlo Simulation**: Runs multiple backtests with random sampling
6. **Visualizations**: Comprehensive charts showing:
   - Sharpe Ratio Distribution
   - Sortino Ratio Distribution
   - Expected Value Distribution
   - Return Distribution
   - Max Drawdown Distribution
   - Pass Rate Visualization
   - Scatter plots (Sharpe vs Sortino, Expected Value vs Return)
   - Cumulative Distribution Functions

## Strategy Configuration

The notebook uses the **Daily Bias 4H Strategy Framework** (`daily_bias_4h_strategy.py`), which includes:

- **4H Session Analysis**: Identifies session-based bias (Asia, London, NY sessions)
- **CISD Detection**: Change in Structure Direction on 5-minute timeframe
- **Entry Rules**: 
  - Long: 4H bias bullish AND Bullish CISD formed within entry window
  - Short: 4H bias bearish AND Bearish CISD formed within entry window
- **Trading Window**: 6am-10am NYC time (configurable)

## Key Metrics Explained

- **Sharpe Ratio**: Risk-adjusted return (>1.0 is good, >2.0 is excellent)
- **Sortino Ratio**: Downside risk-adjusted return (only penalizes negative volatility)
- **Expected Value**: Average profit/loss per trade (positive = profitable strategy)
- **Pass Rate**: Percentage of simulations that pass funded account rules:
  - 2% daily drawdown limit
  - 3% maximum drawdown limit
  - $500 target gain on $5,000 account

## File Structure

```
Quant/
├── notebook.ipynb                    # Main Jupyter notebook
├── daily_bias_4h_strategy.py         # Daily Bias 4H Strategy Framework
├── requirements.txt                  # Python dependencies
├── BTCUSD DATA/                      # BTC historical data
│   ├── BTCUSDT_2019_minute.Last.txt
│   ├── BTCUSDT_2020_minute.Last.txt
│   └── ...
├── ETHUSD DATA/                      # ETH historical data
│   ├── ETHUSDT_2019_minute.Last.txt
│   └── ...
└── SOLUSD DATA/                      # SOL historical data
    ├── SOLUSDT_2023_minute.Last.txt
    └── ...
```

## Requirements

- Python 3.8+
- Jupyter Notebook or JupyterLab
- See `requirements.txt` for Python packages:
  - pandas >= 2.3.0
  - numpy >= 2.3.0
  - matplotlib
  - seaborn
  - pytz >= 2025.2

## Usage Tips

1. **Run cells sequentially**: The notebook is designed to run from top to bottom
2. **Use all available data**: The notebook automatically uses all available CSV data points
3. **Adjust parameters**: Modify `NUM_SIMULATIONS` and `SAMPLE_SIZE` in the Monte Carlo section
4. **Strategy selection**: Toggle `USE_DAILY_BIAS_STRATEGY` to switch between strategies
5. **Visualizations**: All charts are generated automatically after Monte Carlo simulation

## Troubleshooting

### KeyError: 'sharpe_ratio'

If you encounter this error, run the fix cell (cell 22) which adds column mappings for backtester results. The backtester returns `'sharpe'` but the code expects `'sharpe_ratio'`.

### No Data Available

Ensure your data folders (`BTCUSD DATA/`, `ETHUSD DATA/`, etc.) contain `.Last.txt` files with the correct format.

### Import Errors

Make sure `daily_bias_4h_strategy.py` is in the same directory as `notebook.ipynb`.

## Notes

- Historical data is automatically filtered to trading hours (7:30-11:30 AM Chicago time)
- Large datasets are sampled efficiently for Monte Carlo simulations
- All visualizations are generated using matplotlib and seaborn
- The notebook uses up to 10,000 trades per simulation (or all available if less)

## License

This project is for educational and research purposes.
