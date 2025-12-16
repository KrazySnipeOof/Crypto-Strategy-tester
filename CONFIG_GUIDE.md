# Configuration Guide

The app now supports code-based configuration! Edit the `CONFIG` dictionary at the top of `main()` function to set all parameters directly in code.

## Quick Start

1. Open `crypto_quant_liquidity_simulator.py`
2. Find the `CONFIG` dictionary (around line 765)
3. Edit the values you want
4. Run the app - it will automatically use your settings!

## Configuration Options

### Data Source
```python
"data_source": "historical",  # Options: "synthetic", "csv", or "historical"
```

### Historical Data Settings
```python
"historical_symbols": ["BTC", "ETH"],  # Load BTC, ETH, or both
"historical_directories": {
    "BTC": "BTCUSD DATA",  # Path to your BTC data folder
    "ETH": "ETHUSD DATA"   # Path to your ETH data folder
},
"sample_interval": 5,  # Use every 5th minute (1 = all bars)
"max_trades": 0,  # 0 = no limit, or set max number of trades
```

### CSV Data Settings
```python
"csv_directory": "",  # Path to folder with CSV files
"csv_file_paths": [],  # Or specify exact file paths: ["file1.csv", "file2.csv"]
```

### Account Parameters
```python
"account_size": 5000,
"target_gain": 500,
"daily_dd_limit": 0.02,  # 2%
"max_dd_limit": 0.03,  # 3%
```

### Position Parameters
```python
"entry_size": 4800,
"stop_loss": 100,
"take_profit": 300,
```

### Strategy Parameters (for synthetic data)
```python
"trading_days": 6,
"num_trades": 40,
"win_rate": 0.55,
"avg_win": 300,
"avg_loss": 100,
```

### UI Settings
```python
"use_ui_overrides": True,  # Set to False to disable UI and use config only
```

## Examples

### Example 1: Load BTC Historical Data Only
```python
CONFIG = {
    "data_source": "historical",
    "historical_symbols": ["BTC"],
    "historical_directories": {
        "BTC": "BTCUSD DATA",
        "ETH": "ETHUSD DATA"
    },
    "sample_interval": 5,
    "max_trades": 0,
    "use_ui_overrides": False,  # Use config only, no UI controls
    # ... other settings
}
```

### Example 2: Load Both BTC and ETH
```python
CONFIG = {
    "data_source": "historical",
    "historical_symbols": ["BTC", "ETH"],
    "historical_directories": {
        "BTC": "BTCUSD DATA",
        "ETH": "ETHUSD DATA"
    },
    "sample_interval": 10,  # Every 10 minutes
    "max_trades": 1000,  # Limit to 1000 trades
    # ... other settings
}
```

### Example 3: Use CSV Files
```python
CONFIG = {
    "data_source": "csv",
    "csv_directory": "my_trades",  # Folder with CSV files
    # OR
    "csv_file_paths": ["trades_2024.csv", "trades_2025.csv"],
    # ... other settings
}
```

### Example 4: Synthetic Data with Custom Settings
```python
CONFIG = {
    "data_source": "synthetic",
    "trading_days": 12,
    "num_trades": 60,
    "win_rate": 0.60,
    "avg_win": 400,
    "avg_loss": 150,
    # ... other settings
}
```

## How It Works

1. **With UI Overrides (default)**: 
   - UI controls are shown
   - Config values are used as defaults
   - You can change values in the UI
   - Data auto-loads if configured

2. **Without UI Overrides** (`"use_ui_overrides": False`):
   - UI controls are hidden
   - Only config values are used
   - Data auto-loads automatically
   - Backtest runs automatically when data is loaded

## Tips

- Set `"use_ui_overrides": False` for fully automated runs
- Use `"sample_interval": 5` or higher for large datasets to reduce processing time
- Set `"max_trades"` to limit data size for faster testing
- Check that directory paths match your actual folder names

