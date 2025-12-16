# Custom Strategy Guide for Monte Carlo Simulation

This guide explains how to create and use custom trading strategies with the Monte Carlo simulator.

## Quick Start

### Option 1: Use Strategy File (Recommended)

1. Create a Python file (e.g., `my_strategy.py`) with your strategy function:
```python
def strategy(row, context=None):
    # Your strategy logic here
    if row['cvd'] > row['cvd_ma']:
        return 'BUY', 0.8
    return 'HOLD', 0.5
```

2. In `crypto_quant_liquidity_simulator.py`, update CONFIG:
```python
CONFIG = {
    # ... other settings ...
    "strategy_file": "my_strategy.py",
    "data_source": "historical",
    "historical_symbols": ["BTC", "ETH"],
    # ... rest of config ...
}
```

3. Run the app - Monte Carlo will use your strategy automatically!

### Option 2: Inline Strategy Code

In `crypto_quant_liquidity_simulator.py`, add your strategy code to CONFIG:
```python
CONFIG = {
    # ... other settings ...
    "strategy_code": """
def strategy(row, context=None):
    # Buy when CVD is above MA
    if row.get('cvd', 0) > row.get('cvd_ma', 0):
        return 'BUY', 0.8
    return 'HOLD', 0.5
""",
    # ... rest of config ...
}
```

## Strategy Function Requirements

Your strategy function **must** follow this signature:

```python
def strategy(row, context=None):
    """
    Parameters:
    -----------
    row : pd.Series
        Current bar/trade data with these columns:
        - entry_price, exit_price, high, low, volume
        - price_range, vsa_ratio, cvd, cvd_ma
        - liq_level_below, liq_level_above
        - recent_high, recent_low
        - symbol, entry_time
    
    context : dict, optional
        Additional context:
        - equity: current account equity
        - trades: list of previous trades
        - current_date: current trading date
    
    Returns:
    --------
    tuple: (signal_type, confidence)
        - signal_type: 'BUY', 'SELL', or 'HOLD'
        - confidence: float between 0.0 and 1.0
    """
    # Your logic here
    return 'BUY', 0.8  # or 'SELL', 0.6 or 'HOLD', 0.5
```

## Available Data in `row`

You can access these fields from the `row` parameter:

- **Price Data**: `entry_price`, `exit_price`, `high`, `low`
- **Volume Data**: `volume`, `vsa_ratio` (volume/price_range)
- **CVD Data**: `cvd`, `cvd_ma` (Cumulative Volume Delta and its moving average)
- **Liquidity Levels**: `liq_level_below`, `liq_level_above`
- **Recent Levels**: `recent_high`, `recent_low`
- **Metadata**: `symbol`, `entry_time`, `price_range`

## Example Strategies

### Example 1: Simple CVD Strategy
```python
def strategy(row, context=None):
    """Buy when CVD is above its moving average"""
    if row['cvd'] > row['cvd_ma']:
        return 'BUY', 0.8
    return 'HOLD', 0.5
```

### Example 2: Volume Breakout Strategy
```python
def strategy(row, context=None):
    """Buy on high volume breakouts above recent high"""
    vsa_ratio = row.get('vsa_ratio', 0)
    entry_price = row['entry_price']
    recent_high = row.get('recent_high', entry_price)
    
    if vsa_ratio > 500 and entry_price >= recent_high * 0.99:
        return 'BUY', 0.9
    return 'HOLD', 0.5
```

### Example 3: Liquidity Hunt Strategy
```python
def strategy(row, context=None):
    """Buy when price approaches liquidity levels"""
    entry_price = row['entry_price']
    liq_level_below = row.get('liq_level_below', entry_price * 0.98)
    distance_to_liq = abs(entry_price - liq_level_below) / entry_price
    
    if distance_to_liq < 0.003:  # Within 0.3% of liquidity
        return 'BUY', 0.85
    return 'HOLD', 0.5
```

### Example 4: Multi-Factor Strategy
```python
def strategy(row, context=None):
    """Combine multiple factors"""
    cvd = row.get('cvd', 0)
    cvd_ma = row.get('cvd_ma', 0)
    vsa_ratio = row.get('vsa_ratio', 0)
    entry_price = row['entry_price']
    recent_high = row.get('recent_high', entry_price)
    
    # Score signals
    score = 0
    if cvd > cvd_ma:
        score += 1
    if vsa_ratio > 300:
        score += 1
    if entry_price >= recent_high * 0.99:
        score += 1
    
    if score >= 2:
        confidence = 0.6 + (score * 0.1)
        return 'BUY', min(confidence, 1.0)
    return 'HOLD', 0.5
```

### Example 5: Using Context (Account Equity)
```python
def strategy(row, context=None):
    """Adjust strategy based on account equity"""
    if context:
        equity = context.get('equity', 5000)
        # Be more conservative if equity drops
        if equity < 4500:
            return 'HOLD', 0.5
    
    # Normal strategy
    if row['cvd'] > row['cvd_ma']:
        return 'BUY', 0.7
    return 'HOLD', 0.5
```

## Running Monte Carlo with Custom Strategy

1. **Set up your strategy** (file or inline code in CONFIG)

2. **Configure data source**:
```python
CONFIG = {
    "data_source": "historical",
    "historical_symbols": ["BTC", "ETH"],
    "historical_directories": {
        "BTC": "BTCUSD DATA",
        "ETH": "ETHUSD DATA"
    },
    "strategy_file": "my_strategy.py",  # or use strategy_code
    "num_simulations": 1000,  # Run 1000 Monte Carlo simulations
    # ... other settings ...
}
```

3. **Run the app** - The Monte Carlo tab will automatically use your custom strategy!

## Tips

- **Start Simple**: Begin with a basic strategy and add complexity gradually
- **Test Thoroughly**: Use Monte Carlo to test your strategy across many scenarios
- **Use Confidence Levels**: Higher confidence (0.8-1.0) for strong signals, lower (0.5-0.7) for weaker ones
- **Consider Context**: Use the `context` parameter to adjust strategy based on account state
- **Handle Missing Data**: Use `.get()` with defaults for optional fields
- **Backtest First**: Test on Single Backtest tab before running Monte Carlo

## Troubleshooting

**Error: "Strategy file must define a function called 'strategy'"**
- Make sure your Python file has a function named exactly `strategy`

**Error: "Strategy code must define a function called 'strategy'"**
- Check that your inline code defines a `strategy` function

**Strategy not being used**
- Check that `strategy_file` or `strategy_code` is set in CONFIG
- Verify the file path is correct
- Check for syntax errors in your strategy code

**No trades being generated**
- Your strategy might be returning 'HOLD' too often
- Try lowering your confidence thresholds
- Check that your conditions are being met

## See Also

- `example_strategy.py` - Example strategy file with multiple strategies
- `CONFIG_GUIDE.md` - Configuration options
- `README.md` - General app documentation

