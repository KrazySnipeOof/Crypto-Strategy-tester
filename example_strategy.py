"""
Example Custom Strategy for Crypto Quant Liquidity Simulator

This file demonstrates how to create a custom trading strategy.
The strategy function will be called for each bar/trade in the historical data.

Strategy Function Signature:
    def strategy(row, context):
        Parameters:
            row: pd.Series with columns:
                - entry_price, exit_price, high, low, volume
                - price_range, vsa_ratio, cvd, cvd_ma
                - liq_level_below, liq_level_above
                - recent_high, recent_low
                - symbol, entry_time
            context: dict with:
                - equity: current account equity
                - trades: list of previous trades
                - current_date: current trading date
        
        Returns:
            tuple: (signal_type, confidence)
                - signal_type: 'BUY', 'SELL', or 'HOLD'
                - confidence: float between 0.0 and 1.0
"""

def strategy(row, context=None):
    """
    Example strategy: Buy when CVD is above its moving average and price is near liquidity level.
    
    This is just an example - modify this function to implement your own strategy logic.
    """
    # Get data from row
    entry_price = row['entry_price']
    cvd = row.get('cvd', 0)
    cvd_ma = row.get('cvd_ma', 0)
    liq_level_below = row.get('liq_level_below', entry_price * 0.98)
    liq_level_above = row.get('liq_level_above', entry_price * 1.02)
    volume = row.get('volume', 0)
    vsa_ratio = row.get('vsa_ratio', 0)
    
    # Strategy logic
    # 1. Check if CVD is bullish (above MA)
    cvd_bullish = cvd > cvd_ma
    
    # 2. Check if price is near liquidity level (within 0.5% of lower liquidity)
    near_liquidity = abs(entry_price - liq_level_below) / entry_price < 0.005
    
    # 3. Check volume strength
    volume_strong = vsa_ratio > 300
    
    # Combine signals
    buy_signals = 0
    if cvd_bullish:
        buy_signals += 1
    if near_liquidity:
        buy_signals += 1
    if volume_strong:
        buy_signals += 1
    
    # Generate signal
    if buy_signals >= 2:
        confidence = min(0.5 + (buy_signals * 0.15), 1.0)
        return 'BUY', confidence
    elif cvd < cvd_ma * 0.9:  # Strong bearish CVD
        return 'SELL', 0.6
    else:
        return 'HOLD', 0.5

# Alternative simple strategy examples:

def simple_cvd_strategy(row, context=None):
    """Simple strategy: Buy when CVD > CVD MA"""
    cvd = row.get('cvd', 0)
    cvd_ma = row.get('cvd_ma', 0)
    
    if cvd > cvd_ma:
        return 'BUY', 0.8
    return 'HOLD', 0.5

def volume_breakout_strategy(row, context=None):
    """Buy on high volume breakouts"""
    vsa_ratio = row.get('vsa_ratio', 0)
    entry_price = row['entry_price']
    recent_high = row.get('recent_high', entry_price)
    
    if vsa_ratio > 500 and entry_price >= recent_high * 0.99:
        return 'BUY', 0.9
    return 'HOLD', 0.5

def liquidity_hunt_strategy(row, context=None):
    """Buy when price approaches liquidity levels"""
    entry_price = row['entry_price']
    liq_level_below = row.get('liq_level_below', entry_price * 0.98)
    distance_to_liq = abs(entry_price - liq_level_below) / entry_price
    
    if distance_to_liq < 0.003:  # Within 0.3% of liquidity
        return 'BUY', 0.85
    return 'HOLD', 0.5

