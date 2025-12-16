# crypto_quant_liquidity_simulator.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import warnings
import os
import glob
from pathlib import Path

warnings.filterwarnings('ignore')

# ==========================
# CONFIG / CONSTANTS
# ==========================

CHICAGO_TZ = pytz.timezone('America/Chicago')
SYMBOLS = ["BTC", "ETH", "XRP", "SOL"]

BASE_PRICES = {
    "BTC": 90000,
    "ETH": 3500,
    "XRP": 2.5,
    "SOL": 180
}

# ==========================
# TRADE GENERATION (SYNTHETIC)
# ==========================

def generate_realistic_cvd(length=50, trend=0.1):
    """Generate synthetic CVD series (random walk with slight trend)."""
    cvd = np.zeros(length)
    cvd[0] = np.random.normal(0, 1000)
    for i in range(1, length):
        change = np.random.normal(trend * 100, 500)
        cvd[i] = cvd[i-1] + change
    return cvd

def generate_synthetic_trades_multi_with_signals(
    num_trades=40,
    win_rate=0.55,
    avg_win=300,
    avg_loss=100,
    trading_days=6,
    start_date=None
):
    """
    Generate synthetic trades with OHLCV, CVD, and liquidity levels.
    - 6 (or 12) trading days, Mon–Sat
    - Multi‑asset: BTC, ETH, XRP, SOL
    - Only between 7:30–11:00 Chicago time
    """
    if start_date is None:
        today = datetime.utcnow().date()
        start_date = today - timedelta(days=today.weekday())  # last Monday

    trades = []
    trades_per_day = max(1, num_trades // trading_days)
    trade_count = 0
    current_prices = BASE_PRICES.copy()

    for day in range(trading_days):
        trade_date = start_date + timedelta(days=day)

        # daily CVD pattern
        daily_cvd = generate_realistic_cvd(trades_per_day, trend=np.random.choice([-0.05, 0.05, 0.1]))
        daily_cvd_ma = np.mean(daily_cvd)

        for trade_idx in range(trades_per_day):
            if trade_count >= num_trades:
                break

            symbol = np.random.choice(SYMBOLS)
            entry_price = current_prices[symbol]

            # time in 7:30–11:00 Chicago
            hour = np.random.randint(7, 11)  # 7,8,9,10
            if hour == 7:
                minute = np.random.randint(30, 60)  # 7:30–7:59
            else:
                minute = np.random.randint(0, 60)   # full hour

            entry_time_chi = datetime.combine(
                trade_date,
                datetime.min.time().replace(hour=hour, minute=minute)
            )
            entry_time_chi = CHICAGO_TZ.localize(entry_time_chi)
            entry_time = entry_time_chi.astimezone(pytz.UTC)

            # pseudo‑OHLCV
            price_range = entry_price * np.random.uniform(0.001, 0.005)
            high = entry_price + price_range
            low = entry_price - price_range * 0.5
            volume = np.random.uniform(1000, 5000)

            # win/loss outcome
            is_win = np.random.rand() < win_rate
            if is_win:
                pnl_usd = np.random.normal(avg_win, avg_win * 0.15)
            else:
                pnl_usd = -np.random.normal(avg_loss, avg_loss * 0.15)

            exit_price = entry_price + (pnl_usd / 4800) * entry_price

            # VSA proxy
            vsa_ratio = volume / max(price_range, 0.01)

            # CVD
            cvd = daily_cvd[trade_idx]

            # Simulated liquidation levels
            liq_below = entry_price * (1 - np.random.uniform(0.02, 0.03))
            liq_above = entry_price * (1 + np.random.uniform(0.02, 0.03))

            trades.append({
                'symbol': symbol,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'entry_time': entry_time,
                'high': high,
                'low': low,
                'volume': volume,
                'price_range': price_range,
                'vsa_ratio': vsa_ratio,
                'cvd': cvd,
                'cvd_ma': daily_cvd_ma,
                'liq_level_below': liq_below,
                'liq_level_above': liq_above,
                'recent_high': entry_price * 1.02,
                'recent_low': entry_price * 0.98,
                'side': 'LONG'
            })

            current_prices[symbol] = exit_price
            trade_count += 1

        if trade_count >= num_trades:
            break

    return pd.DataFrame(trades)

# ==========================
# CSV DATA IMPORT
# ==========================

def parse_historical_data_file(file_path):
    """
    Parse historical BTC data files in format: YYYYMMDD HHMMSS;Open;High;Low;Close;Volume
    
    Parameters:
    -----------
    file_path : str
        Path to the historical data file (.txt or .csv)
        
    Returns:
    --------
    pd.DataFrame : DataFrame with columns: timestamp, open, high, low, close, volume
    """
    try:
        # Read file - handle both semicolon and comma separated
        df = pd.read_csv(file_path, sep=';', header=None, 
                        names=['datetime_str', 'open', 'high', 'low', 'close', 'volume'],
                        dtype={'open': float, 'high': float, 'low': float, 'close': float, 'volume': float})
        
        # Parse datetime: YYYYMMDD HHMMSS
        df['timestamp'] = pd.to_datetime(df['datetime_str'], format='%Y%m%d %H%M%S', errors='coerce')
        
        # Drop rows with invalid timestamps
        df = df.dropna(subset=['timestamp'])
        
        # Set timezone to UTC (assuming data is in UTC)
        if df['timestamp'].dt.tz is None:
            df['timestamp'] = df['timestamp'].dt.tz_localize(pytz.UTC)
        else:
            df['timestamp'] = df['timestamp'].dt.tz_convert(pytz.UTC)
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    
    except Exception as e:
        # Try comma-separated format as fallback
        try:
            df = pd.read_csv(file_path, sep=',', header=None,
                            names=['datetime_str', 'open', 'high', 'low', 'close', 'volume'],
                            dtype={'open': float, 'high': float, 'low': float, 'close': float, 'volume': float})
            df['timestamp'] = pd.to_datetime(df['datetime_str'], format='%Y%m%d %H%M%S', errors='coerce')
            df = df.dropna(subset=['timestamp'])
            if df['timestamp'].dt.tz is None:
                df['timestamp'] = df['timestamp'].dt.tz_localize(pytz.UTC)
            else:
                df['timestamp'] = df['timestamp'].dt.tz_convert(pytz.UTC)
            df = df.sort_values('timestamp').reset_index(drop=True)
            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        except Exception as e2:
            raise ValueError(f"Error parsing historical data file {file_path}: {str(e2)}")

def convert_historical_to_trades(historical_df, symbol='BTC', 
                                  trading_window_start=(7, 30),
                                  trading_window_end=(11, 0),
                                  min_price_move=0.0005,
                                  lookback_periods=20,
                                  max_trades=None,
                                  sample_interval=1):
    """
    Convert historical OHLCV minute data into trade format.
    
    Parameters:
    -----------
    historical_df : pd.DataFrame
        DataFrame with columns: timestamp, open, high, low, close, volume
    symbol : str
        Trading symbol
    trading_window_start : tuple
        (hour, minute) for start of trading window
    trading_window_end : tuple
        (hour, minute) for end of trading window
    min_price_move : float
        Minimum price movement to consider a trade
    lookback_periods : int
        Number of periods to look back for recent high/low
        
    Returns:
    --------
    pd.DataFrame : Trades DataFrame in expected format
    """
    trades = []
    
    # Filter to trading window (7:30-11:00 AM Chicago time)
    historical_df['chicago_time'] = historical_df['timestamp'].dt.tz_convert(CHICAGO_TZ)
    historical_df['hour'] = historical_df['chicago_time'].dt.hour
    historical_df['minute'] = historical_df['chicago_time'].dt.minute
    
    start_mins = trading_window_start[0] * 60 + trading_window_start[1]
    end_mins = trading_window_end[0] * 60 + trading_window_end[1]
    historical_df['time_mins'] = historical_df['hour'] * 60 + historical_df['minute']
    
    # Filter to trading window
    window_mask = (historical_df['time_mins'] >= start_mins) & (historical_df['time_mins'] <= end_mins)
    window_data = historical_df[window_mask].copy()
    
    if len(window_data) == 0:
        raise ValueError("No data found in trading window (7:30-11:00 AM Chicago time)")
    
    # Calculate rolling statistics
    window_data['price_range'] = window_data['high'] - window_data['low']
    window_data['price_change'] = window_data['close'].pct_change()
    window_data['volume_ma'] = window_data['volume'].rolling(window=lookback_periods, min_periods=1).mean()
    
    # Calculate CVD (simplified: cumulative volume delta based on price direction)
    window_data['volume_delta'] = window_data.apply(
        lambda row: row['volume'] if row['close'] > row['open'] else -row['volume'], axis=1
    )
    window_data['cvd'] = window_data['volume_delta'].cumsum()
    window_data['cvd_ma'] = window_data['cvd'].rolling(window=lookback_periods, min_periods=1).mean()
    
    # Calculate recent high/low
    window_data['recent_high'] = window_data['high'].rolling(window=lookback_periods, min_periods=1).max()
    window_data['recent_low'] = window_data['low'].rolling(window=lookback_periods, min_periods=1).min()
    
    # Generate trades from price movements
    for i in range(1, len(window_data)):
        # Sample every Nth bar if sample_interval > 1
        if i % sample_interval != 0:
            continue
            
        current = window_data.iloc[i]
        previous = window_data.iloc[i-1]
        
        # Entry: use current bar's open as entry, close as potential exit
        entry_price = current['open']
        exit_price = current['close']
        
        # Only create trade if there's meaningful price movement
        price_move_pct = abs((exit_price - entry_price) / entry_price)
        if price_move_pct < min_price_move:
            continue
        
        # Limit number of trades if max_trades is specified
        if max_trades and len(trades) >= max_trades:
            break
        
        # Calculate derived metrics
        price_range = current['price_range']
        volume = current['volume']
        vsa_ratio = volume / max(price_range, 0.01) if price_range > 0 else 0
        
        # Liquidity levels (simplified: based on recent volatility)
        volatility = window_data['price_range'].rolling(window=lookback_periods, min_periods=1).mean().iloc[i]
        liq_below = entry_price - (volatility * 2)
        liq_above = entry_price + (volatility * 2)
        
        trades.append({
            'symbol': symbol,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'entry_time': current['timestamp'],
            'high': current['high'],
            'low': current['low'],
            'volume': volume,
            'price_range': price_range,
            'vsa_ratio': vsa_ratio,
            'cvd': current['cvd'],
            'cvd_ma': current['cvd_ma'],
            'liq_level_below': liq_below,
            'liq_level_above': liq_above,
            'recent_high': current['recent_high'],
            'recent_low': current['recent_low'],
            'side': 'LONG'  # Default to LONG, can be enhanced
        })
    
    return pd.DataFrame(trades)

def load_csv_files(csv_paths=None, csv_directory=None):
    """
    Load multiple CSV files and combine them into a single DataFrame.
    
    Parameters:
    -----------
    csv_paths : list of str, optional
        List of file paths to CSV files
    csv_directory : str, optional
        Directory path containing CSV files (will load all .csv files)
    
    Returns:
    --------
    pd.DataFrame : Combined DataFrame from all CSV files
    """
    all_dataframes = []
    
    # Load from directory if specified
    if csv_directory:
        csv_files = glob.glob(os.path.join(csv_directory, "*.csv"))
        if not csv_files:
            raise ValueError(f"No CSV files found in directory: {csv_directory}")
        csv_paths = csv_files
    
    # Load from file paths if specified
    if csv_paths:
        for csv_path in csv_paths:
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"CSV file not found: {csv_path}")
            
            try:
                df = pd.read_csv(csv_path)
                # Add source file info
                df['source_file'] = os.path.basename(csv_path)
                all_dataframes.append(df)
            except Exception as e:
                raise ValueError(f"Error reading {csv_path}: {str(e)}")
    
    if not all_dataframes:
        raise ValueError("No CSV files provided or found")
    
    # Combine all DataFrames
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    return combined_df

def process_csv_to_trades_format(df, 
                                  symbol_col='symbol',
                                  entry_price_col='entry_price',
                                  exit_price_col='exit_price',
                                  entry_time_col='entry_time',
                                  high_col='high',
                                  low_col='low',
                                  volume_col='volume',
                                  cvd_col='cvd',
                                  cvd_ma_col='cvd_ma',
                                  liq_below_col='liq_level_below',
                                  liq_above_col='liq_level_above',
                                  recent_high_col='recent_high',
                                  recent_low_col='recent_low',
                                  side_col='side'):
    """
    Process CSV DataFrame to match the expected trades format.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame from CSV
    *_col : str
        Column names in the CSV that map to expected fields
        
    Returns:
    --------
    pd.DataFrame : Processed DataFrame in trades format
    """
    required_cols = {
        'symbol': symbol_col,
        'entry_price': entry_price_col,
        'exit_price': exit_price_col,
        'entry_time': entry_time_col
    }
    
    # Check required columns exist
    missing_cols = [col for col in required_cols.values() if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Create processed DataFrame
    processed = pd.DataFrame()
    
    # Map required columns
    processed['symbol'] = df[symbol_col]
    processed['entry_price'] = pd.to_numeric(df[entry_price_col], errors='coerce')
    processed['exit_price'] = pd.to_numeric(df[exit_price_col], errors='coerce')
    
    # Parse entry_time
    if entry_time_col in df.columns:
        processed['entry_time'] = pd.to_datetime(df[entry_time_col], errors='coerce')
        # Convert to UTC if timezone-aware, otherwise assume UTC
        if processed['entry_time'].dt.tz is not None:
            processed['entry_time'] = processed['entry_time'].dt.tz_convert(pytz.UTC)
        else:
            processed['entry_time'] = processed['entry_time'].dt.tz_localize(pytz.UTC)
    
    # Map optional columns with defaults
    optional_mappings = {
        'high': (high_col, lambda x: x['entry_price'] * 1.002),
        'low': (low_col, lambda x: x['entry_price'] * 0.998),
        'volume': (volume_col, 2000.0),
        'price_range': (None, lambda x: abs(x['high'] - x['low'])),
        'vsa_ratio': (None, lambda x: x['volume'] / max(x['price_range'], 0.01)),
        'cvd': (cvd_col, 0.0),
        'cvd_ma': (cvd_ma_col, lambda x: x['cvd']),
        'liq_level_below': (liq_below_col, lambda x: x['entry_price'] * 0.98),
        'liq_level_above': (liq_above_col, lambda x: x['entry_price'] * 1.02),
        'recent_high': (recent_high_col, lambda x: x['entry_price'] * 1.02),
        'recent_low': (recent_low_col, lambda x: x['entry_price'] * 0.98),
        'side': (side_col, 'LONG')
    }
    
    for target_col, (source_col, default) in optional_mappings.items():
        if source_col and source_col in df.columns:
            if target_col in ['cvd', 'cvd_ma', 'volume']:
                processed[target_col] = pd.to_numeric(df[source_col], errors='coerce').fillna(default if isinstance(default, (int, float)) else 0)
            elif target_col == 'side':
                processed[target_col] = df[source_col].fillna(default)
            else:
                processed[target_col] = pd.to_numeric(df[source_col], errors='coerce')
        else:
            if callable(default):
                processed[target_col] = default(processed)
            else:
                processed[target_col] = default
    
    # Calculate derived fields if not present
    if 'price_range' not in processed.columns or processed['price_range'].isna().any():
        processed['price_range'] = abs(processed['high'] - processed['low'])
    
    if 'vsa_ratio' not in processed.columns or processed['vsa_ratio'].isna().any():
        processed['vsa_ratio'] = processed['volume'] / processed['price_range'].replace(0, 0.01)
    
    # Filter out rows with missing critical data
    processed = processed.dropna(subset=['entry_price', 'exit_price', 'entry_time', 'symbol'])
    
    # Sort by entry_time
    processed = processed.sort_values('entry_time').reset_index(drop=True)
    
    return processed

# ==========================
# CUSTOM STRATEGY INTERFACE
# ==========================

def default_strategy(row, context=None):
    """
    Default built-in strategy (Price Action + VSA + CVD + Liquidity).
    This is used if no custom strategy is provided.
    
    Parameters:
    -----------
    row : pd.Series
        Current bar/trade data with columns: entry_price, exit_price, high, low, 
        volume, price_range, vsa_ratio, cvd, cvd_ma, liq_level_below, 
        liq_level_above, recent_high, recent_low, symbol, entry_time
    context : dict, optional
        Additional context (e.g., account equity, previous trades)
        
    Returns:
    --------
    tuple: (signal_type, confidence)
        - signal_type: 'BUY', 'SELL', or 'HOLD'
        - confidence: float between 0 and 1
    """
    entry_price = row['entry_price']
    recent_low = row.get('recent_low', entry_price * 0.98)
    recent_high = row.get('recent_high', entry_price * 1.02)
    
    # Price action
    if entry_price >= recent_high * 0.99:
        pa_score = 0.7
    elif entry_price >= recent_low + (recent_high - recent_low) * 0.5:
        pa_score = 0.5
    else:
        pa_score = 0.2
    
    # VSA
    volume = row.get('volume', 1000)
    price_range = row.get('price_range', 1.0)
    avg_vol = volume / max(price_range, 0.01)
    if avg_vol > 500:
        vsa_score = 0.8
    elif avg_vol > 300:
        vsa_score = 0.6
    else:
        vsa_score = 0.3
    
    # CVD
    cvd = row.get('cvd', 0)
    cvd_ma = row.get('cvd_ma', 0)
    if cvd > cvd_ma and cvd > 0:
        cvd_score = 0.85
    elif cvd > 0:
        cvd_score = 0.6
    elif cvd < cvd_ma:
        cvd_score = 0.2
    else:
        cvd_score = 0.35
    
    # Liquidity
    liq_level_below = row.get('liq_level_below', entry_price * 0.98)
    liq_level_above = row.get('liq_level_above', entry_price * 1.02)
    if liq_level_below < entry_price < liq_level_above:
        distance_to_upper = liq_level_above - entry_price
        distance_to_lower = entry_price - liq_level_below
        if distance_to_lower < distance_to_upper:
            liq_score = 0.8
        else:
            liq_score = 0.5
    else:
        liq_score = 0.3
    
    # Combined score
    weights = {'pa': 0.25, 'vsa': 0.25, 'cvd': 0.25, 'liq': 0.25}
    score = pa_score * weights['pa'] + vsa_score * weights['vsa'] + cvd_score * weights['cvd'] + liq_score * weights['liq']
    
    if score > 0.55:
        return 'BUY', min(score, 1.0)
    elif score < 0.45:
        return 'SELL', min(1.0 - score, 1.0)
    else:
        return 'HOLD', 0.5

def load_strategy_from_file(file_path):
    """
    Load a custom strategy function from a Python file.
    
    The file should define a function called 'strategy' that takes (row, context) 
    and returns (signal_type, confidence).
    
    Parameters:
    -----------
    file_path : str
        Path to Python file containing strategy function
        
    Returns:
    --------
    function : Strategy function
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("custom_strategy", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    if not hasattr(module, 'strategy'):
        raise ValueError(f"Strategy file {file_path} must define a function called 'strategy'")
    
    return module.strategy

def load_strategy_from_code(strategy_code):
    """
    Load a custom strategy function from code string.
    
    Parameters:
    -----------
    strategy_code : str
        Python code string that defines a 'strategy' function
        
    Returns:
    --------
    function : Strategy function
    """
    namespace = {}
    exec(strategy_code, namespace)
    
    if 'strategy' not in namespace:
        raise ValueError("Strategy code must define a function called 'strategy'")
    
    return namespace['strategy']

def load_custom_strategy(config):
    """
    Load custom strategy from config (file or code).
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary
        
    Returns:
    --------
    function or None : Strategy function, or None to use default
    """
    if config.get("strategy_file") and os.path.exists(config["strategy_file"]):
        try:
            return load_strategy_from_file(config["strategy_file"])
        except Exception as e:
            st.error(f"Error loading strategy from file: {str(e)}")
            return None
    
    if config.get("strategy_code"):
        try:
            return load_strategy_from_code(config["strategy_code"])
        except Exception as e:
            st.error(f"Error loading strategy from code: {str(e)}")
            return None
    
    return None

# ==========================
# ENHANCED BACKTESTER
# ==========================

class EnhancedSignalBacktester:
    """
    - Trading window: 7:30–11:00 AM Chicago time
    - Entry signals: Price action + VSA + CVD + Liquidity levels
    - Multi‑asset: BTC, ETH, XRP, SOL
    - Funded rules: 2% daily DD, 3% max DD, $500 target on $5k
    """

    def __init__(self, account_size=5000, daily_dd_limit=0.02, max_dd_limit=0.03,
                 entry_size=4800, stop_loss=100, take_profit=300, target_gain=500,
                 custom_strategy=None):
        self.account_size = account_size
        self.daily_dd_limit = daily_dd_limit
        self.max_dd_limit = max_dd_limit
        self.entry_size = entry_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.target_gain = target_gain
        self.custom_strategy = custom_strategy  # Custom strategy function

        self.equity_curve = []
        self.trade_log = []
        self.daily_equity = []
        self.signal_log = []
        self.violated_rules = False
        self.violation_reason = ""
        self.context = {'equity': account_size, 'trades': []}  # Context for strategy

    # ---------- Helper: trading window ----------

    def _is_trading_window(self, entry_time):
        """Check if time is between 7:30–11:00 AM Chicago."""
        if entry_time.tzinfo is None:
            entry_time = entry_time.replace(tzinfo=pytz.UTC)
        chicago_time = entry_time.astimezone(CHICAGO_TZ)
        hour = chicago_time.hour
        minute = chicago_time.minute

        start_mins = 7 * 60 + 30  # 7:30
        end_mins = 11 * 60        # 11:00
        current_mins = hour * 60 + minute
        return start_mins <= current_mins <= end_mins

    # ---------- Signal components ----------

    def calculate_price_action_signal(self, row):
        entry_price = row['entry_price']
        recent_low = row.get('recent_low', entry_price * 0.98)
        recent_high = row.get('recent_high', entry_price * 1.02)

        if entry_price >= recent_high * 0.99:
            return 0.7  # breakout / strong
        elif entry_price >= recent_low + (recent_high - recent_low) * 0.5:
            return 0.5
        else:
            return 0.2

    def calculate_vsa_signal(self, row):
        volume = row.get('volume', 1000)
        price_range = row.get('price_range', 1.0)
        avg_vol = volume / max(price_range, 0.01)

        if avg_vol > 500:
            return 0.8
        elif avg_vol > 300:
            return 0.6
        else:
            return 0.3

    def calculate_cvd_signal(self, row):
        cvd = row.get('cvd', 0)
        cvd_ma = row.get('cvd_ma', 0)

        if cvd > cvd_ma and cvd > 0:
            return 0.85
        elif cvd > 0:
            return 0.6
        elif cvd < cvd_ma:
            return 0.2
        else:
            return 0.35

    def calculate_liquidity_signal(self, row):
        entry_price = row['entry_price']
        liq_level_below = row.get('liq_level_below', entry_price * 0.98)
        liq_level_above = row.get('liq_level_above', entry_price * 1.02)

        if liq_level_below < entry_price < liq_level_above:
            distance_to_upper = liq_level_above - entry_price
            distance_to_lower = entry_price - liq_level_below
            if distance_to_lower < distance_to_upper:
                return 0.8
            else:
                return 0.5
        else:
            return 0.3

    def generate_entry_signal(self, row):
        pa = self.calculate_price_action_signal(row)
        vsa = self.calculate_vsa_signal(row)
        cvd = self.calculate_cvd_signal(row)
        liq = self.calculate_liquidity_signal(row)

        weights = {'pa': 0.25, 'vsa': 0.25, 'cvd': 0.25, 'liq': 0.25}
        score = pa * weights['pa'] + vsa * weights['vsa'] + cvd * weights['cvd'] + liq * weights['liq']

        if score > 0.55:
            return score, 'BUY', min(score, 1.0)
        elif score < 0.45:
            return score, 'SELL', min(1.0 - score, 1.0)
        else:
            return score, 'HOLD', 0.5

    # ---------- Main simulation ----------

    def run_simulation(self, trades_df):
        self.equity_curve = [self.account_size]
        self.trade_log = []
        self.daily_equity = []
        self.signal_log = []
        self.violated_rules = False
        self.violation_reason = ""

        current_equity = self.account_size
        peak_equity = self.account_size
        daily_peak = self.account_size
        current_date = None

        for _, trade in trades_df.iterrows():
            entry_time = trade.get('entry_time', datetime.utcnow())
            if isinstance(entry_time, str):
                entry_time = pd.to_datetime(entry_time)
            if entry_time.tzinfo is None:
                entry_time = entry_time.replace(tzinfo=pytz.UTC)

            # trading window check
            if not self._is_trading_window(entry_time):
                self.signal_log.append({
                    'entry_time': entry_time,
                    'symbol': trade.get('symbol', 'UNK'),
                    'signal_type': 'SKIPPED',
                    'reason': 'Outside trading window',
                    'signal_strength': 0
                })
                continue

            # Update context
            self.context['equity'] = current_equity
            self.context['current_date'] = trade_date
            
            # Generate signal using custom strategy or default
            if self.custom_strategy:
                try:
                    sig_type, conf = self.custom_strategy(trade, self.context)
                    score = conf  # Use confidence as score
                except Exception as e:
                    # Fallback to default if custom strategy fails
                    score, sig_type, conf = self.generate_entry_signal(trade)
                    self.signal_log.append({
                        'entry_time': entry_time,
                        'symbol': trade.get('symbol', 'UNK'),
                        'signal_type': 'ERROR',
                        'signal_strength': 0,
                        'confidence': 0,
                        'error': str(e)
                    })
            else:
                score, sig_type, conf = self.generate_entry_signal(trade)
            
            self.signal_log.append({
                'entry_time': entry_time,
                'symbol': trade.get('symbol', 'UNK'),
                'signal_type': sig_type,
                'signal_strength': score,
                'confidence': conf
            })

            if sig_type != 'BUY':
                continue

            entry_price = trade['entry_price']
            exit_price = trade['exit_price']
            side = trade.get('side', 'LONG')

            if side == 'LONG':
                pnl = self.entry_size * ((exit_price - entry_price) / entry_price)
            else:
                pnl = self.entry_size * ((entry_price - exit_price) / entry_price)

            new_equity = current_equity + pnl
            trade_date = entry_time.date()

            if current_date is None:
                current_date = trade_date
                daily_peak = current_equity
            elif trade_date != current_date:
                self.daily_equity.append({
                    'date': current_date,
                    'equity': current_equity,
                    'daily_dd_pct': (daily_peak - current_equity) / daily_peak if daily_peak > 0 else 0
                })
                current_date = trade_date
                daily_peak = current_equity

            # daily DD
            if daily_peak > 0:
                daily_dd = (daily_peak - new_equity) / daily_peak
                if daily_dd > self.daily_dd_limit:
                    self.violated_rules = True
                    self.violation_reason = f"Daily DD exceeded on {trade_date}: {daily_dd*100:.2f}%"
                    return self._finalize_stats(current_equity, peak_equity, True)

            # max DD
            if peak_equity > 0:
                max_dd = (peak_equity - new_equity) / peak_equity
                if max_dd > self.max_dd_limit:
                    self.violated_rules = True
                    self.violation_reason = f"Max DD exceeded: {max_dd*100:.2f}%"
                    return self._finalize_stats(current_equity, peak_equity, True)

            current_equity = new_equity
            if current_equity > peak_equity:
                peak_equity = current_equity
            if current_equity > daily_peak:
                daily_peak = current_equity

            trade_record = {
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': pnl,
                'equity': current_equity,
                'date': entry_time,
                'symbol': trade.get('symbol', 'UNK'),
                'signal_strength': score
            }
            self.trade_log.append(trade_record)
            self.context['trades'].append(trade_record)

            self.equity_curve.append(current_equity)

            if current_equity - self.account_size >= self.target_gain:
                return self._finalize_stats(current_equity, peak_equity, False, passed=True)

        return self._finalize_stats(current_equity, peak_equity, False)

    # ---------- Final stats / risk metrics ----------

    def _finalize_stats(self, final_equity, peak_equity, violated, passed=False):
        equity = np.array(self.equity_curve)
        total_pnl = final_equity - self.account_size
        return_pct = (total_pnl / self.account_size) * 100

        rolling_max = np.maximum.accumulate(equity)
        drawdowns = (rolling_max - equity) / rolling_max
        max_dd = drawdowns.max() * 100 if len(drawdowns) > 0 else 0

        if len(self.daily_equity) > 1:
            deq = pd.DataFrame(self.daily_equity).sort_values('date')
            deq['ret'] = deq['equity'].pct_change().fillna(0)
            daily_returns = deq['ret'].values
        else:
            daily_returns = np.array([0.0])

        ann_factor = 312
        avg_daily_ret = daily_returns.mean()
        std_daily_ret = daily_returns.std(ddof=1) if len(daily_returns) > 1 else 0

        sharpe = (avg_daily_ret * ann_factor) / (std_daily_ret * np.sqrt(ann_factor)) if std_daily_ret > 0 else 0

        downside = daily_returns[daily_returns < 0]
        downside_std = downside.std(ddof=1) if len(downside) > 0 else 0
        sortino = (avg_daily_ret * ann_factor) / (downside_std * np.sqrt(ann_factor)) if downside_std > 0 else 0

        ann_return = ((1 + avg_daily_ret) ** ann_factor) - 1
        calmar = ann_return / (max_dd / 100) if max_dd > 0 else 0

        gross_profit = sum(max(t['pnl'], 0) for t in self.trade_log)
        gross_loss = -sum(min(t['pnl'], 0) for t in self.trade_log)
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else np.inf

        num_trades = len(self.trade_log)
        winning_trades = len([t for t in self.trade_log if t['pnl'] > 0])
        win_rate = (winning_trades / num_trades * 100) if num_trades > 0 else 0
        avg_trade = (total_pnl / num_trades) if num_trades > 0 else 0

        daily_dds = [d['daily_dd_pct'] for d in self.daily_equity]
        avg_daily_dd = np.mean(daily_dds) * 100 if daily_dds else 0
        max_daily_dd = np.max(daily_dds) * 100 if daily_dds else 0

        return {
            'final_equity': final_equity,
            'total_pnl': total_pnl,
            'return_pct': return_pct,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'avg_trade_pnl': avg_trade,
            'max_dd_pct': max_dd,
            'avg_daily_dd_pct': avg_daily_dd,
            'max_daily_dd_pct': max_daily_dd,
            'violated_rules': violated,
            'violation_reason': self.violation_reason,
            'passed_account': passed or (total_pnl >= self.target_gain and not violated),
            'equity_curve': self.equity_curve,
            'trade_log': self.trade_log,
            'signal_log': self.signal_log,
            'sharpe': sharpe,
            'sortino': sortino,
            'calmar': calmar,
            'profit_factor': profit_factor,
            'ann_return_pct': ann_return * 100
        }

# ==========================
# STREAMLIT APP
# ==========================

def main():
    # ==========================
    # CONFIGURATION (Edit these values directly in code)
    # ==========================
    CONFIG = {
        # Data Source: "synthetic", "csv", or "historical"
        "data_source": "historical",
        
        # Historical Data Settings (only used if data_source == "historical")
        "historical_symbols": ["BTC", "ETH"],  # List of symbols to load: ["BTC"], ["ETH"], or ["BTC", "ETH"]
        "historical_directories": {
            "BTC": "BTCUSD DATA",  # Path to BTC data directory
            "ETH": "ETHUSD DATA"   # Path to ETH data directory
        },
        "sample_interval": 5,  # Use every Nth minute (1 = all bars, 5 = every 5 minutes)
        "max_trades": 0,  # Max trades to load (0 = no limit)
        
        # CSV Settings (only used if data_source == "csv")
        "csv_directory": "",  # Path to CSV files directory (empty = use file upload)
        "csv_file_paths": [],  # List of specific CSV file paths (empty = use directory)
        
        # Account Parameters
        "account_size": 5000,
        "target_gain": 500,
        "daily_dd_limit": 0.02,  # 2%
        "max_dd_limit": 0.03,  # 3%
        
        # Position Parameters
        "entry_size": 4800,
        "stop_loss": 100,
        "take_profit": 300,
        
        # Strategy Parameters (only used if data_source == "synthetic")
        "trading_days": 6,
        "num_trades": 40,
        "win_rate": 0.55,
        "avg_win": 300,
        "avg_loss": 100,
        
        # Monte Carlo Settings
        "num_simulations": 500,
        
        # Custom Strategy Settings
        "strategy_file": "",  # Path to Python file with custom strategy function (empty = use default)
        "strategy_code": None,  # Inline strategy code as string (alternative to strategy_file)
        # Example strategy_code:
        # """
        # def strategy(row, context):
        #     # Your strategy logic here
        #     # Return ('BUY', confidence) or ('SELL', confidence) or ('HOLD', confidence)
        #     if row['cvd'] > row['cvd_ma']:
        #         return 'BUY', 0.8
        #     return 'HOLD', 0.5
        # """
        
        # UI Settings
        "use_ui_overrides": True,  # Set to False to disable UI controls and use config only
    }
    # ==========================
    
    st.set_page_config(layout="wide", page_title="Crypto Quant Liquidity Simulator")

    st.title("Crypto Quant Liquidity Simulator")
    st.markdown("""
    **Funded account simulator for BTC, ETH, XRP, SOL**  
    - Trading window: **7:30–11:00 AM Chicago time**  
    - Entries: **Price Action + VSA + CVD + Liquidity/Liquidation levels**  
    - Rules: **2% daily DD, 3% max DD, $500 target on $5k**  
    """)
    
    # Load custom strategy
    custom_strategy = None
    if CONFIG.get("strategy_file") or CONFIG.get("strategy_code"):
        custom_strategy = load_custom_strategy(CONFIG)
        if custom_strategy:
            st.success("✓ Custom strategy loaded from config")
    
    # Show config status if not using UI overrides
    if not CONFIG["use_ui_overrides"]:
        strategy_info = "Custom" if custom_strategy else "Default"
        st.info(f"⚙️ **Configuration Mode**: Using settings from code (CONFIG dict). Data source: {CONFIG['data_source']}, Symbols: {CONFIG.get('historical_symbols', 'N/A')}, Strategy: {strategy_info}")

    # Sidebar - Use config values, allow UI overrides if enabled
    if CONFIG["use_ui_overrides"]:
        st.sidebar.header("⚙️ Account Parameters")
        account_size = st.sidebar.number_input("Account Size ($)", value=CONFIG["account_size"], min_value=1000, step=100)
        target_gain = st.sidebar.number_input("Target Gain ($)", value=CONFIG["target_gain"], min_value=50, step=50)
        daily_dd_limit = st.sidebar.slider("Daily DD Limit (%)", 1.0, 5.0, CONFIG["daily_dd_limit"]*100, step=0.1) / 100
        max_dd_limit = st.sidebar.slider("Max DD Limit (%)", 1.0, 5.0, CONFIG["max_dd_limit"]*100, step=0.1) / 100

        st.sidebar.header("⚙️ Position Parameters")
        entry_size = st.sidebar.number_input("Entry Size ($)", value=CONFIG["entry_size"], min_value=100, step=100)
        stop_loss = st.sidebar.number_input("Stop Loss ($)", value=CONFIG["stop_loss"], min_value=10, step=10)
        take_profit = st.sidebar.number_input("Take Profit ($)", value=CONFIG["take_profit"], min_value=10, step=10)

        st.sidebar.header("📊 Data Source")
        data_source_map = {"synthetic": 0, "csv": 1, "historical": 2}
        data_source_idx = data_source_map.get(CONFIG["data_source"], 0)
        data_source_ui = st.sidebar.radio(
            "Choose data source:",
            ["Synthetic Data", "CSV Files", "Historical Crypto Data"],
            index=data_source_idx
        )
        # Map UI selection back to config format
        if data_source_ui == "Synthetic Data":
            data_source = "synthetic"
        elif data_source_ui == "CSV Files":
            data_source = "csv"
        else:
            data_source = "historical"
    else:
        # Use config values directly, no UI
        account_size = CONFIG["account_size"]
        target_gain = CONFIG["target_gain"]
        daily_dd_limit = CONFIG["daily_dd_limit"]
        max_dd_limit = CONFIG["max_dd_limit"]
        entry_size = CONFIG["entry_size"]
        stop_loss = CONFIG["stop_loss"]
        take_profit = CONFIG["take_profit"]
        data_source = CONFIG["data_source"]
        
        st.sidebar.info("⚙️ Using configuration from code (edit CONFIG dict to change)")
        st.sidebar.text(f"Data Source: {data_source}")
        st.sidebar.text(f"Account: ${account_size}")
        st.sidebar.text(f"Target: ${target_gain}")
    
    # Auto-load data based on config
    historical_data = None
    historical_info = None
    csv_data = None
    csv_info = None
    
    # Automatically load historical data if configured
    if data_source == "historical" and CONFIG["historical_symbols"]:
        try:
            all_trades_list = []
            total_files = 0
            
            for symbol in CONFIG["historical_symbols"]:
                hist_directory = CONFIG["historical_directories"].get(symbol, f"{symbol}USD DATA")
                
                if os.path.exists(hist_directory):
                    # Find all .txt and .csv files in directory
                    hist_files = glob.glob(os.path.join(hist_directory, "*.txt")) + \
                                glob.glob(os.path.join(hist_directory, "*.csv"))
                    
                    # Filter files by symbol
                    symbol_pattern = f"{symbol}USDT" if symbol in ["BTC", "ETH"] else symbol
                    hist_files = [f for f in hist_files if symbol_pattern.upper() in os.path.basename(f).upper()]
                    
                    if hist_files:
                        symbol_historical = []
                        for hist_file in sorted(hist_files):
                            try:
                                hist_df = parse_historical_data_file(hist_file)
                                symbol_historical.append(hist_df)
                                total_files += 1
                            except Exception as e:
                                if CONFIG["use_ui_overrides"]:
                                    st.sidebar.warning(f"Skipping {os.path.basename(hist_file)}: {str(e)}")
                        
                        if symbol_historical:
                            combined_hist = pd.concat(symbol_historical, ignore_index=True)
                            combined_hist = combined_hist.sort_values('timestamp').reset_index(drop=True)
                            
                            symbol_trades = convert_historical_to_trades(
                                combined_hist, 
                                symbol=symbol,
                                sample_interval=CONFIG["sample_interval"],
                                max_trades=CONFIG["max_trades"] if CONFIG["max_trades"] > 0 else None
                            )
                            all_trades_list.append(symbol_trades)
                            if CONFIG["use_ui_overrides"]:
                                st.sidebar.success(f"✓ Auto-loaded {len(symbol_trades)} {symbol} trades from {len(hist_files)} file(s)")
                else:
                    if CONFIG["use_ui_overrides"]:
                        st.sidebar.warning(f"{symbol} directory not found: {hist_directory}")
            
            if all_trades_list:
                historical_data = pd.concat(all_trades_list, ignore_index=True)
                historical_data = historical_data.sort_values('entry_time').reset_index(drop=True)
                historical_info = f"Auto-loaded {len(historical_data)} trades ({', '.join(CONFIG['historical_symbols'])}) from {total_files} file(s)"
                if CONFIG["use_ui_overrides"]:
                    st.sidebar.success(historical_info)
        except Exception as e:
            if CONFIG["use_ui_overrides"]:
                st.sidebar.error(f"Error auto-loading historical data: {str(e)}")
    
    # Automatically load CSV data if configured
    if data_source == "csv":
        if CONFIG["csv_file_paths"]:
            try:
                csv_data_raw = load_csv_files(csv_paths=CONFIG["csv_file_paths"])
                csv_data = process_csv_to_trades_format(csv_data_raw)
                csv_info = f"Auto-loaded {len(csv_data)} trades from {len(CONFIG['csv_file_paths'])} file(s)"
                if CONFIG["use_ui_overrides"]:
                    st.sidebar.success(csv_info)
            except Exception as e:
                if CONFIG["use_ui_overrides"]:
                    st.sidebar.error(f"Error auto-loading CSV: {str(e)}")
        elif CONFIG["csv_directory"] and os.path.exists(CONFIG["csv_directory"]):
            try:
                csv_data_raw = load_csv_files(csv_directory=CONFIG["csv_directory"])
                csv_data = process_csv_to_trades_format(csv_data_raw)
                csv_info = f"Auto-loaded {len(csv_data)} trades from directory"
                if CONFIG["use_ui_overrides"]:
                    st.sidebar.success(csv_info)
            except Exception as e:
                if CONFIG["use_ui_overrides"]:
                    st.sidebar.error(f"Error auto-loading CSV: {str(e)}")
    
    # UI-based CSV loading (only if not auto-loaded and UI overrides enabled)
    if data_source == "csv" and not csv_data and CONFIG["use_ui_overrides"]:
        st.sidebar.subheader("CSV Import Options")
        import_method = st.sidebar.radio(
            "Import method:",
            ["Upload Files", "Directory Path"],
            index=0
        )
        
        if import_method == "Upload Files":
            uploaded_files = st.sidebar.file_uploader(
                "Upload CSV files",
                type=['csv'],
                accept_multiple_files=True,
                help="Upload one or more CSV files. Required columns: symbol, entry_price, exit_price, entry_time"
            )
            
            if uploaded_files:
                try:
                    # Save uploaded files temporarily
                    temp_files = []
                    for uploaded_file in uploaded_files:
                        temp_path = os.path.join(Path.cwd(), f"temp_{uploaded_file.name}")
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        temp_files.append(temp_path)
                    
                    # Load CSV files
                    csv_data_raw = load_csv_files(csv_paths=temp_files)
                    csv_data = process_csv_to_trades_format(csv_data_raw)
                    csv_info = f"Loaded {len(csv_data)} trades from {len(uploaded_files)} file(s)"
                    
                    # Clean up temp files
                    for temp_path in temp_files:
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                    
                    st.sidebar.success(csv_info)
                except Exception as e:
                    st.sidebar.error(f"Error loading CSV: {str(e)}")
                    csv_data = None
        
        else:  # Directory Path
            csv_directory = st.sidebar.text_input(
                "CSV Directory Path",
                value="",
                help="Enter path to directory containing CSV files"
            )
            
            if csv_directory and os.path.exists(csv_directory):
                try:
                    csv_data_raw = load_csv_files(csv_directory=csv_directory)
                    csv_data = process_csv_to_trades_format(csv_data_raw)
                    csv_info = f"Loaded {len(csv_data)} trades from directory"
                    st.sidebar.success(csv_info)
                except Exception as e:
                    st.sidebar.error(f"Error loading CSV: {str(e)}")
                    csv_data = None
            elif csv_directory:
                st.sidebar.warning("Directory not found")
        
        if csv_data is not None:
            st.sidebar.info(f"**CSV Data Summary:**\n- Rows: {len(csv_data)}\n- Symbols: {csv_data['symbol'].nunique()}\n- Date Range: {csv_data['entry_time'].min()} to {csv_data['entry_time'].max()}")
    
    # UI-based Historical data loading (only if not auto-loaded and UI overrides enabled)
    if data_source == "historical" and not historical_data and CONFIG["use_ui_overrides"]:
        st.sidebar.subheader("Historical Data Options")
        
        # Symbol selection
        selected_symbols = st.sidebar.multiselect(
            "Select Symbols",
            ["BTC", "ETH"],
            default=["BTC"],
            help="Select which cryptocurrency data to load"
        )
        
        if not selected_symbols:
            st.sidebar.warning("Please select at least one symbol")
        
        # Sampling controls (always visible)
        sample_int = st.sidebar.slider("Sample Interval (minutes)", 1, 10, 5, 
                                      help="Use every Nth minute bar (1 = all bars, 5 = every 5 minutes)")
        max_trades_val = st.sidebar.number_input("Max Trades (0 = all)", value=0, min_value=0, 
                                                help="Limit number of trades (0 = no limit)")
        
        hist_import_method = st.sidebar.radio(
            "Import method:",
            ["Upload Files", "Directory Path"],
            index=1
        )
        
        if hist_import_method == "Upload Files":
            uploaded_hist_files = st.sidebar.file_uploader(
                "Upload Historical Data Files",
                type=['txt', 'csv'],
                accept_multiple_files=True,
                help="Upload BTCUSDT or ETHUSDT minute data files (format: YYYYMMDD HHMMSS;Open;High;Low;Close;Volume)"
            )
            
            if uploaded_hist_files and selected_symbols:
                try:
                    # Group files by symbol based on filename
                    symbol_files = {symbol: [] for symbol in selected_symbols}
                    
                    for uploaded_file in uploaded_hist_files:
                        filename_upper = uploaded_file.name.upper()
                        # Detect symbol from filename
                        if "BTCUSDT" in filename_upper or "BTCUSD" in filename_upper:
                            if "BTC" in selected_symbols:
                                symbol_files["BTC"].append(uploaded_file)
                        elif "ETHUSDT" in filename_upper or "ETHUSD" in filename_upper:
                            if "ETH" in selected_symbols:
                                symbol_files["ETH"].append(uploaded_file)
                        else:
                            # If can't detect, assign to first selected symbol
                            if selected_symbols:
                                symbol_files[selected_symbols[0]].append(uploaded_file)
                    
                    # Process each symbol's files
                    all_trades_list = []
                    total_files = 0
                    
                    for symbol in selected_symbols:
                        if symbol_files[symbol]:
                            symbol_historical = []
                            for uploaded_file in symbol_files[symbol]:
                                temp_path = os.path.join(Path.cwd(), f"temp_hist_{uploaded_file.name}")
                                with open(temp_path, "wb") as f:
                                    f.write(uploaded_file.getbuffer())
                                
                                # Parse historical data
                                hist_df = parse_historical_data_file(temp_path)
                                symbol_historical.append(hist_df)
                                total_files += 1
                                
                                # Clean up temp file
                                if os.path.exists(temp_path):
                                    os.remove(temp_path)
                            
                            if symbol_historical:
                                # Combine all historical data for this symbol
                                combined_hist = pd.concat(symbol_historical, ignore_index=True)
                                combined_hist = combined_hist.sort_values('timestamp').reset_index(drop=True)
                                
                                # Convert to trades format
                                symbol_trades = convert_historical_to_trades(
                                    combined_hist, 
                                    symbol=symbol,
                                    sample_interval=sample_int,
                                    max_trades=max_trades_val if max_trades_val > 0 else None
                                )
                                all_trades_list.append(symbol_trades)
                                st.sidebar.success(f"✓ Loaded {len(symbol_trades)} {symbol} trades from {len(symbol_files[symbol])} file(s)")
                    
                    # Combine all symbol trades
                    if all_trades_list:
                        historical_data = pd.concat(all_trades_list, ignore_index=True)
                        historical_data = historical_data.sort_values('entry_time').reset_index(drop=True)
                        historical_info = f"Loaded {len(historical_data)} trades ({', '.join(selected_symbols)}) from {total_files} file(s)"
                        st.sidebar.success(historical_info)
                    else:
                        st.sidebar.warning("No valid files found for selected symbols")
                        
                except Exception as e:
                    st.sidebar.error(f"Error loading historical data: {str(e)}")
                    historical_data = None
        
        else:  # Directory Path
            # Symbol to directory mapping
            symbol_dirs = {
                "BTC": "BTCUSD DATA",
                "ETH": "ETHUSD DATA"
            }
            
            # Load data for each selected symbol
            if selected_symbols:
                all_trades_list = []
                total_files = 0
                
                for symbol in selected_symbols:
                    # Get directory path for this symbol
                    default_dir = symbol_dirs.get(symbol, f"{symbol}USD DATA")
                    hist_directory = st.sidebar.text_input(
                        f"{symbol} Data Directory",
                        value=default_dir,
                        key=f"hist_dir_{symbol}",
                        help=f"Enter path to directory containing {symbol} historical data files"
                    )
                    
                    if hist_directory and os.path.exists(hist_directory):
                        try:
                            # Find all .txt and .csv files in directory
                            hist_files = glob.glob(os.path.join(hist_directory, "*.txt")) + \
                                        glob.glob(os.path.join(hist_directory, "*.csv"))
                            
                            # Filter files by symbol if possible (e.g., BTCUSDT files for BTC)
                            symbol_pattern = f"{symbol}USDT" if symbol in ["BTC", "ETH"] else symbol
                            hist_files = [f for f in hist_files if symbol_pattern.upper() in os.path.basename(f).upper()]
                            
                            if not hist_files:
                                st.sidebar.warning(f"No {symbol} data files found in {hist_directory}")
                            else:
                                # Parse all files for this symbol
                                symbol_historical = []
                                for hist_file in sorted(hist_files):
                                    try:
                                        hist_df = parse_historical_data_file(hist_file)
                                        symbol_historical.append(hist_df)
                                        total_files += 1
                                    except Exception as e:
                                        st.sidebar.warning(f"Skipping {os.path.basename(hist_file)}: {str(e)}")
                                
                                if symbol_historical:
                                    # Combine all historical data for this symbol
                                    combined_hist = pd.concat(symbol_historical, ignore_index=True)
                                    combined_hist = combined_hist.sort_values('timestamp').reset_index(drop=True)
                                    
                                    # Convert to trades format for this symbol
                                    symbol_trades = convert_historical_to_trades(
                                        combined_hist, 
                                        symbol=symbol,
                                        sample_interval=sample_int,
                                        max_trades=max_trades_val if max_trades_val > 0 else None
                                    )
                                    all_trades_list.append(symbol_trades)
                                    st.sidebar.success(f"✓ Loaded {len(symbol_trades)} {symbol} trades from {len(hist_files)} file(s)")
                        except Exception as e:
                            st.sidebar.error(f"Error loading {symbol} data: {str(e)}")
                    elif hist_directory:
                        st.sidebar.warning(f"{symbol} directory not found: {hist_directory}")
                
                # Combine all symbol trades
                if all_trades_list:
                    historical_data = pd.concat(all_trades_list, ignore_index=True)
                    historical_data = historical_data.sort_values('entry_time').reset_index(drop=True)
                    historical_info = f"Loaded {len(historical_data)} trades ({', '.join(selected_symbols)}) from {total_files} file(s)"
                    st.sidebar.success(historical_info)
        
        if historical_data is not None:
            symbols_in_data = historical_data['symbol'].unique().tolist()
            symbol_summary = ", ".join(symbols_in_data)
            st.sidebar.info(f"**Historical Data Summary:**\n- Trades: {len(historical_data)}\n- Symbols: {symbol_summary}\n- Date Range: {historical_data['entry_time'].min()} to {historical_data['entry_time'].max()}\n- Price Range: ${historical_data['entry_price'].min():.2f} - ${historical_data['entry_price'].max():.2f}")
        
        with st.sidebar.expander("📋 Data Format Help"):
            st.markdown("""
            **CSV Format (Required Columns):**
            - `symbol`: Trading symbol (e.g., BTC, ETH)
            - `entry_price`: Entry price (numeric)
            - `exit_price`: Exit price (numeric)
            - `entry_time`: Entry timestamp (datetime)
            
            **Historical Crypto Data Format:**
            Semicolon-separated: `YYYYMMDD HHMMSS;Open;High;Low;Close;Volume`
            
            Supports BTC and ETH data files (BTCUSDT, ETHUSDT formats).
            
            Example:
            ```
            20250101 083000;93576.0;93610.93;93537.5;93610.93;8.21827
            20250101 083100;93610.93;93652.0;93606.2;93652.0;12.14029
            ```
            
            Files are automatically filtered to trading window (7:30-11:00 AM Chicago time).
            You can load multiple symbols simultaneously.
            """)

    if CONFIG["use_ui_overrides"]:
        st.sidebar.header("⚙️ Strategy Parameters")
        trading_days_idx = 0 if CONFIG["trading_days"] == 6 else 1
        trading_days = st.sidebar.selectbox("Simulation Length (Trading Days)", [6, 12], index=trading_days_idx)
        num_trades = st.sidebar.slider(f"Total Trades over {trading_days} days", 10, 100, CONFIG["num_trades"], step=5)
        win_rate = st.sidebar.slider("Estimated Win Rate (%)", 40, 80, int(CONFIG["win_rate"]*100), step=1) / 100
        avg_win = st.sidebar.number_input("Avg Win per Trade ($)", value=CONFIG["avg_win"], min_value=10, step=10)
        avg_loss = st.sidebar.number_input("Avg Loss per Trade ($)", value=CONFIG["avg_loss"], min_value=5, step=5)

        st.sidebar.header("🎲 Monte Carlo")
        num_simulations = st.sidebar.slider("Number of Simulations", 100, 5000, CONFIG["num_simulations"], step=100)
        
        # Strategy section
        st.sidebar.header("📈 Custom Strategy")
        strategy_option = st.sidebar.radio(
            "Strategy Source:",
            ["Default Strategy", "Strategy File", "Inline Code"],
            index=0 if not custom_strategy else (1 if CONFIG.get("strategy_file") else 2)
        )
        
        ui_custom_strategy = None
        if strategy_option == "Strategy File":
            strategy_file_path = st.sidebar.text_input(
                "Strategy File Path",
                value=CONFIG.get("strategy_file", ""),
                help="Path to Python file with 'strategy' function"
            )
            if strategy_file_path and os.path.exists(strategy_file_path):
                try:
                    ui_custom_strategy = load_strategy_from_file(strategy_file_path)
                    st.sidebar.success("✓ Strategy loaded from file")
                except Exception as e:
                    st.sidebar.error(f"Error: {str(e)}")
        elif strategy_option == "Inline Code":
            strategy_code_input = st.sidebar.text_area(
                "Strategy Code",
                value=CONFIG.get("strategy_code", """def strategy(row, context):
    # Example: Buy when CVD > CVD MA
    if row.get('cvd', 0) > row.get('cvd_ma', 0):
        return 'BUY', 0.8
    return 'HOLD', 0.5"""),
                height=150,
                help="Define a 'strategy' function that takes (row, context) and returns (signal, confidence)"
            )
            if st.sidebar.button("Load Strategy Code"):
                try:
                    ui_custom_strategy = load_strategy_from_code(strategy_code_input)
                    st.sidebar.success("✓ Strategy loaded from code")
                except Exception as e:
                    st.sidebar.error(f"Error: {str(e)}")
        
        # Use UI strategy if provided, otherwise use config strategy
        if ui_custom_strategy:
            custom_strategy = ui_custom_strategy
    else:
        # Use config values directly
        trading_days = CONFIG["trading_days"]
        num_trades = CONFIG["num_trades"]
        win_rate = CONFIG["win_rate"]
        avg_win = CONFIG["avg_win"]
        avg_loss = CONFIG["avg_loss"]
        num_simulations = CONFIG["num_simulations"]

    tab1, tab2, tab3, tab4 = st.tabs(["Single Backtest", "Signal Analysis", "Monte Carlo", "Sensitivity"])

    # ---------- Tab 1 ----------
    with tab1:
        st.subheader("Single Backtest")
        
        # Show data source info
        if historical_data is not None:
            symbols_used = ", ".join(historical_data['symbol'].unique())
            st.info(f"📊 Using Historical Crypto data ({symbols_used}): {historical_info}")
        elif csv_data is not None:
            st.info(f"📊 Using CSV data: {csv_info}")
        else:
            st.info("📊 Using synthetic data")

        # Auto-run if data is loaded and UI overrides are disabled
        auto_run = False
        if not CONFIG["use_ui_overrides"] and (historical_data is not None or csv_data is not None):
            auto_run = True
            if 'auto_run_complete' not in st.session_state:
                st.session_state.auto_run_complete = False
        
        run_backtest = False
        if auto_run and not st.session_state.get('auto_run_complete', False):
            run_backtest = True
            st.session_state.auto_run_complete = True
        elif CONFIG["use_ui_overrides"]:
            run_backtest = st.button("▶️ Run Backtest", use_container_width=True, key="run_bt")
        else:
            run_backtest = st.button("▶️ Run Backtest", use_container_width=True, key="run_bt")
        
        if run_backtest:
            # Priority: Historical data > CSV data > Synthetic data
            if historical_data is not None:
                trades = historical_data.copy()
                st.success(f"Using {len(trades)} trades from Historical BTC data")
            elif csv_data is not None:
                trades = csv_data.copy()
                st.success(f"Using {len(trades)} trades from CSV data")
            else:
                trades = generate_synthetic_trades_multi_with_signals(
                    num_trades=num_trades,
                    win_rate=win_rate,
                    avg_win=avg_win,
                    avg_loss=avg_loss,
                    trading_days=trading_days
                )
                st.success(f"Generated {len(trades)} synthetic trades")

            backtester = EnhancedSignalBacktester(
                account_size=account_size,
                daily_dd_limit=daily_dd_limit,
                max_dd_limit=max_dd_limit,
                entry_size=entry_size,
                stop_loss=stop_loss,
                take_profit=take_profit,
                target_gain=target_gain,
                custom_strategy=custom_strategy
            )

            stats = backtester.run_simulation(trades)
            st.session_state.last_stats = stats

        if 'last_stats' in st.session_state:
            stats = st.session_state.last_stats

            st.subheader("📈 Performance")
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            with m1:
                st.metric("Final Equity", f"${stats['final_equity']:.0f}", f"${stats['total_pnl']:+.0f}")
            with m2:
                st.metric("Return %", f"{stats['return_pct']:.2f}%", f"Target: ${target_gain}")
            with m3:
                st.metric("# Trades", stats['num_trades'], f"Win Rate: {stats['win_rate']:.1f}%")
            with m4:
                st.metric("Avg Trade PnL", f"${stats['avg_trade_pnl']:.0f}")
            with m5:
                st.metric("Max DD", f"{stats['max_dd_pct']:.2f}%", f"Limit: {max_dd_limit*100:.1f}%")
            with m6:
                status = "✅ PASS" if stats['passed_account'] else "❌ FAIL"
                color = "green" if stats['passed_account'] else "red"
                st.markdown(f"<h4 style='color:{color}; text-align:center'>{status}</h4>", unsafe_allow_html=True)

            r1, r2, r3, r4 = st.columns(4)
            with r1:
                st.metric("Sharpe", f"{stats['sharpe']:.2f}")
            with r2:
                st.metric("Sortino", f"{stats['sortino']:.2f}")
            with r3:
                st.metric("Calmar", f"{stats['calmar']:.2f}")
            with r4:
                st.metric("Profit Factor", f"{stats['profit_factor']:.2f}")

            if stats['violated_rules']:
                st.error(f"Rule Violation: {stats['violation_reason']}")

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=stats['equity_curve'],
                name='Equity',
                line=dict(color='blue', width=2),
                fill='tozeroy',
                fillcolor='rgba(0, 100, 200, 0.1)'
            ))
            fig.add_hline(y=account_size + target_gain, line_dash="dash", line_color="green",
                          annotation_text="Target")
            fig.add_hline(y=account_size * (1 - max_dd_limit), line_dash="dash", line_color="red",
                          annotation_text="Max DD Limit")
            fig.update_layout(
                title="Equity Curve",
                xaxis_title="Trade #",
                yaxis_title="Equity ($)",
                height=400,
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Trade Log"):
                tdf = pd.DataFrame(stats['trade_log'])
                if not tdf.empty:
                    tdf['pnl'] = tdf['pnl'].map(lambda x: f"${x:+.2f}")
                    tdf['equity'] = tdf['equity'].map(lambda x: f"${x:.2f}")
                    st.dataframe(tdf, use_container_width=True)

    # ---------- Tab 2 ----------
    with tab2:
        st.subheader("Signal Analysis")
        if 'last_stats' in st.session_state:
            stats = st.session_state.last_stats
            sdf = pd.DataFrame(stats['signal_log'])

            col1, col2, col3 = st.columns(3)
            with col1:
                buys = len(sdf[sdf['signal_type'] == 'BUY'])
                st.metric("BUY Signals", buys)
            with col2:
                skipped = len(sdf[sdf['signal_type'] == 'SKIPPED'])
                st.metric("Skipped (Out of Window)", skipped)
            with col3:
                st.metric("Avg Signal Strength", f"{sdf['signal_strength'].mean():.2f}")

            fig = px.histogram(sdf, x='signal_strength', nbins=20, title="Signal Strength Distribution")
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Full Signal Log"):
                st.dataframe(sdf, use_container_width=True)

    # ---------- Tab 3 ----------
    with tab3:
        st.subheader("Monte Carlo: Pass Probability")
        if csv_data is not None:
            st.info("ℹ️ Note: Monte Carlo uses synthetic data for multiple simulations. CSV data is used only for Single Backtest.")
        if st.button("▶️ Run Monte Carlo", use_container_width=True, key="run_mc"):
            progress = st.progress(0)
            pass_count = 0
            results = []

            for i in range(num_simulations):
                trades = generate_synthetic_trades_multi_with_signals(
                    num_trades=num_trades,
                    win_rate=win_rate,
                    avg_win=avg_win,
                    avg_loss=avg_loss,
                    trading_days=trading_days
                )
                backtester = EnhancedSignalBacktester(
                    account_size=account_size,
                    daily_dd_limit=daily_dd_limit,
                    max_dd_limit=max_dd_limit,
                    entry_size=entry_size,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    target_gain=target_gain
                )
                s = backtester.run_simulation(trades)
                if s['passed_account']:
                    pass_count += 1
                results.append(s)
                progress.progress((i + 1) / num_simulations)

            st.session_state.mc_results = results
            st.session_state.pass_rate = pass_count / num_simulations

        if 'mc_results' in st.session_state:
            results = st.session_state.mc_results
            pass_rate = st.session_state.pass_rate

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                color = "green" if pass_rate >= 0.6 else "orange" if pass_rate >= 0.4 else "red"
                st.markdown(f"<h3 style='color:{color}'>Pass Rate: {pass_rate*100:.1f}%</h3>", unsafe_allow_html=True)
            with m2:
                st.metric("Avg Return", f"{np.mean([r['return_pct'] for r in results]):.2f}%")
            with m3:
                st.metric("Avg Max DD", f"{np.mean([r['max_dd_pct'] for r in results]):.2f}%")
            with m4:
                st.metric("Avg Sharpe", f"{np.mean([r['sharpe'] for r in results]):.2f}")

            col1, col2 = st.columns(2)
            with col1:
                rets = [r['return_pct'] for r in results]
                fig = px.histogram(rets, nbins=30, title="Return Distribution (%)")
                fig.add_vline(x=np.mean(rets), line_dash="dash", line_color="blue",
                              annotation_text=f"Mean: {np.mean(rets):.2f}%")
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                dds = [r['max_dd_pct'] for r in results]
                fig = px.histogram(dds, nbins=30, title="Max DD Distribution (%)")
                fig.add_vline(x=max_dd_limit*100, line_dash="dash", line_color="red",
                              annotation_text=f"Limit: {max_dd_limit*100:.1f}%")
                st.plotly_chart(fig, use_container_width=True)

    # ---------- Tab 4 ----------
    with tab4:
        st.subheader("Parameter Sensitivity")
        param = st.selectbox("Parameter to vary:", ["Win Rate (%)", "Avg Win ($)", "Avg Loss ($)", "Num Trades"])
        if st.button("▶️ Run Sensitivity", use_container_width=True, key="run_sens"):
            if param == "Win Rate (%)":
                prange = np.arange(45, 75, 2)
            elif param == "Avg Win ($)":
                prange = np.arange(100, 500, 25)
            elif param == "Avg Loss ($)":
                prange = np.arange(20, 150, 10)
            else:
                prange = np.arange(10, 80, 5)

            sens = []
            progress = st.progress(0)

            for i, val in enumerate(prange):
                if param == "Win Rate (%)":
                    test_wr = val / 100
                    test_aw = avg_win
                    test_al = avg_loss
                    test_nt = num_trades
                elif param == "Avg Win ($)":
                    test_wr = win_rate
                    test_aw = val
                    test_al = avg_loss
                    test_nt = num_trades
                elif param == "Avg Loss ($)":
                    test_wr = win_rate
                    test_aw = avg_win
                    test_al = val
                    test_nt = num_trades
                else:
                    test_wr = win_rate
                    test_aw = avg_win
                    test_al = avg_loss
                    test_nt = int(val)

                pass_cnt = 0
                for _ in range(100):
                    trades = generate_synthetic_trades_multi_with_signals(
                        num_trades=test_nt,
                        win_rate=test_wr,
                        avg_win=test_aw,
                        avg_loss=test_al,
                        trading_days=trading_days
                    )
                    backtester = EnhancedSignalBacktester(
                        account_size=account_size,
                        daily_dd_limit=daily_dd_limit,
                        max_dd_limit=max_dd_limit,
                        entry_size=entry_size,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        target_gain=target_gain
                    )
                    s = backtester.run_simulation(trades)
                    if s['passed_account']:
                        pass_cnt += 1

                sens.append({'param_value': val, 'pass_rate': pass_cnt / 100})
                progress.progress((i + 1) / len(prange))

            sdf = pd.DataFrame(sens)
            sdf.columns = [param, "Pass Rate (%)"]
            sdf["Pass Rate (%)"] = sdf["Pass Rate (%)"] * 100

            fig = px.line(sdf, x=param, y="Pass Rate (%)", markers=True,
                          title=f"Pass Rate vs {param}")
            fig.add_hline(y=60, line_dash="dash", line_color="green",
                          annotation_text="Target 60%")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(sdf.round(2), use_container_width=True)

    st.divider()
    st.caption("Crypto Quant Liquidity Simulator – synthetic for now; plug in real APIs later.")

if __name__ == "__main__":
    main()

