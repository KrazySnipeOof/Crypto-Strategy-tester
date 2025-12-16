"""
4-Hour Session Reversal Strategy - Candle-by-Candle Breakdown
==============================================================

This strategy implements the Previous Candle Theory (PCT) + VSA framework
with a candle-by-candle breakdown for session profiles.

Key Concept: Each closed 4H candle becomes the "previous candle" for the next one.
We track candle sequences within session groups:
- Asia Reversal: Candle 1 (NY PM), Candle 2 (Asia Open), Candle 3 (Late Asia)
- London Reversal: Candle 1 (Late Asia), Candle 2 (London Open), Candle 3 (London Reversal)
- NY 6am Reversal: Candle 1 (London Leg), Candle 2 (NY Open)
- NY 10am Reversal: Candle 1 (NY Open), Candle 2 (10am Reversal)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz

# Timezone for session calculations (EST/EDT)
EST_TZ = pytz.timezone('US/Eastern')

# 4H Session Times (EST)
SESSION_TIMES = {
    'frankfurt': (22, 2),   # 22:00-02:00 EST (Late Asia/Pre-London)
    'london': (2, 6),       # 02:00-06:00 EST (London Open/Expansion)
    'ny_open': (6, 10),     # 06:00-10:00 EST (NY Open/6am Reversal)
    'ny_silver': (10, 14),  # 10:00-14:00 EST (10am Reversal)
    'ny_pm': (14, 18),     # 14:00-18:00 EST (NY PM/Close)
    'asia_open': (18, 22), # 18:00-22:00 EST (Asia Open)
}

# Session Groups (candle sequences)
SESSION_GROUPS = {
    'asia_reversal': {
        'candle_1': 'ny_pm',      # 14:00-18:00 - Sets PCH/PCL
        'candle_2': 'asia_open',   # 18:00-22:00 - Attacks Candle 1
        'candle_3': 'frankfurt'   # 22:00-02:00 - Forms other side or consolidates
    },
    'london_reversal': {
        'candle_1': 'frankfurt',  # 22:00-02:00 - Anchor for London
        'candle_2': 'london',     # 02:00-06:00 - Creates trend
        'candle_3': 'ny_open'     # 06:00-10:00 - Reversal or continuation
    },
    'ny_6am_reversal': {
        'candle_1': 'london',     # 02:00-06:00 - London leg
        'candle_2': 'ny_open'    # 06:00-10:00 - NY Open reversal
    },
    'ny_10am_reversal': {
        'candle_1': 'ny_open',    # 06:00-10:00 - NY morning leg
        'candle_2': 'ny_silver'   # 10:00-14:00 - 10am reversal
    }
}


def initialize_context(context):
    """Initialize context with 4H candle tracking structures."""
    if context is None:
        context = {}
    
    if '4h_candles' not in context:
        context['4h_candles'] = {}  # symbol -> list of completed 4H candles
    
    if 'previous_4h_candle' not in context:
        context['previous_4h_candle'] = {}  # symbol -> previous completed 4H candle
    
    if 'current_4h_candle' not in context:
        context['current_4h_candle'] = {}  # symbol -> current accumulating 4H candle
    
    if 'session_bias' not in context:
        context['session_bias'] = {}  # symbol -> current session bias
    
    if 'last_4h_close_time' not in context:
        context['last_4h_close_time'] = {}  # symbol -> last 4H candle close time
    
    if 'candle_sequence' not in context:
        context['candle_sequence'] = {}  # symbol -> current candle sequence info
    
    return context


def get_4h_session(timestamp):
    """
    Determine which 4H session a timestamp belongs to.
    
    Returns: (session_name, session_start_hour, session_end_hour)
    """
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=pytz.UTC)
    
    est_time = timestamp.astimezone(EST_TZ)
    hour = est_time.hour
    
    # Frankfurt session wraps: 22:00-02:00 (covers hours 22, 23, 0, 1)
    if hour >= 22 or hour < 2:
        return ('frankfurt', 22, 2)
    elif 2 <= hour < 6:
        return ('london', 2, 6)
    elif 6 <= hour < 10:
        return ('ny_open', 6, 10)
    elif 10 <= hour < 14:
        return ('ny_silver', 10, 14)
    elif 14 <= hour < 18:
        return ('ny_pm', 14, 18)
    else:  # 18 <= hour < 22
        return ('asia_open', 18, 22)


def get_session_group_and_candle_number(session_name):
    """
    Determine which session group and candle number this session belongs to.
    
    Returns: (group_name, candle_number) or (None, None) if not part of a tracked group
    """
    for group_name, candles in SESSION_GROUPS.items():
        if session_name == candles.get('candle_1'):
            return (group_name, 1)
        elif session_name == candles.get('candle_2'):
            return (group_name, 2)
        elif session_name == candles.get('candle_3'):
            return (group_name, 3)
    
    return (None, None)


def get_4h_candle_start(timestamp):
    """Get the start time of the 4H candle containing this timestamp."""
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=pytz.UTC)
    
    est_time = timestamp.astimezone(EST_TZ)
    session_name, start_hour, end_hour = get_4h_session(timestamp)
    
    # Handle Frankfurt session wrapping to next day
    if session_name == 'frankfurt':
        if est_time.hour < 2:
            candle_start = est_time.replace(hour=22, minute=0, second=0, microsecond=0) - timedelta(days=1)
        else:
            candle_start = est_time.replace(hour=22, minute=0, second=0, microsecond=0)
    else:
        candle_start = est_time.replace(hour=start_hour, minute=0, second=0, microsecond=0)
    
    return candle_start


def is_4h_candle_close(timestamp, last_close_time=None):
    """
    Check if this timestamp represents a 4H candle close.
    
    Returns: (is_close, candle_start_time)
    """
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=pytz.UTC)
    
    current_candle_start = get_4h_candle_start(timestamp)
    
    if last_close_time is None:
        # First check - if we're at session boundary, it's a close
        est_time = timestamp.astimezone(EST_TZ)
        if est_time.minute >= 59:
            return True, current_candle_start
        return False, current_candle_start
    
    last_candle_start = get_4h_candle_start(last_close_time)
    
    # If candle start time changed, we've moved to a new 4H candle
    if current_candle_start != last_candle_start:
        return True, last_candle_start  # Return the previous candle's start (which just closed)
    
    return False, current_candle_start


def update_4h_candle(context, row, symbol):
    """Update or create 4H candle for the current session."""
    entry_time = row['entry_time']
    if isinstance(entry_time, str):
        entry_time = pd.to_datetime(entry_time)
    if entry_time.tzinfo is None:
        entry_time = entry_time.replace(tzinfo=pytz.UTC)
    
    # Get current 4H session and candle start
    session_name, start_hour, end_hour = get_4h_session(entry_time)
    candle_start = get_4h_candle_start(entry_time)
    
    # Check if we've moved to a new 4H candle
    last_close_time = context['last_4h_close_time'].get(symbol)
    is_close, closed_candle_start = is_4h_candle_close(entry_time, last_close_time)
    
    if is_close and symbol in context['current_4h_candle']:
        # Finalize previous 4H candle
        prev_candle = context['current_4h_candle'][symbol].copy()
        prev_candle['session'] = get_4h_session(pd.Timestamp(prev_candle['start_time']).tz_localize('UTC').astimezone(EST_TZ))[0]
        prev_candle['close_time'] = entry_time
        prev_candle['pch'] = prev_candle['high']  # Previous Candle High
        prev_candle['pcl'] = prev_candle['low']    # Previous Candle Low
        
        # Store in 4H candles list
        if symbol not in context['4h_candles']:
            context['4h_candles'][symbol] = []
        context['4h_candles'][symbol].append(prev_candle)
        
        # Keep only last 20 candles for analysis
        if len(context['4h_candles'][symbol]) > 20:
            context['4h_candles'][symbol] = context['4h_candles'][symbol][-20:]
        
        # Update previous 4H candle (this becomes the setup candle)
        context['previous_4h_candle'][symbol] = prev_candle.copy()
        
        # Determine session group and candle number for the completed candle
        group_name, candle_num = get_session_group_and_candle_number(prev_candle['session'])
        if group_name:
            context['candle_sequence'][symbol] = {
                'group': group_name,
                'candle_number': candle_num,
                'completed_candle': prev_candle
            }
        
        # Calculate session bias for the completed candle
        calculate_candle_bias(context, symbol, prev_candle)
        
        # Update last close time
        context['last_4h_close_time'][symbol] = entry_time
    
    # Initialize or update current 4H candle
    if symbol not in context['current_4h_candle'] or is_close:
        context['current_4h_candle'][symbol] = {
            'start_time': candle_start,
            'session': session_name,
            'open': row['entry_price'],
            'high': row.get('high', row['entry_price']),
            'low': row.get('low', row['entry_price']),
            'close': row.get('exit_price', row['entry_price']),
            'volume': row.get('volume', 0),
            'bars': 1
        }
    else:
        # Update current 4H candle
        current_candle = context['current_4h_candle'][symbol]
        current_candle['high'] = max(current_candle['high'], row.get('high', row['entry_price']))
        current_candle['low'] = min(current_candle['low'], row.get('low', row['entry_price']))
        current_candle['close'] = row.get('exit_price', row['entry_price'])
        current_candle['volume'] += row.get('volume', 0)
        current_candle['bars'] += 1


def calculate_volume_ma(context, symbol, session_name, lookback=10):
    """Calculate volume moving average for 4H candles."""
    if symbol not in context['4h_candles']:
        return None
    
    candles = context['4h_candles'][symbol]
    if len(candles) < 2:
        return None
    
    # For Asia sessions, compare to recent Asian sessions only
    if session_name in ['asia_open', 'frankfurt']:
        asia_candles = [c for c in candles[-lookback*2:] if c.get('session') in ['asia_open', 'frankfurt']]
        if len(asia_candles) >= 3:
            volumes = [c['volume'] for c in asia_candles[-3:]]
            return np.mean(volumes)
    
    # General 4H volume MA
    volumes = [c['volume'] for c in candles[-lookback:]]
    return np.mean(volumes) if volumes else None


def calculate_candle_bias(context, symbol, completed_candle):
    """
    Calculate bias based on candle-by-candle breakdown within session groups.
    """
    if symbol not in context['previous_4h_candle']:
        context['session_bias'][symbol] = 'NEUTRAL'
        return
    
    prev_candle = context['previous_4h_candle'][symbol]
    pch = prev_candle.get('pch', prev_candle['high'])
    pcl = prev_candle.get('pcl', prev_candle['low'])
    
    current_high = completed_candle['high']
    current_low = completed_candle['low']
    current_close = completed_candle['close']
    current_open = completed_candle['open']
    current_volume = completed_candle['volume']
    session_name = completed_candle.get('session', 'unknown')
    
    # Get candle sequence info
    seq_info = context['candle_sequence'].get(symbol, {})
    group_name = seq_info.get('group')
    candle_num = seq_info.get('candle_number')
    
    # Calculate volume MA
    volume_ma = calculate_volume_ma(context, symbol, session_name)
    if volume_ma is None or volume_ma == 0:
        volume_ma = current_volume
    
    volume_ratio = current_volume / volume_ma if volume_ma > 0 else 1.0
    
    # Session-specific volume thresholds
    if session_name in ['asia_open', 'frankfurt']:
        is_high_volume = volume_ratio >= 1.3
        is_ultra_high_volume = volume_ratio >= 1.8
        is_low_volume = volume_ratio <= 0.8
    else:
        is_high_volume = volume_ratio >= 1.5
        is_ultra_high_volume = volume_ratio >= 2.0
        is_low_volume = volume_ratio <= 0.7
    
    # Price action analysis
    price_change = current_close - current_open
    closes_above_pch = current_close > pch
    closes_below_pcl = current_close < pcl
    wick_above_pch = current_high > pch and current_close <= pch
    wick_below_pcl = current_low < pcl and current_close >= pcl
    is_inside_bar = current_high <= pch and current_low >= pcl
    
    # Compare volume to previous candle
    if len(context['4h_candles'][symbol]) > 1:
        prev_volume = context['4h_candles'][symbol][-2]['volume']
        volume_higher_than_prev = current_volume > prev_volume
    else:
        volume_higher_than_prev = False
    
    # Apply candle-by-candle logic based on session group
    
    # ASIA REVERSAL GROUP
    if group_name == 'asia_reversal':
        if candle_num == 2:  # Asia Open (18:00-22:00)
            if wick_above_pch:
                # Spike above PCH then close back inside
                if is_high_volume:
                    context['session_bias'][symbol] = 'BEARISH_REVERSAL'  # Asia reversal short
                    return
            
            if wick_below_pcl:
                # Spike below PCL then close back inside
                if is_high_volume:
                    context['session_bias'][symbol] = 'BULLISH_REVERSAL'  # Asia reversal long
                    return
            
            if closes_above_pch or closes_below_pcl:
                # Clean close beyond PCH/PCL
                if is_high_volume:
                    direction = 'BULLISH' if closes_above_pch else 'BEARISH'
                    context['session_bias'][symbol] = f'{direction}_CONTINUATION'
                    return
        
        elif candle_num == 3:  # Late Asia/Pre-Frankfurt (22:00-02:00)
            # Forms other side of Asian range or consolidates
            if wick_below_pcl and current_close > pcl:
                # Sweep PCL and snap back (second reversal)
                if is_high_volume:
                    context['session_bias'][symbol] = 'BULLISH_REVERSAL'
                    return
            
            if is_inside_bar:
                context['session_bias'][symbol] = 'NEUTRAL_CONSOLIDATION'
                return
    
    # LONDON REVERSAL GROUP
    elif group_name == 'london_reversal':
        if candle_num == 2:  # London Open/Expansion (02:00-06:00)
            if wick_above_pch:
                # Sweep above Candle 1 PCH then close back inside
                if is_high_volume:
                    context['session_bias'][symbol] = 'BEARISH_BIAS'  # Expect down toward PCL
                    return
            
            if wick_below_pcl:
                # Sweep below Candle 1 PCL then close back inside
                if is_high_volume:
                    context['session_bias'][symbol] = 'BULLISH_BIAS'  # Expect up toward PCH
                    return
            
            if closes_above_pch or closes_below_pcl:
                # Strong displacement outside PCH/PCL
                if is_high_volume:
                    direction = 'BULLISH' if closes_above_pch else 'BEARISH'
                    context['session_bias'][symbol] = f'{direction}_CONTINUATION'
                    return
        
        elif candle_num == 3:  # London Reversal/NY Setup (06:00-10:00)
            # Uses Candle 2's high/low as liquidity
            candle_2 = context['4h_candles'][symbol][-2] if len(context['4h_candles'][symbol]) >= 2 else None
            if candle_2:
                candle_2_pch = candle_2['high']
                candle_2_pcl = candle_2['low']
                
                # Wick through Candle 2 high then fails
                if current_high > candle_2_pch and current_close < candle_2_pch:
                    if is_ultra_high_volume:  # Climactic volume
                        context['session_bias'][symbol] = 'BEARISH_REVERSAL'  # London reversal short
                        return
                
                # Sweep Candle 2 low then reclaim
                if current_low < candle_2_pcl and current_close > candle_2_pcl:
                    if is_ultra_high_volume:  # Climactic volume
                        context['session_bias'][symbol] = 'BULLISH_REVERSAL'  # London reversal long
                        return
            
            # If expands in same direction with strong volume
            if (closes_above_pch or closes_below_pcl) and is_high_volume:
                direction = 'BULLISH' if closes_above_pch else 'BEARISH'
                context['session_bias'][symbol] = f'{direction}_CONTINUATION'  # Trend day
                return
    
    # NY 6AM REVERSAL GROUP
    elif group_name == 'ny_6am_reversal':
        if candle_num == 2:  # NY Open/6am Reversal (06:00-10:00)
            # Check if Candle 1 (London) was bullish or bearish
            candle_1 = context['4h_candles'][symbol][-2] if len(context['4h_candles'][symbol]) >= 2 else None
            if candle_1:
                london_was_bullish = candle_1['close'] > candle_1['open']
                
                # Bearish 6am reversal
                if london_was_bullish and wick_above_pch and current_close < pch:
                    if volume_higher_than_prev:  # Volume higher than London candle
                        context['session_bias'][symbol] = 'BEARISH_REVERSAL'  # Upthrust/buying climax
                        return
                
                # Bullish 6am reversal
                if not london_was_bullish and wick_below_pcl and current_close > pcl:
                    if is_high_volume:  # Stopping volume
                        context['session_bias'][symbol] = 'BULLISH_REVERSAL'  # Spring/stopping volume
                        return
            
            # NY continuation
            if (closes_above_pch or closes_below_pcl) and is_high_volume:
                direction = 'BULLISH' if closes_above_pch else 'BEARISH'
                context['session_bias'][symbol] = f'{direction}_CONTINUATION'
                return
    
    # NY 10AM REVERSAL GROUP
    elif group_name == 'ny_10am_reversal':
        if candle_num == 2:  # 10am Reversal (10:00-14:00)
            # Check if Candle 1 (NY Open) was impulsive
            candle_1 = context['4h_candles'][symbol][-2] if len(context['4h_candles'][symbol]) >= 2 else None
            if candle_1:
                ny_open_was_bullish = candle_1['close'] > candle_1['open']
                
                # Bearish 10am reversal (top of day)
                if ny_open_was_bullish and wick_above_pch and current_close < pch:
                    if is_ultra_high_volume:  # Session high volume
                        context['session_bias'][symbol] = 'BEARISH_REVERSAL'  # Buying climax
                        return
                
                # Bullish 10am reversal (bottom of day)
                if not ny_open_was_bullish and wick_below_pcl and current_close > pcl:
                    if is_ultra_high_volume:  # Selling climax absorbed
                        context['session_bias'][symbol] = 'BULLISH_REVERSAL'
                        return
            
            # No 10am reversal / continuation
            if (closes_above_pch or closes_below_pcl) and is_high_volume:
                direction = 'BULLISH' if closes_above_pch else 'BEARISH'
                context['session_bias'][symbol] = f'{direction}_CONTINUATION'
                return
    
    # Default: Apply standard PCT+VSA logic
    if closes_above_pch:
        if is_high_volume:
            context['session_bias'][symbol] = 'BULLISH_CONTINUATION'
        else:
            context['session_bias'][symbol] = 'NEUTRAL'
    elif closes_below_pcl:
        if is_high_volume:
            context['session_bias'][symbol] = 'BEARISH_CONTINUATION'
        else:
            context['session_bias'][symbol] = 'NEUTRAL'
    else:
        context['session_bias'][symbol] = 'NEUTRAL'


def get_trading_direction_from_bias(bias):
    """Convert session bias to trading direction."""
    bullish_biases = ['BULLISH_REVERSAL', 'BULLISH_CONTINUATION', 'BULLISH_BIAS']
    bearish_biases = ['BEARISH_REVERSAL', 'BEARISH_CONTINUATION', 'BEARISH_BIAS']
    
    if bias in bullish_biases:
        return 'BULLISH'
    elif bias in bearish_biases:
        return 'BEARISH'
    else:
        return 'NEUTRAL'


def strategy(row, context=None):
    """
    Main strategy function implementing 4H Session Reversal Strategy with candle-by-candle breakdown.
    """
    # Initialize context
    context = initialize_context(context)
    
    # Get symbol
    symbol = row.get('symbol', 'BTC')
    
    # Update 4H candle tracking
    update_4h_candle(context, row, symbol)
    
    # Get current session
    entry_time = row['entry_time']
    if isinstance(entry_time, str):
        entry_time = pd.to_datetime(entry_time)
    if entry_time.tzinfo is None:
        entry_time = entry_time.replace(tzinfo=pytz.UTC)
    
    session_name, start_hour, end_hour = get_4h_session(entry_time)
    
    # Get current session bias
    current_bias = context['session_bias'].get(symbol, 'NEUTRAL')
    trading_direction = get_trading_direction_from_bias(current_bias)
    
    # If we don't have enough data yet, hold
    if symbol not in context['previous_4h_candle']:
        return 'HOLD', 0.5
    
    # Get previous 4H candle PCH/PCL
    prev_candle = context['previous_4h_candle'][symbol]
    pch = prev_candle.get('pch', prev_candle['high'])
    pcl = prev_candle.get('pcl', prev_candle['low'])
    
    # Get current 4H candle data
    current_candle = context['current_4h_candle'].get(symbol, {})
    current_high = current_candle.get('high', row.get('high', row['entry_price']))
    current_low = current_candle.get('low', row.get('low', row['entry_price']))
    current_close = row.get('exit_price', row['entry_price'])
    entry_price = row['entry_price']
    
    # Get other indicators
    vsa_ratio = row.get('vsa_ratio', 0)
    cvd = row.get('cvd', 0)
    cvd_ma = row.get('cvd_ma', 0)
    volume = row.get('volume', 0)
    
    # Check if price is interacting with PCH/PCL (within 0.1% for entry consideration)
    near_pch = abs(entry_price - pch) / pch < 0.001
    near_pcl = abs(entry_price - pcl) / pcl < 0.001
    wick_hit_pch = current_high > pch * 0.9999
    wick_hit_pcl = current_low < pcl * 1.0001
    
    # Get candle sequence info for session-specific logic
    seq_info = context['candle_sequence'].get(symbol, {})
    group_name = seq_info.get('group')
    candle_num = seq_info.get('candle_number')
    
    # Calculate volume MA for current session
    volume_ma = calculate_volume_ma(context, symbol, session_name)
    if volume_ma is None or volume_ma == 0:
        volume_ma = volume
    
    volume_ratio = volume / volume_ma if volume_ma > 0 else 1.0
    
    # Session-specific volume thresholds
    if session_name in ['asia_open', 'frankfurt']:
        is_high_volume = volume_ratio >= 1.3
        is_ultra_high_volume = volume_ratio >= 1.8
    else:
        is_high_volume = volume_ratio >= 1.5
        is_ultra_high_volume = volume_ratio >= 2.0
    
    # Apply candle-by-candle entry logic
    
    # ASIA REVERSAL - Candle 2 (Asia Open)
    if group_name == 'asia_reversal' and candle_num == 2:
        # Spike above PCH then close back inside
        if wick_hit_pch and current_close < pch:
            if is_high_volume:
                return 'SELL', 0.75  # Asia reversal short
        
        # Spike below PCL then close back inside
        if wick_hit_pcl and current_close > pcl:
            if is_high_volume:
                return 'BUY', 0.75  # Asia reversal long
    
    # LONDON REVERSAL - Candle 3 (London Reversal/NY Setup)
    elif group_name == 'london_reversal' and candle_num == 3:
        # Get Candle 2's PCH/PCL
        if len(context['4h_candles'][symbol]) >= 2:
            candle_2 = context['4h_candles'][symbol][-2]
            candle_2_pch = candle_2['high']
            candle_2_pcl = candle_2['low']
            
            # Wick through Candle 2 high then fails
            if current_high > candle_2_pch and current_close < candle_2_pch:
                if is_ultra_high_volume:
                    return 'SELL', 0.85  # London reversal short
            
            # Sweep Candle 2 low then reclaim
            if current_low < candle_2_pcl and current_close > candle_2_pcl:
                if is_ultra_high_volume:
                    return 'BUY', 0.85  # London reversal long
    
    # NY 6AM REVERSAL - Candle 2 (NY Open)
    elif group_name == 'ny_6am_reversal' and candle_num == 2:
        # Bearish reversal: Wick above PCH, volume higher than London
        if wick_hit_pch and current_close < pch:
            if len(context['4h_candles'][symbol]) >= 2:
                london_volume = context['4h_candles'][symbol][-2]['volume']
                if volume > london_volume:
                    return 'SELL', 0.85  # Upthrust/buying climax
        
        # Bullish reversal: Wick below PCL, stopping volume
        if wick_hit_pcl and current_close > pcl:
            if is_ultra_high_volume:
                return 'BUY', 0.85  # Spring/stopping volume
    
    # NY 10AM REVERSAL - Candle 2 (10am Reversal)
    elif group_name == 'ny_10am_reversal' and candle_num == 2:
        # Bearish reversal: Top of day, climactic rejection
        if wick_hit_pch and current_close < pch:
            if is_ultra_high_volume:
                return 'SELL', 0.9  # Buying climax
        
        # Bullish reversal: Bottom of day, panic absorption
        if wick_hit_pcl and current_close > pcl:
            if is_ultra_high_volume:
                return 'BUY', 0.9  # Selling climax absorbed
    
    # General signal generation filtered by session bias
    signal_score = 0.0
    max_score = 0.0
    
    # 1. CVD signal (30%)
    cvd_weight = 0.3
    max_score += cvd_weight
    if cvd > cvd_ma:
        cvd_signal = 0.8 if cvd > cvd_ma * 1.1 else 0.6
    else:
        cvd_signal = 0.2 if cvd < cvd_ma * 0.9 else 0.4
    signal_score += cvd_signal * cvd_weight
    
    # 2. VSA signal (25%)
    vsa_weight = 0.25
    max_score += vsa_weight
    if vsa_ratio > 500:
        vsa_signal = 0.8
    elif vsa_ratio > 300:
        vsa_signal = 0.6
    else:
        vsa_signal = 0.3
    signal_score += vsa_signal * vsa_weight
    
    # 3. Price action relative to PCH/PCL (25%)
    pa_weight = 0.25
    max_score += pa_weight
    if near_pch or near_pcl:
        pa_signal = 0.7  # Near liquidity level
    elif entry_price > pch:
        pa_signal = 0.6
    elif entry_price < pcl:
        pa_signal = 0.4
    else:
        pa_signal = 0.5
    signal_score += pa_signal * pa_weight
    
    # 4. Session bias alignment (20%)
    bias_weight = 0.2
    max_score += bias_weight
    
    if trading_direction == 'BULLISH':
        if 'REVERSAL' in current_bias:
            bias_signal = 1.0
        elif 'CONTINUATION' in current_bias:
            bias_signal = 0.9
        elif 'BIAS' in current_bias:
            bias_signal = 0.8
        else:
            bias_signal = 0.3
    elif trading_direction == 'BEARISH':
        if 'REVERSAL' in current_bias:
            bias_signal = 1.0
        elif 'CONTINUATION' in current_bias:
            bias_signal = 0.9
        else:
            bias_signal = 0.3
    else:
        bias_signal = 0.5
    
    signal_score += bias_signal * bias_weight
    
    # Normalize score
    normalized_score = signal_score / max_score if max_score > 0 else 0.5
    
    # Generate signal based on session bias and signal strength
    if trading_direction == 'BULLISH':
        if normalized_score > 0.65:
            confidence = min(0.9, normalized_score)
            return 'BUY', confidence
    elif trading_direction == 'BEARISH':
        if normalized_score > 0.65:
            confidence = min(0.9, normalized_score)
            return 'SELL', confidence
    
    return 'HOLD', 0.5
