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

# Timezone for trading window (Chicago)
CHICAGO_TZ = pytz.timezone('America/Chicago')

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
    
    # CISD tracking (5-minute timeframe)
    if '5m_candles' not in context:
        context['5m_candles'] = {}  # symbol -> list of completed 5m candles
    
    if 'current_5m_candle' not in context:
        context['current_5m_candle'] = {}  # symbol -> current accumulating 5m candle
    
    if 'last_5m_close_time' not in context:
        context['last_5m_close_time'] = {}  # symbol -> last 5m candle close time
    
    if 'cisd_signals' not in context:
        context['cisd_signals'] = {}  # symbol -> recent CISD signals (bullish/bearish)
    
    if 'structure_legs' not in context:
        context['structure_legs'] = {}  # symbol -> list of structure legs (bullish/bearish swings)
    
    if 'bullish_cisd_formed' not in context:
        context['bullish_cisd_formed'] = {}  # symbol -> bool, bullish CISD signal active
    
    if 'bearish_cisd_formed' not in context:
        context['bearish_cisd_formed'] = {}  # symbol -> bool, bearish CISD signal active
    
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


def get_5m_candle_start(timestamp):
    """Get the start time of the 5-minute candle containing this timestamp."""
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=pytz.UTC)
    
    # Round down to nearest 5 minutes
    minutes = timestamp.minute
    rounded_minutes = (minutes // 5) * 5
    
    candle_start = timestamp.replace(minute=rounded_minutes, second=0, microsecond=0)
    return candle_start


def update_5m_candle(context, row, symbol):
    """Update or create 5-minute candle for CISD tracking."""
    entry_time = row['entry_time']
    if isinstance(entry_time, str):
        entry_time = pd.to_datetime(entry_time)
    if entry_time.tzinfo is None:
        entry_time = entry_time.replace(tzinfo=pytz.UTC)
    
    candle_start = get_5m_candle_start(entry_time)
    
    # Check if we've moved to a new 5m candle
    last_close_time = context['last_5m_close_time'].get(symbol)
    
    if last_close_time is not None:
        last_candle_start = get_5m_candle_start(last_close_time)
        if candle_start != last_candle_start:
            # Finalize previous 5m candle
            prev_5m = context['current_5m_candle'][symbol].copy()
            prev_5m['close_time'] = entry_time
            
            # Store in 5m candles list
            if symbol not in context['5m_candles']:
                context['5m_candles'][symbol] = []
            context['5m_candles'][symbol].append(prev_5m)
            
            # Keep only last 100 candles for analysis
            if len(context['5m_candles'][symbol]) > 100:
                context['5m_candles'][symbol] = context['5m_candles'][symbol][-100:]
            
            # Detect CISD on closed 5m candle
            detect_cisd(context, symbol, prev_5m)
            
            # Update last close time
            context['last_5m_close_time'][symbol] = entry_time
    
    # Initialize or update current 5m candle
    if symbol not in context['current_5m_candle'] or (last_close_time is not None and candle_start != get_5m_candle_start(last_close_time)):
        context['current_5m_candle'][symbol] = {
            'start_time': candle_start,
            'open': row['entry_price'],
            'high': row.get('high', row['entry_price']),
            'low': row.get('low', row['entry_price']),
            'close': row.get('exit_price', row['entry_price']),
            'volume': row.get('volume', 0),
            'bars': 1
        }
        if last_close_time is None:
            context['last_5m_close_time'][symbol] = entry_time
    else:
        # Update current 5m candle
        current_5m = context['current_5m_candle'][symbol]
        current_5m['high'] = max(current_5m['high'], row.get('high', row['entry_price']))
        current_5m['low'] = min(current_5m['low'], row.get('low', row['entry_price']))
        current_5m['close'] = row.get('exit_price', row['entry_price'])
        current_5m['volume'] += row.get('volume', 0)
        current_5m['bars'] += 1


def detect_cisd(context, symbol, closed_5m_candle):
    """
    Detect Change in Structure Direction (CISD) on 5-minute timeframe.
    
    CISD Rules:
    - Bullish CISD: Price closes above the open of the most recent bearish leg
    - Bearish CISD: Price closes below the open of the most recent bullish leg
    """
    if symbol not in context['5m_candles']:
        return
    
    candles_5m = context['5m_candles'][symbol]
    if len(candles_5m) < 3:
        return
    
    current_close = closed_5m_candle['close']
    current_open = closed_5m_candle['open']
    
    # Determine if current candle is bullish or bearish
    is_bullish = current_close > current_open
    
    # Update structure legs (swing detection simplified)
    # Look for swing highs and lows in recent candles
    if len(candles_5m) >= 5:
        # Check for swing high (bearish leg start)
        recent_candles = candles_5m[-5:]
        swing_high_idx = None
        swing_low_idx = None
        
        # Find swing high (local maximum)
        for i in range(1, len(recent_candles) - 1):
            if recent_candles[i]['high'] > recent_candles[i-1]['high'] and \
               recent_candles[i]['high'] > recent_candles[i+1]['high']:
                swing_high_idx = len(candles_5m) - len(recent_candles) + i
                break
        
        # Find swing low (local minimum)
        for i in range(1, len(recent_candles) - 1):
            if recent_candles[i]['low'] < recent_candles[i-1]['low'] and \
               recent_candles[i]['low'] < recent_candles[i+1]['low']:
                swing_low_idx = len(candles_5m) - len(recent_candles) + i
                break
        
        # Track structure legs
        if symbol not in context['structure_legs']:
            context['structure_legs'][symbol] = []
        
        # Add bearish leg if swing high found
        if swing_high_idx is not None:
            swing_candle = candles_5m[swing_high_idx]
            # Check if this is a new bearish leg
            if not context['structure_legs'][symbol] or \
               context['structure_legs'][symbol][-1]['type'] != 'bearish':
                context['structure_legs'][symbol].append({
                    'type': 'bearish',
                    'open': swing_candle['open'],
                    'high': swing_candle['high'],
                    'time': swing_candle.get('start_time', swing_candle.get('close_time'))
                })
                # Keep only last 5 legs
                if len(context['structure_legs'][symbol]) > 5:
                    context['structure_legs'][symbol] = context['structure_legs'][symbol][-5:]
        
        # Add bullish leg if swing low found
        if swing_low_idx is not None:
            swing_candle = candles_5m[swing_low_idx]
            # Check if this is a new bullish leg
            if not context['structure_legs'][symbol] or \
               context['structure_legs'][symbol][-1]['type'] != 'bullish':
                context['structure_legs'][symbol].append({
                    'type': 'bullish',
                    'open': swing_candle['open'],
                    'low': swing_candle['low'],
                    'time': swing_candle.get('start_time', swing_candle.get('close_time'))
                })
                # Keep only last 5 legs
                if len(context['structure_legs'][symbol]) > 5:
                    context['structure_legs'][symbol] = context['structure_legs'][symbol][-5:]
    
    # Detect CISD
    legs = context['structure_legs'].get(symbol, [])
    
    if legs:
        last_leg = legs[-1]
        
        # Bullish CISD: Price closes above the open of the most recent bearish leg
        if last_leg['type'] == 'bearish':
            bearish_leg_open = last_leg['open']
            if current_close > bearish_leg_open:
                context['bullish_cisd_formed'][symbol] = True
                # Store signal with timestamp
                if symbol not in context['cisd_signals']:
                    context['cisd_signals'][symbol] = []
                context['cisd_signals'][symbol].append({
                    'type': 'bullish',
                    'time': closed_5m_candle.get('close_time', closed_5m_candle.get('start_time')),
                    'price': current_close,
                    'trigger_level': bearish_leg_open
                })
                # Keep only last 10 signals
                if len(context['cisd_signals'][symbol]) > 10:
                    context['cisd_signals'][symbol] = context['cisd_signals'][symbol][-10:]
        
        # Bearish CISD: Price closes below the open of the most recent bullish leg
        elif last_leg['type'] == 'bullish':
            bullish_leg_open = last_leg['open']
            if current_close < bullish_leg_open:
                context['bearish_cisd_formed'][symbol] = True
                # Store signal with timestamp
                if symbol not in context['cisd_signals']:
                    context['cisd_signals'][symbol] = []
                context['cisd_signals'][symbol].append({
                    'type': 'bearish',
                    'time': closed_5m_candle.get('close_time', closed_5m_candle.get('start_time')),
                    'price': current_close,
                    'trigger_level': bullish_leg_open
                })
                # Keep only last 10 signals
                if len(context['cisd_signals'][symbol]) > 10:
                    context['cisd_signals'][symbol] = context['cisd_signals'][symbol][-10:]


def is_within_4h_candle_window(entry_time, session_name):
    """
    Check if entry_time is within the active 4H candle window.
    For CISD entries, we focus on the 6am-10am NY Open candle.
    """
    if session_name != 'ny_open':
        return False
    
    # Check if we're within the 6am-10am EST window (which is the NY Open 4H candle)
    if entry_time.tzinfo is None:
        entry_time = entry_time.replace(tzinfo=pytz.UTC)
    
    est_time = entry_time.astimezone(EST_TZ)
    hour = est_time.hour
    
    # 6am-10am EST window
    return 6 <= hour < 10


def is_valid_trading_time(entry_time):
    """
    Check if entry_time is within valid trading window.
    
    Valid trading hours: 7:00 AM to 11:30 AM Chicago time, Monday through Saturday.
    
    Returns: True if valid, False otherwise
    """
    if isinstance(entry_time, str):
        entry_time = pd.to_datetime(entry_time)
    if entry_time.tzinfo is None:
        entry_time = entry_time.replace(tzinfo=pytz.UTC)
    
    # Convert to Chicago timezone
    chicago_time = entry_time.astimezone(CHICAGO_TZ)
    
    # Check day of week: Monday=0, Sunday=6
    weekday = chicago_time.weekday()
    
    # Only allow Monday through Saturday (0-5), exclude Sunday (6)
    if weekday >= 6:  # Sunday
        return False
    
    # Check time: 7:00 AM to 11:30 AM Chicago time
    hour = chicago_time.hour
    minute = chicago_time.minute
    
    # Convert to minutes since midnight for easier comparison
    time_minutes = hour * 60 + minute
    start_minutes = 7 * 60 + 0   # 7:00 AM
    end_minutes = 11 * 60 + 30    # 11:30 AM
    
    return start_minutes <= time_minutes <= end_minutes


def strategy(row, context=None):
    """
    Main strategy function implementing 4H Session Reversal Strategy with candle-by-candle breakdown.
    
    Trading Window: 7:00 AM to 11:30 AM Chicago time, Monday through Saturday.
    
    Entry Criteria:
    1. CISD (Change in Structure Direction) - Multi-timeframe confirmation:
       - 4H Chart: Session-profile model defines bias and marks 4H candle high/low
       - 5M Chart: CISD signals must occur within the active 4H candle (6am-10am NY Open)
       - Entry Rule: 
         * Bullish CISD + Bullish 4H bias → LONG entry
         * Bearish CISD + Bearish 4H bias → SHORT entry
       - CISD Definition:
         * Bullish CISD: Price closes above the open of the most recent bearish leg
         * Bearish CISD: Price closes below the open of the most recent bullish leg
    
    2. Traditional 4H Session Reversal patterns (fallback if no CISD):
       - Asia Reversal, London Reversal, NY 6am/10am reversals
       - Based on PCH/PCL interactions with volume confirmation
    """
    # Check if within valid trading window
    entry_time = row['entry_time']
    if not is_valid_trading_time(entry_time):
        return 'HOLD', 0.5
    
    # Initialize context
    context = initialize_context(context)
    
    # Get symbol
    symbol = row.get('symbol', 'BTC')
    
    # Update 4H candle tracking
    update_4h_candle(context, row, symbol)
    
    # Update 5-minute candle tracking for CISD detection
    update_5m_candle(context, row, symbol)
    
    # Get current session
    if isinstance(entry_time, str):
        entry_time = pd.to_datetime(entry_time)
    if entry_time.tzinfo is None:
        entry_time = entry_time.replace(tzinfo=pytz.UTC)
    
    session_name, start_hour, end_hour = get_4h_session(entry_time)
    
    # Get current session bias
    current_bias = context['session_bias'].get(symbol, 'NEUTRAL')
    trading_direction = get_trading_direction_from_bias(current_bias)
    
    # CISD Entry Logic (Multi-timeframe confirmation)
    # Check if we're in the 6am-10am 4H candle window and have CISD signals
    if is_within_4h_candle_window(entry_time, session_name):
        # Check for recent CISD signals (within last 5 minutes of current time)
        recent_cisd_signals = []
        if symbol in context['cisd_signals']:
            for signal in context['cisd_signals'][symbol]:
                signal_time = signal['time']
                if isinstance(signal_time, str):
                    signal_time = pd.to_datetime(signal_time)
                if signal_time.tzinfo is None:
                    signal_time = signal_time.replace(tzinfo=pytz.UTC)
                
                # Check if signal is recent (within last 30 minutes)
                time_diff = (entry_time - signal_time).total_seconds() / 60
                if time_diff <= 30:  # Within last 30 minutes
                    recent_cisd_signals.append(signal)
        
        # Bullish CISD Entry: 4H bias is bullish AND bullish CISD formed
        if trading_direction == 'BULLISH':
            bullish_cisd = context['bullish_cisd_formed'].get(symbol, False)
            # Check if we have a recent bullish CISD signal
            has_recent_bullish_cisd = any(s['type'] == 'bullish' for s in recent_cisd_signals)
            
            if bullish_cisd or has_recent_bullish_cisd:
                # Additional confirmation: Check if price is near or above CISD trigger level
                current_price = row['entry_price']
                if recent_cisd_signals:
                    latest_bullish = next((s for s in reversed(recent_cisd_signals) if s['type'] == 'bullish'), None)
                    if latest_bullish:
                        trigger_level = latest_bullish['trigger_level']
                        # Entry if price is at or above trigger level (or within 0.1%)
                        if current_price >= trigger_level * 0.999:
                            # High confidence entry based on CISD + 4H bias alignment
                            return 'BUY', 0.9
                else:
                    # If no recent signals but flag is set, still allow entry
                    return 'BUY', 0.85
        
        # Bearish CISD Entry: 4H bias is bearish AND bearish CISD formed
        elif trading_direction == 'BEARISH':
            bearish_cisd = context['bearish_cisd_formed'].get(symbol, False)
            # Check if we have a recent bearish CISD signal
            has_recent_bearish_cisd = any(s['type'] == 'bearish' for s in recent_cisd_signals)
            
            if bearish_cisd or has_recent_bearish_cisd:
                # Additional confirmation: Check if price is near or below CISD trigger level
                current_price = row['entry_price']
                if recent_cisd_signals:
                    latest_bearish = next((s for s in reversed(recent_cisd_signals) if s['type'] == 'bearish'), None)
                    if latest_bearish:
                        trigger_level = latest_bearish['trigger_level']
                        # Entry if price is at or below trigger level (or within 0.1%)
                        if current_price <= trigger_level * 1.001:
                            # High confidence entry based on CISD + 4H bias alignment
                            return 'SELL', 0.9
                else:
                    # If no recent signals but flag is set, still allow entry
                    return 'SELL', 0.85
    
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
