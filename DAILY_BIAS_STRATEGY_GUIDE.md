# 4-Hour Session Reversal Strategy Guide - Candle-by-Candle Breakdown

## Overview

The **4-Hour Session Reversal Strategy** implements the Previous Candle Theory (PCT) + Volume Spread Analysis (VSA) framework with a **candle-by-candle breakdown** for session profiles.

### Key Concept

**Each closed 4H candle becomes the "previous candle" for the next one.** We track candle sequences within session groups and apply specific logic based on which candle in the sequence we're analyzing.

## Session Groups and Candle Sequences

The strategy tracks four main session groups, each with specific candle sequences:

### 1. Asia Reversal Group (relative to NY Close)

**Candle 1: NY PM / Close (14:00-18:00 EST)**
- Role: Sets **PCH/PCL** for Asian session
- Task: Mark its high/low when it closes at 18:00

**Candle 2: Asia Open (18:00-22:00 EST)**
- Watch: Does this candle take PCH or PCL of Candle 1?
  - **Spike above PCH then close back inside** + relatively high Asia volume → **Asia reversal SHORT** toward Candle 1 PCL
  - **Spike below PCL then close back inside** + volume spike → **Asia reversal LONG** toward Candle 1 PCH
  - **Clean close beyond PCH/PCL** + stronger-than-usual Asia volume → continuation of NY direction

**Candle 3: Late Asia / Pre-Frankfurt (22:00-02:00 EST)**
- Often forms the other side of the Asian range:
  - If Candle 2 reversed down from PCH but didn't reach PCL, Candle 3 may **sweep PCL** and snap back (second reversal, forming full Asia range)
  - If Candle 2 already firmly broke out with volume, Candle 3 usually **consolidates inside** the new direction

### 2. London Reversal Group (relative to Asian/Pre-London)

**Candle 1: Late Asia / Pre-London (22:00-02:00 EST)**
- Role: Anchor for London; mark its **PCH/PCL** at 02:00

**Candle 2: London Open / Expansion (02:00-06:00 EST)**
- This candle often **creates the trend to be reversed later**
- Candle-by-candle logic:
  - **Sweep above Candle 1 PCH then close back inside** + high volume → **bearish bias**, expecting London to drive down toward Candle 1 PCL
  - **Sweep below Candle 1 PCL then close back inside** + high volume → **bullish bias**, London drives up
  - **Strong displacement outside PCH/PCL** + high volume → London establishing a trending session; next candle likely continuation

**Candle 3: London Reversal / NY Setup (06:00-10:00 EST)**
- Uses **Candle 2's high/low** as liquidity:
  - If Candle 2 trended up, Candle 3 often **wicks through Candle 2 high** then fails:
    - Wick > Candle 2 PCH, close back below + **climactic volume** → **London reversal SHORT**
  - If Candle 2 trended down, Candle 3 may **sweep Candle 2 low** then reclaim:
    - Wick < PCL, close back above + **climactic volume** → **London reversal LONG**
  - If Candle 3 simply expands in same direction with strong volume → **no reversal, trend day**; bias stays with Candle 2 direction

### 3. NY 6am Reversal Group

**Candle 1: London Leg (02:00-06:00 EST)**
- Role: Sets the **London move** and its PCH/PCL. Mark these at 06:00

**Candle 2: NY Open / 6am Reversal (06:00-10:00 EST)**
- Candle-by-candle interpretation:
  - **Bearish 6am reversal:**
    - London candle (C1) was bullish
    - 6am candle (C2) **trades above C1 high**, prints a long upper wick, and closes back below C1 PCH
    - Volume on C2 is **higher than C1** → upthrust / buying climax → daily bias flips bearish toward C1 PCL
  - **Bullish 6am reversal:**
    - London candle was bearish
    - 6am candle **trades below C1 low**, long lower wick, closes back inside range + **high volume** → spring / stopping volume → bias flips bullish toward C1 PCH
  - **NY continuation:**
    - 6am candle **closes outside** C1 range with strong body + **high volume** in direction of the breakout → bias stays with London trend, expect continuation into 10am

### 4. NY 10am Reversal Group

**Candle 1: NY Open Move (06:00-10:00 EST)**
- Role: Defines the **NY morning leg** and intraday PCH/PCL. Mark its high/low at 10:00

**Candle 2: 10am Reversal (10:00-14:00 EST)**
- Candle-by-candle logic:
  - **Bearish 10am reversal (top of day):**
    - C1 was impulsive bullish (often already above London high)
    - C2 quickly **pushes through C1 high**, volume spikes (session high), but candle closes back below C1 PCH → buying climax into London close → bias flips bearish for the rest of NY
  - **Bullish 10am reversal (bottom of day):**
    - C1 was impulsive bearish (often below London low)
    - C2 **sweeps C1 low**, prints long lower wick on high volume, and closes back inside → selling climax → bias flips bullish
  - **No 10am reversal / continuation:**
    - C2 simply extends C1's trend with a strong close beyond C1's extreme + **sustained high volume** → treat 10am as continuation leg, not reversal; hold bias with morning trend

## How It Works

### Step 1: 4H Candle Aggregation

The strategy aggregates minute-by-minute data into 4H candles:
- Tracks each 4H session period (Frankfurt, London, NY Open, NY Silver, NY PM, Asia Open)
- Accumulates OHLCV data for each 4H period
- Identifies when a 4H candle closes (session boundary)

### Step 2: Candle Sequence Tracking

When a 4H candle closes:
- Determines which session group it belongs to (Asia Reversal, London Reversal, NY 6am, NY 10am)
- Identifies which candle number in the sequence (Candle 1, 2, or 3)
- Stores the completed candle with PCH/PCL marked

### Step 3: Previous Candle Analysis

Each closed candle becomes the "previous candle" for the next one:
- PCH (Previous Candle High) = high of the closed candle
- PCL (Previous Candle Low) = low of the closed candle
- These levels become the liquidity targets for the next candle

### Step 4: Candle-by-Candle Logic

Applies specific logic based on:
- **Session group** (which sequence we're in)
- **Candle number** (1, 2, or 3 in the sequence)
- **Price interaction** with PCH/PCL (wick through, close outside, inside bar)
- **Volume confirmation** (high, ultra-high, or low relative to session average)

### Step 5: Volume Analysis

For each session, calculates:
- **Session-Specific Volume MA**: 
  - NY/London: 10-period average of all 4H candles
  - Asia: 3-period average of recent Asian sessions only
- **Volume Classification**:
  - Ultra-high volume: ≥200% of average (NY/London) or ≥180% (Asia)
  - High volume: ≥150% of average (NY/London) or ≥130% (Asia)
  - Low volume: ≤70% of average (NY/London) or ≤80% (Asia)

### Step 6: Signal Generation

Generates signals based on:
- **Candle sequence logic** (specific patterns for each candle in each group)
- **Real-time price interaction** with PCH/PCL
- **Volume confirmation** (ultra-high for reversals, high for continuations)
- **CVD alignment** (Cumulative Volume Delta vs its MA)
- **VSA ratio** (volume relative to price range)

## Usage

### Option 1: Configure in `crypto_quant_liquidity_simulator.py`

Edit the `CONFIG` dictionary:

```python
CONFIG = {
    # ... other settings ...
    "strategy_file": "daily_bias_4h_strategy.py",
    "data_source": "historical",
    "historical_symbols": ["BTC", "ETH"],
    # ... rest of config ...
}
```

### Option 2: Use in UI

1. Launch the app
2. Go to sidebar → **Custom Strategy** section
3. Select **Strategy File**
4. Enter path: `daily_bias_4h_strategy.py`
5. Run backtest

## Strategy Parameters

The strategy uses these default parameters:

**Volume Thresholds**:
- NY/London Ultra-High: 200% of 4H average
- NY/London High: 150% of 4H average
- Asia Ultra-High: 180% of Asian session average
- Asia High: 130% of Asian session average

**Signal Thresholds**:
- Reversal signals: Require ultra-high volume + price rejection
- Continuation signals: Require high volume + price displacement
- General signals: Score > 0.65 with bias alignment

**Session Times (EST)**:
- Frankfurt: 22:00-02:00 (wraps to next day)
- London: 02:00-06:00
- NY Open: 06:00-10:00
- NY Silver Bullet: 10:00-14:00
- NY PM: 14:00-18:00
- Asia Open: 18:00-22:00

## Execution Checklist

For each session:

1. **End-of-candle**: When a key 4H candle closes (NY PM, 22:00, 02:00, 06:00, 10:00), mark its high/low
2. **Next 4H candle**: Watch how the new candle trades relative to those levels:
   - **Wick through level then close back inside** + volume spike → **reversal**
   - **Close outside with strong body** + confirming volume → **continuation**
3. **Bias**: Assign bullish/bearish/neutral bias for that session based on which PCT + VSA scenario the pair of candles (previous + current) matches

## Example Scenarios

### Scenario 1: NY 6am Bearish Reversal
- **Candle 1 (London 2-6am)**: High $95,000, Low $94,500, Close $94,900 (bullish)
- **Candle 2 (NY Open 6-10am)**: Wicks up to $95,050 (above PCH), volume 2.5x London volume, closes at $94,800 (back below PCH)
- **Analysis**: Upthrust/buying climax - volume higher than London, price rejected
- **Signal**: SELL at rejection, target $94,500 (PCL)

### Scenario 2: London Reversal (Candle 3)
- **Candle 1 (Frankfurt 22-2am)**: High $94,800, Low $94,400
- **Candle 2 (London 2-6am)**: High $95,200, Low $94,600, Close $95,100 (trended up)
- **Candle 3 (NY Open 6-10am)**: Wicks to $95,250 (above C2 high), ultra-high volume, closes at $95,000 (back below C2 high)
- **Analysis**: Climactic volume on rejection - London reversal
- **Signal**: SELL - London reversal short

### Scenario 3: Asia Reversal (Candle 2)
- **Candle 1 (NY PM 2-6pm)**: High $95,100, Low $94,700
- **Candle 2 (Asia Open 6-10pm)**: Spikes to $95,150 (above PCH), moderate Asia volume (1.3x average), closes at $94,950 (back inside)
- **Analysis**: Spike above PCH then close back inside - Asia reversal
- **Signal**: SELL - Asia reversal short toward PCL

## Performance Considerations

### Advantages

- **Candle-by-Candle Logic**: Understands market structure across candle sequences
- **Session-Aware**: Tracks which candle in which sequence we're analyzing
- **Liquidity Focus**: Targets PCH/PCL levels where stops are likely clustered
- **Volume Confirmation**: Distinguishes real moves from fakeouts
- **Reversal Detection**: Identifies session reversals early (6am Judas Swing, 10am Silver Bullet, London reversals)

### Limitations

- Requires sufficient historical data (at least 1 completed 4H candle)
- Session detection depends on accurate timestamps
- Asia volume analysis requires separate tracking (naturally lower volume)
- May miss opportunities during consolidation periods
- Complex logic requires careful tracking of candle sequences

## Customization

You can customize the strategy by modifying:

1. **Volume Thresholds**: Adjust ratios for high/ultra-high volume detection
2. **Session Times**: Modify `SESSION_TIMES` dictionary if using different timezone
3. **Candle Sequences**: Adjust `SESSION_GROUPS` if using different session structures
4. **Signal Weights**: Change weights for CVD, VSA, Price Action, and Bias components
5. **Entry Timing**: Adjust when signals trigger (e.g., require more confirmation)

## Troubleshooting

**No signals generated**:
- Check if you have at least 1 completed 4H candle
- Verify session detection is working (check EST timezone conversion)
- Ensure data includes volume information
- Verify candle sequences are being tracked correctly

**All signals are HOLD**:
- Session bias might be NEUTRAL
- Price might not be interacting with PCH/PCL
- Volume might not meet thresholds
- Candle sequence might not be recognized

**Wrong session detected**:
- Verify timestamps are in UTC
- Check timezone conversion (EST/EDT handling)
- Ensure `EST_TZ` is set correctly

**Candle sequences not tracking**:
- Check that `SESSION_GROUPS` dictionary matches your session structure
- Verify session names match between `get_4h_session()` and `SESSION_GROUPS`
- Ensure candles are closing properly (check `is_4h_candle_close()` logic)

## See Also

- `STRATEGY_GUIDE.md` - General strategy development guide
- `example_strategy.py` - Example strategy implementations
- `CONFIG_GUIDE.md` - Configuration options
