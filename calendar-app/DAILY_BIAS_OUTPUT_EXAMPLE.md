# Daily Bias Output Example

This document shows how the daily bias details are displayed when a user clicks on a date in the calendar.

## Example Output

When a user clicks on a date (e.g., January 15, 2023 for BTC), the following section appears below the calendar:

---

### Daily Bias Details

**Symbol:** BTC  
**Date:** January 15, 2023  
**Bias:** 🟢 **BULLISH**

**Additional Data:**
- **Open:** 16,500.00
- **High:** 16,800.00
- **Low:** 16,400.00
- **Close:** 16,700.00
- **Volume:** 1,234,567.00
- **Scenario:** 1

---

## Visual Representation

```
┌─────────────────────────────────────────┐
│         Daily Bias Details               │
├─────────────────────────────────────────┤
│ Symbol:        BTC                       │
│ Date:          January 15, 2023          │
│ Bias:          [BULLISH] (green badge)   │
│                                          │
│ ─────────────────────────────────────    │
│                                          │
│ Additional Data                          │
│ Open:           16,500.00                │
│ High:           16,800.00                │
│ Low:            16,400.00                │
│ Close:          16,700.00                │
│ Volume:         1,234,567.00             │
│ Scenario:       1                        │
└─────────────────────────────────────────┘
```

## Behavior

1. **When a date with data is clicked:**
   - Shows symbol, formatted date, and bias (with color-coded badge)
   - Displays all additional CSV columns (open, high, low, close, volume, scenario, etc.)
   - Numbers are formatted with thousand separators
   - Bias badge is color-coded: Green (bullish), Red (bearish), Grey (neutral)

2. **When a date without data is clicked:**
   - Shows symbol and formatted date
   - Displays "No bias data available for this date" message
   - Bias is shown as "Neutral" (grey)

3. **Styling:**
   - Smooth slide-in animation when details appear
   - Clean, card-based layout
   - Color-coded bias badge matching calendar day colors
   - Responsive design that works on all screen sizes

## Code Location

The output is rendered by the `DateDetails` component located at:
- `src/components/DateDetails.tsx`
- `src/components/DateDetails.css`

