# Daily Bias Calendar

An interactive calendar UI that displays daily bias data for cryptocurrencies, color-coded by bias type (bullish, bearish, neutral).

## Features

- 📅 Interactive calendar with month/year navigation
- 🎨 Color-coded days: Green (bullish), Red (bearish), Grey (neutral)
- 🔄 Multi-symbol support with symbol selector
- 📊 Detailed view showing bias and additional data when a date is clicked
- 📱 Responsive design with modern UI

## Setup

1. **Install dependencies:**
   ```bash
   npm install
   ```

2. **Prepare your CSV files:**
   - Place your CSV files in the `public` folder
   - CSV files must contain at least these columns:
     - `symbol` (or `ticker`, `crypto`) - cryptocurrency ticker
     - `date` (or `datetime`, `timestamp`) - date in YYYY-MM-DD format
     - `bias` (or `daily_bias`, `bias_label`) - one of: "bullish", "bearish", "neutral"
   - Additional columns (e.g., open, high, low, close, volume) will be displayed in the details view

3. **Update CSV file paths:**
   - Edit `src/App.tsx` and update the `csvFiles` array with your file paths:
     ```typescript
     const csvFiles = [
       '/your-bias-data.csv',
       // Add more CSV files as needed
     ]
     ```

4. **Run the development server:**
   ```bash
   npm run dev
   ```

5. **Open your browser:**
   - Navigate to the URL shown in the terminal (usually `http://localhost:5173`)

## Building for Production

```bash
npm run build
```

The built files will be in the `dist` folder.

## CSV Format Example

```csv
symbol,date,bias,open,high,low,close,volume
BTC,2023-01-01,bullish,16500,16800,16400,16700,1234567
BTC,2023-01-02,bearish,16700,16750,16500,16550,987654
ETH,2023-01-01,neutral,1200,1220,1190,1210,2345678
```

## Project Structure

```
calendar-app/
├── src/
│   ├── components/
│   │   ├── Calendar.tsx          # Main calendar component
│   │   ├── Calendar.css
│   │   ├── SymbolSelector.tsx    # Symbol selection buttons
│   │   ├── SymbolSelector.css
│   │   ├── DateDetails.tsx       # Date detail display
│   │   └── DateDetails.css
│   ├── utils/
│   │   └── csvParser.ts          # CSV parsing logic
│   ├── App.tsx                   # Main app component
│   ├── App.css
│   ├── main.tsx                  # Entry point
│   └── index.css                 # Global styles
├── public/                        # Place CSV files here
├── package.json
└── README.md
```

## Usage

1. Select a cryptocurrency symbol using the buttons at the top
2. Navigate between months using the left/right arrows
3. Click on any date to see detailed bias information
4. Days are color-coded:
   - 🟢 Green = Bullish bias
   - 🔴 Red = Bearish bias
   - ⚪ Grey = Neutral bias or no data

## Notes

- The calendar automatically determines the date range for each symbol based on the data
- Days outside the data range are shown in grey and are not clickable
- Navigation arrows are disabled when you reach the start/end of available data
- The calendar fills all days in the range, showing neutral grey for dates without data

