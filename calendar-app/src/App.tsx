import { useState, useEffect } from 'react'
import Calendar from './components/Calendar'
import SymbolSelector from './components/SymbolSelector'
import DateDetails from './components/DateDetails'
import { parseCSVFiles, BiasData, SymbolData } from './utils/csvParser'
import './App.css'

function App() {
  const [allData, setAllData] = useState<Map<string, SymbolData>>(new Map())
  const [selectedSymbol, setSelectedSymbol] = useState<string>('')
  const [selectedDate, setSelectedDate] = useState<Date | null>(null)
  const [selectedBiasData, setSelectedBiasData] = useState<BiasData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    // Load CSV files from public folder
    // Add your CSV files here - they should be placed in the public/ folder
    const csvFiles = [
      '/bias-data.csv', // Exported bias data from backtest
      // '/sample-bias-data.csv', // Sample file for testing (commented out)
    ]

    parseCSVFiles(csvFiles)
      .then((data) => {
        setAllData(data)
        if (data.size > 0) {
          // Set first symbol as default
          const firstSymbol = Array.from(data.keys())[0]
          setSelectedSymbol(firstSymbol)
        }
        setLoading(false)
      })
      .catch((err) => {
        setError(err.message)
        setLoading(false)
      })
  }, [])

  const handleDateSelect = (date: Date, biasData: BiasData | null) => {
    setSelectedDate(date)
    setSelectedBiasData(biasData)
  }

  const handleSymbolChange = (symbol: string) => {
    setSelectedSymbol(symbol)
    setSelectedDate(null)
    setSelectedBiasData(null)
  }

  if (loading) {
    return (
      <div className="app-container">
        <div className="loading">Loading bias data...</div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="app-container">
        <div className="error">
          Error: {error}
          <br />
          <small>Please ensure CSV files are in the public folder with columns: symbol, date, bias</small>
        </div>
      </div>
    )
  }

  if (allData.size === 0) {
    return (
      <div className="app-container">
        <div className="error">
          No data found. Please add CSV files to the public folder.
        </div>
      </div>
    )
  }

  const symbolData = allData.get(selectedSymbol)

  if (!symbolData) {
    return (
      <div className="app-container">
        <div className="error">Symbol data not found</div>
      </div>
    )
  }

  return (
    <div className="app-container">
      <div className="app-header">
        <h1>Daily Bias Calendar</h1>
        <SymbolSelector
          symbols={Array.from(allData.keys())}
          selectedSymbol={selectedSymbol}
          onSymbolChange={handleSymbolChange}
        />
      </div>

      <div className="calendar-wrapper">
        <Calendar
          symbolData={symbolData}
          onDateSelect={handleDateSelect}
          selectedDate={selectedDate}
        />
      </div>

      {selectedDate && (
        <DateDetails
          symbol={selectedSymbol}
          date={selectedDate}
          biasData={selectedBiasData}
        />
      )}
    </div>
  )
}

export default App

