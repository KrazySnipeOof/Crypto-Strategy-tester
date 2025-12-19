import './SymbolSelector.css'

interface SymbolSelectorProps {
  symbols: string[]
  selectedSymbol: string
  onSymbolChange: (symbol: string) => void
}

function SymbolSelector({ symbols, selectedSymbol, onSymbolChange }: SymbolSelectorProps) {
  return (
    <div className="symbol-selector">
      {symbols.map((symbol) => (
        <button
          key={symbol}
          className={`symbol-button ${selectedSymbol === symbol ? 'active' : ''}`}
          onClick={() => onSymbolChange(symbol)}
        >
          {symbol}
        </button>
      ))}
    </div>
  )
}

export default SymbolSelector

