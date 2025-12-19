import './DateDetails.css'

interface BiasData {
  symbol: string
  date: string
  bias: 'bullish' | 'bearish' | 'neutral'
  [key: string]: any // For extra fields like open, high, low, close, volume, etc.
}

interface DateDetailsProps {
  symbol: string
  date: Date
  biasData: BiasData | null
}

function DateDetails({ symbol, date, biasData }: DateDetailsProps) {
  const formatDate = (date: Date): string => {
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
    })
  }

  const getBiasColor = (bias: string): string => {
    switch (bias) {
      case 'bullish':
        return '#4caf50'
      case 'bearish':
        return '#f44336'
      default:
        return '#9e9e9e'
    }
  }

  const getBiasLabel = (bias: string): string => {
    return bias.charAt(0).toUpperCase() + bias.slice(1)
  }

  // Get extra fields (exclude symbol, date, bias)
  const getExtraFields = (): Array<{ key: string; value: any }> => {
    if (!biasData) return []
    const excluded = ['symbol', 'date', 'bias']
    return Object.entries(biasData)
      .filter(([key]) => !excluded.includes(key))
      .map(([key, value]) => ({ key, value }))
  }

  const extraFields = getExtraFields()

  return (
    <div className="date-details">
      <h3>Daily Bias Details</h3>
      <div className="details-content">
        <div className="detail-row">
          <span className="detail-label">Symbol:</span>
          <span className="detail-value">{symbol}</span>
        </div>
        <div className="detail-row">
          <span className="detail-label">Date:</span>
          <span className="detail-value">{formatDate(date)}</span>
        </div>
        <div className="detail-row">
          <span className="detail-label">Bias:</span>
          <span
            className="detail-value bias-badge"
            style={{
              backgroundColor: getBiasColor(biasData?.bias || 'neutral'),
              color: 'white',
            }}
          >
            {getBiasLabel(biasData?.bias || 'neutral')}
          </span>
        </div>
        {extraFields.length > 0 && (
          <>
            <div className="detail-divider"></div>
            <div className="extra-fields">
              <h4>Additional Data</h4>
              {extraFields.map(({ key, value }) => (
                <div key={key} className="detail-row">
                  <span className="detail-label">
                    {key.charAt(0).toUpperCase() + key.slice(1)}:
                  </span>
                  <span className="detail-value">
                    {typeof value === 'number'
                      ? value.toLocaleString(undefined, {
                          maximumFractionDigits: 2,
                        })
                      : String(value)}
                  </span>
                </div>
              ))}
            </div>
          </>
        )}
        {!biasData && (
          <div className="no-data-message">
            No bias data available for this date.
          </div>
        )}
      </div>
    </div>
  )
}

export default DateDetails

