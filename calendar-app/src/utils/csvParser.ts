export interface BiasData {
  symbol: string
  date: string
  bias: 'bullish' | 'bearish' | 'neutral'
  [key: string]: any // For extra fields
}

export interface SymbolData {
  symbol: string
  startDate: Date
  endDate: Date
  biasMap: Map<string, BiasData> // date string -> BiasData
}

/**
 * Parse CSV file and extract bias data
 */
async function parseCSVFile(filePath: string): Promise<BiasData[]> {
  try {
    const response = await fetch(filePath)
    if (!response.ok) {
      throw new Error(`Failed to fetch ${filePath}: ${response.statusText}`)
    }
    const text = await response.text()
    return parseCSVText(text)
  } catch (error) {
    console.error(`Error loading ${filePath}:`, error)
    throw error
  }
}

/**
 * Parse CSV text content
 */
function parseCSVText(csvText: string): BiasData[] {
  const lines = csvText.trim().split('\n')
  if (lines.length < 2) {
    throw new Error('CSV file must have at least a header and one data row')
  }

  // Parse header
  const header = lines[0].split(',').map((h) => h.trim().toLowerCase())
  
  // Find required columns (case-insensitive)
  const symbolIdx = header.findIndex((h) => 
    h === 'symbol' || h === 'ticker' || h === 'crypto'
  )
  const dateIdx = header.findIndex((h) => 
    h === 'date' || h === 'datetime' || h === 'timestamp'
  )
  const biasIdx = header.findIndex((h) => 
    h === 'bias' || h === 'daily_bias' || h === 'bias_label'
  )

  if (symbolIdx === -1 || dateIdx === -1 || biasIdx === -1) {
    throw new Error(
      'CSV must contain columns: symbol (or ticker/crypto), date (or datetime/timestamp), bias (or daily_bias/bias_label)'
    )
  }

  // Parse data rows
  const data: BiasData[] = []
  for (let i = 1; i < lines.length; i++) {
    const values = lines[i].split(',').map((v) => v.trim())
    if (values.length !== header.length) continue

    const symbol = values[symbolIdx]
    const dateStr = values[dateIdx]
    const bias = values[biasIdx].toLowerCase()

    // Validate bias value
    if (!['bullish', 'bearish', 'neutral'].includes(bias)) {
      console.warn(`Invalid bias value: ${bias}, skipping row ${i + 1}`)
      continue
    }

    // Parse date (handle various formats)
    let date: Date
    try {
      // Try ISO format first (YYYY-MM-DD)
      if (dateStr.match(/^\d{4}-\d{2}-\d{2}/)) {
        date = new Date(dateStr.split(' ')[0]) // Take date part if datetime
      } else {
        date = new Date(dateStr)
      }
      if (isNaN(date.getTime())) {
        console.warn(`Invalid date: ${dateStr}, skipping row ${i + 1}`)
        continue
      }
    } catch (e) {
      console.warn(`Error parsing date: ${dateStr}, skipping row ${i + 1}`)
      continue
    }

    // Create bias data object
    const biasData: BiasData = {
      symbol,
      date: date.toISOString().split('T')[0], // Store as YYYY-MM-DD
      bias: bias as 'bullish' | 'bearish' | 'neutral',
    }

    // Add extra fields
    header.forEach((col, idx) => {
      if (idx !== symbolIdx && idx !== dateIdx && idx !== biasIdx) {
        const value = values[idx]
        // Try to parse as number if possible
        const numValue = parseFloat(value)
        biasData[col] = isNaN(numValue) ? value : numValue
      }
    })

    data.push(biasData)
  }

  return data
}

/**
 * Parse multiple CSV files and organize by symbol
 */
export async function parseCSVFiles(
  filePaths: string[]
): Promise<Map<string, SymbolData>> {
  const allData: BiasData[] = []

  // Load all CSV files
  for (const filePath of filePaths) {
    try {
      const data = await parseCSVFile(filePath)
      allData.push(...data)
    } catch (error) {
      console.warn(`Skipping ${filePath}:`, error)
    }
  }

  if (allData.length === 0) {
    throw new Error('No valid data found in any CSV files')
  }

  // Group by symbol
  const symbolMap = new Map<string, BiasData[]>()
  for (const data of allData) {
    if (!symbolMap.has(data.symbol)) {
      symbolMap.set(data.symbol, [])
    }
    symbolMap.get(data.symbol)!.push(data)
  }

  // Create SymbolData for each symbol
  const result = new Map<string, SymbolData>()

  for (const [symbol, dataList] of symbolMap.entries()) {
    // Find earliest and latest dates
    const dates = dataList
      .map((d) => new Date(d.date))
      .filter((d) => !isNaN(d.getTime()))
      .sort((a, b) => a.getTime() - b.getTime())

    if (dates.length === 0) continue

    const startDate = dates[0]
    const endDate = dates[dates.length - 1]

    // Create bias map
    const biasMap = new Map<string, BiasData>()
    for (const data of dataList) {
      biasMap.set(data.date, data)
    }

    result.set(symbol, {
      symbol,
      startDate,
      endDate,
      biasMap,
    })
  }

  return result
}

