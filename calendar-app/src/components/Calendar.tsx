import { useState } from 'react'
import { SymbolData, BiasData } from '../utils/csvParser'
import './Calendar.css'

interface CalendarProps {
  symbolData: SymbolData
  onDateSelect: (date: Date, biasData: BiasData | null) => void
  selectedDate: Date | null
}

function Calendar({ symbolData, onDateSelect, selectedDate }: CalendarProps) {
  const [currentMonth, setCurrentMonth] = useState(new Date())

  const { startDate, endDate, biasMap } = symbolData

  // Get first day of month and number of days
  const firstDayOfMonth = new Date(
    currentMonth.getFullYear(),
    currentMonth.getMonth(),
    1
  )
  const lastDayOfMonth = new Date(
    currentMonth.getFullYear(),
    currentMonth.getMonth() + 1,
    0
  )

  // Get the day of week for the first day (0 = Sunday, 1 = Monday, etc.)
  let firstDayWeekday = firstDayOfMonth.getDay()
  // Convert to Monday = 0 format
  firstDayWeekday = firstDayWeekday === 0 ? 6 : firstDayWeekday - 1

  const daysInMonth = lastDayOfMonth.getDate()

  // Navigation functions
  const goToPreviousMonth = () => {
    setCurrentMonth(
      new Date(currentMonth.getFullYear(), currentMonth.getMonth() - 1, 1)
    )
  }

  const goToNextMonth = () => {
    setCurrentMonth(
      new Date(currentMonth.getFullYear(), currentMonth.getMonth() + 1, 1)
    )
  }

  const goToPreviousYear = () => {
    setCurrentMonth(
      new Date(currentMonth.getFullYear() - 1, currentMonth.getMonth(), 1)
    )
  }

  const goToNextYear = () => {
    setCurrentMonth(
      new Date(currentMonth.getFullYear() + 1, currentMonth.getMonth(), 1)
    )
  }

  const goToYear = (year: number) => {
    setCurrentMonth(
      new Date(year, currentMonth.getMonth(), 1)
    )
  }

  // Get available years from data range
  const getAvailableYears = (): number[] => {
    const years: number[] = []
    const startYear = startDate.getFullYear()
    const endYear = endDate.getFullYear()
    for (let year = startYear; year <= endYear; year++) {
      years.push(year)
    }
    return years
  }

  // Check if navigation buttons should be disabled
  const canGoPrevious = () => {
    const prevMonth = new Date(
      currentMonth.getFullYear(),
      currentMonth.getMonth() - 1,
      1
    )
    return prevMonth >= new Date(startDate.getFullYear(), startDate.getMonth(), 1)
  }

  const canGoNext = () => {
    const nextMonth = new Date(
      currentMonth.getFullYear(),
      currentMonth.getMonth() + 1,
      1
    )
    return nextMonth <= new Date(endDate.getFullYear(), endDate.getMonth(), 1)
  }

  const canGoPreviousYear = () => {
    const prevYear = new Date(
      currentMonth.getFullYear() - 1,
      currentMonth.getMonth(),
      1
    )
    return prevYear >= new Date(startDate.getFullYear(), startDate.getMonth(), 1)
  }

  const canGoNextYear = () => {
    const nextYear = new Date(
      currentMonth.getFullYear() + 1,
      currentMonth.getMonth(),
      1
    )
    return nextYear <= new Date(endDate.getFullYear(), endDate.getMonth(), 1)
  }

  // Get date key for lookup
  const getDateKey = (date: Date): string => {
    return date.toISOString().split('T')[0]
  }

  // Check if date is in range
  const isDateInRange = (day: number): boolean => {
    const date = new Date(
      currentMonth.getFullYear(),
      currentMonth.getMonth(),
      day
    )
    return date >= startDate && date <= endDate
  }

  // Get bias for a specific date
  const getBiasForDate = (day: number): BiasData | null => {
    const date = new Date(
      currentMonth.getFullYear(),
      currentMonth.getMonth(),
      day
    )
    const dateKey = getDateKey(date)
    return biasMap.get(dateKey) || null
  }

  // Check if date is selected
  const isDateSelected = (day: number): boolean => {
    if (!selectedDate) return false
    const date = new Date(
      currentMonth.getFullYear(),
      currentMonth.getMonth(),
      day
    )
    return (
      date.getDate() === selectedDate.getDate() &&
      date.getMonth() === selectedDate.getMonth() &&
      date.getFullYear() === selectedDate.getFullYear()
    )
  }

  // Handle date click
  const handleDateClick = (day: number) => {
    const date = new Date(
      currentMonth.getFullYear(),
      currentMonth.getMonth(),
      day
    )
    if (isDateInRange(day)) {
      const biasData = getBiasForDate(day)
      onDateSelect(date, biasData)
    }
  }

  // Render calendar cells
  const renderCalendarDays = () => {
    const cells = []

    // Empty cells for days before month starts
    for (let i = 0; i < firstDayWeekday; i++) {
      cells.push(
        <div key={`empty-${i}`} className="calendar-day empty"></div>
      )
    }

    // Days of the month
    for (let day = 1; day <= daysInMonth; day++) {
      const inRange = isDateInRange(day)
      const biasData = inRange ? getBiasForDate(day) : null
      const bias = biasData?.bias || 'neutral'
      const selected = isDateSelected(day)

      cells.push(
        <div
          key={day}
          className={`calendar-day ${bias} ${inRange ? 'in-range' : 'out-of-range'} ${selected ? 'selected' : ''}`}
          onClick={() => handleDateClick(day)}
          title={
            inRange && biasData
              ? `${getDateKey(new Date(currentMonth.getFullYear(), currentMonth.getMonth(), day))}: ${bias}`
              : ''
          }
        >
          {day}
        </div>
      )
    }

    return cells
  }

  const monthNames = [
    'January',
    'February',
    'March',
    'April',
    'May',
    'June',
    'July',
    'August',
    'September',
    'October',
    'November',
    'December',
  ]

  const weekdays = ['Mo', 'Tu', 'We', 'Th', 'Fr', 'Sa', 'Su']

  const availableYears = getAvailableYears()

  return (
    <div className="calendar">
      <div className="calendar-header">
        <div className="nav-group">
          <button
            className="nav-button"
            onClick={goToPreviousMonth}
            disabled={!canGoPrevious()}
            title="Previous month"
          >
            ←
          </button>
          <div className="month-year">
            {monthNames[currentMonth.getMonth()]} {currentMonth.getFullYear()}
          </div>
          <button
            className="nav-button"
            onClick={goToNextMonth}
            disabled={!canGoNext()}
            title="Next month"
          >
            →
          </button>
        </div>
        <div className="year-navigation">
          <button
            className="nav-button year-nav"
            onClick={goToPreviousYear}
            disabled={!canGoPreviousYear()}
            title="Previous year"
          >
            «
          </button>
          <select
            className="year-selector"
            value={currentMonth.getFullYear()}
            onChange={(e) => goToYear(parseInt(e.target.value))}
            title="Select year"
          >
            {availableYears.map((year) => (
              <option key={year} value={year}>
                {year}
              </option>
            ))}
          </select>
          <button
            className="nav-button year-nav"
            onClick={goToNextYear}
            disabled={!canGoNextYear()}
            title="Next year"
          >
            »
          </button>
        </div>
      </div>

      <div className="calendar-weekdays">
        {weekdays.map((day) => (
          <div key={day} className="weekday">
            {day}
          </div>
        ))}
      </div>

      <div className="calendar-grid">{renderCalendarDays()}</div>

      <div className="calendar-legend">
        <div className="legend-item">
          <div className="legend-color bullish"></div>
          <span>Bullish</span>
        </div>
        <div className="legend-item">
          <div className="legend-color bearish"></div>
          <span>Bearish</span>
        </div>
        <div className="legend-item">
          <div className="legend-color neutral"></div>
          <span>Neutral</span>
        </div>
      </div>
    </div>
  )
}

export default Calendar

