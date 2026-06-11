import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import CalibrationPanel from '../components/CalibrationPanel'

describe('CalibrationPanel', () => {
  it('renders the Start Calibration button', () => {
    render(<CalibrationPanel apiUrl="http://localhost:8000" />)
    expect(screen.getByText('Start Calibration')).toBeTruthy()
  })

  it('shows initial idle status message', () => {
    render(<CalibrationPanel apiUrl="http://localhost:8000" />)
    expect(screen.getByText('Not calibrated')).toBeTruthy()
  })

  it('renders the description text', () => {
    render(<CalibrationPanel apiUrl="http://localhost:8000" />)
    expect(screen.getByText(/30-second personal alpha baseline/i)).toBeTruthy()
  })
})
