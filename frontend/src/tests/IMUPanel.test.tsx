import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import IMUPanel from '../components/IMUPanel'

describe('IMUPanel', () => {
  it('renders Pitch, Roll, and Motion RMS labels', () => {
    render(<IMUPanel pitchDeg={5.2} rollDeg={-3.1} motionRms={0.04} />)
    expect(screen.getByText('Pitch')).toBeTruthy()
    expect(screen.getByText('Roll')).toBeTruthy()
    expect(screen.getByText('Motion RMS')).toBeTruthy()
  })

  it('renders pitch and roll with one decimal place', () => {
    render(<IMUPanel pitchDeg={10} rollDeg={-5} motionRms={0.01} />)
    expect(screen.getByText('10.0')).toBeTruthy()
    expect(screen.getByText('-5.0')).toBeTruthy()
  })

  it('renders degree unit symbols', () => {
    render(<IMUPanel pitchDeg={3} rollDeg={2} motionRms={0.02} />)
    const degLabels = screen.getAllByText('\u00b0')
    expect(degLabels.length).toBeGreaterThanOrEqual(2)
  })

  it('renders em-dash when all values are null', () => {
    render(<IMUPanel pitchDeg={null} rollDeg={null} motionRms={null} />)
    const dashes = screen.getAllByText('\u2014')
    expect(dashes.length).toBeGreaterThanOrEqual(2)
  })
})
