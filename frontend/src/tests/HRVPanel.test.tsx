import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import HRVPanel from '../components/HRVPanel'

describe('HRVPanel', () => {
  it('renders all three metric labels', () => {
    render(<HRVPanel hrBpm={65} hrv={42.5} rrBpm={14.0} />)
    expect(screen.getByText('Heart Rate')).toBeTruthy()
    expect(screen.getByText('HRV RMSSD')).toBeTruthy()
    expect(screen.getByText('Breathing Rate')).toBeTruthy()
  })

  it('renders numeric values with one decimal place', () => {
    render(<HRVPanel hrBpm={72} hrv={38} rrBpm={16} />)
    expect(screen.getByText('72.0')).toBeTruthy()
    expect(screen.getByText('38.0')).toBeTruthy()
    expect(screen.getByText('16.0')).toBeTruthy()
  })

  it('renders em-dash when values are null', () => {
    render(<HRVPanel hrBpm={null} hrv={null} rrBpm={null} />)
    const dashes = screen.getAllByText('—')
    expect(dashes.length).toBeGreaterThanOrEqual(3)
  })

  it('renders unit labels', () => {
    render(<HRVPanel hrBpm={60} hrv={30} rrBpm={12} />)
    const bpmLabels = screen.getAllByText('bpm')
    expect(bpmLabels.length).toBeGreaterThanOrEqual(2)
    expect(screen.getByText('ms')).toBeTruthy()
  })
})
