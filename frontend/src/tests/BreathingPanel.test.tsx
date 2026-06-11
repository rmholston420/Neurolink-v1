import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import BreathingPanel from '../components/BreathingPanel'

describe('BreathingPanel', () => {
  it('renders all three metric labels', () => {
    render(<BreathingPanel rrBpm={14.2} rrPpg={13.8} rrAccel={14.6} />)
    expect(screen.getByText('Fused Rate')).toBeTruthy()
    expect(screen.getByText('PPG-derived')).toBeTruthy()
    expect(screen.getByText('Accel-derived')).toBeTruthy()
  })

  it('renders numeric values with one decimal place', () => {
    render(<BreathingPanel rrBpm={12} rrPpg={11} rrAccel={13} />)
    expect(screen.getByText('12.0')).toBeTruthy()
    expect(screen.getByText('11.0')).toBeTruthy()
    expect(screen.getByText('13.0')).toBeTruthy()
  })

  it('renders em-dash when rrBpm is null', () => {
    render(<BreathingPanel rrBpm={null} />)
    const dashes = screen.getAllByText('\u2014')
    expect(dashes.length).toBeGreaterThanOrEqual(1)
  })

  it('renders bpm unit labels', () => {
    render(<BreathingPanel rrBpm={14} rrPpg={13} rrAccel={15} />)
    const bpmLabels = screen.getAllByText('bpm')
    expect(bpmLabels.length).toBeGreaterThanOrEqual(3)
  })
})
