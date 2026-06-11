import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import BreathingPanel from '../components/BreathingPanel'

describe('BreathingPanel', () => {
  it('renders em-dash placeholder when rrBpm is null', () => {
    const { container } = render(<BreathingPanel rrBpm={null} />)
    // Component renders \u2014 (em-dash) for null values, not '--'
    expect(container.textContent).toContain('\u2014')
  })

  it('renders the Fused Rate row label', () => {
    render(<BreathingPanel rrBpm={14} />)
    expect(screen.getByText('Fused Rate')).toBeTruthy()
  })

  it('renders the numeric value as a decimal string', () => {
    render(<BreathingPanel rrBpm={14} />)
    // Component renders toFixed(1) — produces '14.0'
    expect(screen.getByText('14.0')).toBeTruthy()
  })

  it('renders bpm unit label', () => {
    render(<BreathingPanel rrBpm={12} />)
    // Unit label is 'bpm', not 'breaths/min'
    expect(screen.getAllByText('bpm').length).toBeGreaterThan(0)
  })

  it('renders gracefully when all optional values are null simultaneously', () => {
    const { container } = render(<BreathingPanel rrBpm={null} />)
    expect(container.firstChild).toBeTruthy()
  })
})
