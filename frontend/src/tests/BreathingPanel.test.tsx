import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import BreathingPanel from '../components/BreathingPanel'

describe('BreathingPanel', () => {
  it('shows "--" when rrBpm is null', () => {
    render(<BreathingPanel rrBpm={null} />)
    expect(screen.getAllByText('--').length).toBeGreaterThan(0)
  })

  it('renders the rate value when provided', () => {
    render(<BreathingPanel rrBpm={14} />)
    expect(screen.getByText('14')).toBeTruthy()
  })

  it('shows breaths/min unit label', () => {
    render(<BreathingPanel rrBpm={12} />)
    expect(screen.getByText(/breaths\/min/i)).toBeTruthy()
  })

  it('renders gracefully when all optional values are null simultaneously', () => {
    const { container } = render(<BreathingPanel rrBpm={null} />)
    // Component should mount without throwing
    expect(container.firstChild).toBeTruthy()
  })
})
