import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import BandPowerChart from '../components/BandPowerChart'
import type { BandPowers } from '../types'

describe('BandPowerChart', () => {
  it('renders "No data" when bands is null', () => {
    render(<BandPowerChart bands={null} />)
    expect(screen.getByText('No data')).toBeTruthy()
  })

  it('renders all 5 band labels when bands are present', () => {
    const bands: BandPowers = { delta: 0.1, theta: 0.2, alpha: 0.3, beta: 0.2, gamma: 0.2 }
    render(<BandPowerChart bands={bands} />)
    expect(screen.getByText(/Alpha/i)).toBeTruthy()
    expect(screen.getByText(/Theta/i)).toBeTruthy()
    expect(screen.getByText(/Beta/i)).toBeTruthy()
    expect(screen.getByText(/Delta/i)).toBeTruthy()
    expect(screen.getByText(/Gamma/i)).toBeTruthy()
  })

  it('renders without crashing when all band values are zero', () => {
    const bands: BandPowers = { delta: 0, theta: 0, alpha: 0, beta: 0, gamma: 0 }
    const { container } = render(<BandPowerChart bands={bands} />)
    expect(container.querySelector('[style*="flex"]')).toBeTruthy()
    expect(screen.queryByText('No data')).toBeNull()
  })

  it('falls back to raw key name for an unrecognised band', () => {
    // Cast lets us slip an extra key past TypeScript to exercise the
    // BAND_LABELS[band] ?? band fallback on lines 49-55
    const bands = { delta: 0.1, theta: 0.1, alpha: 0.1, beta: 0.1, gamma: 0.1, unknown_band: 0.2 } as unknown as BandPowers
    render(<BandPowerChart bands={bands} />)
    // The label falls back to the raw key when not found in BAND_LABELS
    expect(screen.getByText('unknown_band')).toBeTruthy()
  })
})
