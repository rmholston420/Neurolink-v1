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
    // Should still render the band label rows, not the "No data" fallback
    expect(container.querySelector('[style*="flex"]')).toBeTruthy()
    expect(screen.queryByText('No data')).toBeNull()
  })
})
