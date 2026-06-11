import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import BandPowerChart from '../components/BandPowerChart'

describe('BandPowerChart', () => {
  it('renders "No data" when bands is null', () => {
    render(<BandPowerChart bands={null} />)
    expect(screen.getByText('No data')).toBeTruthy()
  })

  it('renders all 5 band labels', () => {
    const bands = { alpha: 0.3, theta: 0.2, beta: 0.15, delta: 0.25, gamma: 0.1 }
    render(<BandPowerChart bands={bands} />)
    expect(screen.getByText(/Alpha/i)).toBeTruthy()
    expect(screen.getByText(/Theta/i)).toBeTruthy()
    expect(screen.getByText(/Beta/i)).toBeTruthy()
    expect(screen.getByText(/Delta/i)).toBeTruthy()
    expect(screen.getByText(/Gamma/i)).toBeTruthy()
  })

  it('renders percentage values', () => {
    const bands = { alpha: 0.3, theta: 0.2, beta: 0.15, delta: 0.25, gamma: 0.1 }
    render(<BandPowerChart bands={bands} />)
    expect(screen.getByText('30.0%')).toBeTruthy()
  })
})
