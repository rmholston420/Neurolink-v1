import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import EA1Score from '../components/EA1Score'

describe('EA1Score', () => {
  it('renders "No data" when ea1 is null', () => {
    render(<EA1Score ea1={null} />)
    expect(screen.getByText('No data')).toBeTruthy()
  })

  it('shows Eligible badge when eligible', () => {
    const ea1 = {
      eligible: true,
      score: 1.0,
      criteria_met: 5,
      criteria_total: 5,
      label: 'Eligible',
      overlay_mode: 'X5',
    }
    render(<EA1Score ea1={ea1} />)
    expect(screen.getByText('Eligible')).toBeTruthy()
  })

  it('shows 100% score', () => {
    const ea1 = {
      eligible: true,
      score: 1.0,
      criteria_met: 5,
      criteria_total: 5,
      label: 'Eligible',
      overlay_mode: 'X5',
    }
    render(<EA1Score ea1={ea1} />)
    expect(screen.getByText('100%')).toBeTruthy()
  })
})
