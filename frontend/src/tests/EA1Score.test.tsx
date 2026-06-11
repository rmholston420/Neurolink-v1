import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import EA1Score from '../components/EA1Score'
import type { EA1Result } from '../types'

const makeEA1 = (overrides: Partial<EA1Result> = {}): EA1Result => ({
  eligible: true,
  score: 1.0,
  criteria_met: 5,
  criteria_total: 5,
  label: 'Eligible',
  overlay_mode: 'X5',
  gates: {},
  criteria: {},
  alchemical_stage: 'Rubedo',
  s_space_coords: null,
  s_space_region: 'A',
  integration_coverage: 1.0,
  ...overrides,
})

describe('EA1Score', () => {
  it('renders "No data" when ea1 is null', () => {
    render(<EA1Score ea1={null} />)
    expect(screen.getByText('No data')).toBeTruthy()
  })

  it('shows Eligible badge when eligible is true', () => {
    render(<EA1Score ea1={makeEA1()} />)
    expect(screen.getByText('Eligible')).toBeTruthy()
  })

  it('shows 100% score when score is 1.0', () => {
    render(<EA1Score ea1={makeEA1()} />)
    expect(screen.getByText('100%')).toBeTruthy()
  })

  it('shows ineligible label and partial score when not eligible', () => {
    const ea1 = makeEA1({ eligible: false, score: 0.4, criteria_met: 2, label: 'Not eligible' })
    render(<EA1Score ea1={ea1} />)
    expect(screen.getByText('Not eligible')).toBeTruthy()
    expect(screen.getByText('40%')).toBeTruthy()
  })

  it('shows correct criteria count', () => {
    const ea1 = makeEA1({ criteria_met: 3, criteria_total: 5 })
    render(<EA1Score ea1={ea1} />)
    expect(screen.getByText(/3 \/ 5 criteria met/)).toBeTruthy()
  })
})
