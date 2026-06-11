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

  it('shows Eligible badge when eligible', () => {
    render(<EA1Score ea1={makeEA1()} />)
    expect(screen.getByText('Eligible')).toBeTruthy()
  })

  it('shows 100% score', () => {
    render(<EA1Score ea1={makeEA1()} />)
    expect(screen.getByText('100%')).toBeTruthy()
  })
})
