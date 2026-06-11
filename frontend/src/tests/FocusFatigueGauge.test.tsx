import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import FocusFatigueGauge from '../components/FocusFatigueGauge'

describe('FocusFatigueGauge', () => {
  it('renders focus state label', () => {
    render(
      <FocusFatigueGauge
        focusState="HIGH_FOCUS"
        focusScore={0.9}
        fatigueScore={0.1}
      />
    )
    expect(screen.getByText('HIGH FOCUS')).toBeTruthy()
  })

  it('renders fatigue score percentage', () => {
    render(
      <FocusFatigueGauge
        focusState="MODERATE_FOCUS"
        focusScore={0.6}
        fatigueScore={0.5}
      />
    )
    expect(screen.getByText(/Fatigue Score: 50%/)).toBeTruthy()
  })
})
