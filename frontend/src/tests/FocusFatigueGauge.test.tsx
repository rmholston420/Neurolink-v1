import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import FocusFatigueGauge from '../components/FocusFatigueGauge'

describe('FocusFatigueGauge', () => {
  it('renders focus and fatigue labels', () => {
    render(
      <FocusFatigueGauge focusState="unknown" focusScore={0} fatigueScore={0} />
    )
    expect(screen.getByText(/focus/i)).toBeTruthy()
    expect(screen.getByText(/fatigue/i)).toBeTruthy()
  })

  it('displays high_focus state label', () => {
    render(
      <FocusFatigueGauge focusState="high_focus" focusScore={0.9} fatigueScore={0.05} />
    )
    expect(screen.getByText(/high.?focus/i)).toBeTruthy()
  })

  it('displays low_focus state label', () => {
    render(
      <FocusFatigueGauge focusState="low_focus" focusScore={0.2} fatigueScore={0.1} />
    )
    expect(screen.getByText(/low.?focus/i)).toBeTruthy()
  })

  it('displays unknown state label', () => {
    render(
      <FocusFatigueGauge focusState="unknown" focusScore={0.5} fatigueScore={0.3} />
    )
    expect(screen.getByText(/unknown/i)).toBeTruthy()
  })

  it('renders high fatigue score without crashing', () => {
    const { container } = render(
      <FocusFatigueGauge focusState="high_focus" focusScore={0.5} fatigueScore={0.95} />
    )
    expect(container.firstChild).toBeTruthy()
  })

  it('renders percentage values for focus and fatigue', () => {
    render(
      <FocusFatigueGauge focusState="high_focus" focusScore={0.75} fatigueScore={0.5} />
    )
    expect(screen.getByText(/75%/)).toBeTruthy()
    expect(screen.getByText(/50%/)).toBeTruthy()
  })
})
