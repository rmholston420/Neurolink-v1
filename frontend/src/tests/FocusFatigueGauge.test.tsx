import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import FocusFatigueGauge from '../components/FocusFatigueGauge'

describe('FocusFatigueGauge', () => {
  it('renders focus and fatigue section labels', () => {
    render(
      <FocusFatigueGauge focusState="unknown" focusScore={0} fatigueScore={0} />
    )
    expect(screen.getByText(/focus state/i)).toBeTruthy()
    expect(screen.getByText(/fatigue/i)).toBeTruthy()
  })

  it('displays high_focus state badge', () => {
    render(
      <FocusFatigueGauge focusState="high_focus" focusScore={0.9} fatigueScore={0.05} />
    )
    // Component transforms 'high_focus' -> 'high focus' (replaces underscore with space)
    expect(screen.getByText(/high focus/i)).toBeTruthy()
  })

  it('displays low_focus state badge', () => {
    render(
      <FocusFatigueGauge focusState="low_focus" focusScore={0.2} fatigueScore={0.1} />
    )
    expect(screen.getByText(/low focus/i)).toBeTruthy()
  })

  it('displays unknown state badge', () => {
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

  it('renders fatigue percentage in the label text', () => {
    render(
      <FocusFatigueGauge focusState="high_focus" focusScore={0.75} fatigueScore={0.5} />
    )
    // Fatigue score is rendered inline in the label: 'Fatigue Score:  50 %'
    // The text node contains '50' and '%' with whitespace — use a function matcher
    expect(
      screen.getByText((content) => content.includes('50') && content.includes('%'))
    ).toBeTruthy()
  })
})
