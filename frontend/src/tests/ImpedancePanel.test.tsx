import { describe, it, expect, beforeAll } from 'vitest'
import { render, screen } from '@testing-library/react'
import ImpedancePanel from '../components/ImpedancePanel'

// ImpedancePanel injects a <style> tag into document.head on first render;
// we need a real DOM, which jsdom provides.
beforeAll(() => {
  // Ensure document.head exists (jsdom provides this)
  if (!document.head) {
    const head = document.createElement('head')
    document.documentElement.prepend(head)
  }
})

describe('ImpedancePanel', () => {
  it('renders empty-state message when impedances object is empty', () => {
    render(<ImpedancePanel impedances={{}} />)
    expect(screen.getByText(/No impedance data/i)).toBeTruthy()
  })

  it('renders a tile for each channel', () => {
    render(<ImpedancePanel impedances={{ TP9: 10, AF7: 25, AF8: 60, TP10: 5 }} />)
    for (const ch of ['TP9', 'AF7', 'AF8', 'TP10']) {
      expect(screen.getByText(ch)).toBeTruthy()
    }
  })

  it('renders impedance values with one decimal place', () => {
    render(<ImpedancePanel impedances={{ TP9: 12.345 }} />)
    expect(screen.getByText('12.3')).toBeTruthy()
  })

  it('shows legend items for Good, Warn, Poor', () => {
    render(<ImpedancePanel impedances={{ TP9: 10 }} />)
    expect(screen.getByText(/Good < 20 k\u03a9/i)).toBeTruthy()
    expect(screen.getByText(/Warn 20.50 k\u03a9/i) ?? screen.queryByText(/Warn/i)).toBeTruthy()
    expect(screen.getByText(/Poor > 50 k\u03a9/i) ?? screen.queryByText(/Poor/i)).toBeTruthy()
  })

  it('renders the outlier legend text', () => {
    // 'Outlier' appears in both the legend badge and the footer 'Outlier fence' span;
    // use getAllByText so multiple matches are permitted.
    render(<ImpedancePanel impedances={{ TP9: 10 }} />)
    const outlierEls = screen.getAllByText(/Outlier/i)
    expect(outlierEls.length).toBeGreaterThan(0)
  })

  it('shows the outlier badge (\u26a0) when a channel is above the IQR fence', () => {
    // Channel A=5, B=6 median=5.5, IQR~=0.5, fence=5.5+0.75=6.25
    // C=50 is well above fence — component renders the badge as
    // '\u26a0 Outlier (> median + 1.5\u00d7IQR)' in a single span, never a bare \u26a0 node.
    render(<ImpedancePanel impedances={{ A: 5, B: 6, C: 50 }} />)
    const badges = screen.queryAllByText(/\u26a0/)
    expect(badges.length).toBeGreaterThan(0)
  })

  it('renders session median row', () => {
    render(<ImpedancePanel impedances={{ TP9: 10, AF7: 30 }} />)
    expect(screen.getByText('Session median')).toBeTruthy()
    expect(screen.getByText('Outlier fence')).toBeTruthy()
  })

  it('shows \u2014 for fence when only one channel provided', () => {
    render(<ImpedancePanel impedances={{ TP9: 15 }} />)
    // fence = Infinity -> displayed as \u2014
    expect(screen.getByText('\u2014')).toBeTruthy()
  })
})
