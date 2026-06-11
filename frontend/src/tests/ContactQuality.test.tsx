import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import ContactQuality from '../components/ContactQuality'

describe('ContactQuality', () => {
  it('shows Good Contact when poorContact is false', () => {
    render(<ContactQuality poorContact={false} contactQuality={98} />)
    expect(screen.getByText(/good|ok/i)).toBeTruthy()
  })

  it('shows Poor Contact when poorContact is true', () => {
    render(<ContactQuality poorContact={true} contactQuality={10} />)
    expect(screen.getByText(/poor/i)).toBeTruthy()
  })

  it('renders when contactQuality is exactly 0', () => {
    // 0 is falsy — component must handle it without falling back to "--"
    const { container } = render(<ContactQuality poorContact={false} contactQuality={0} />)
    expect(container.firstChild).toBeTruthy()
    // Should show 0, not "--"
    expect(screen.queryByText('--')).toBeNull()
  })

  it('renders when contactQuality is null', () => {
    render(<ContactQuality poorContact={false} contactQuality={null} />)
    // Null should produce a placeholder, not crash
    expect(screen.getAllByText('--').length).toBeGreaterThan(0)
  })
})
