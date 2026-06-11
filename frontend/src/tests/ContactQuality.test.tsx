import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import ContactQuality from '../components/ContactQuality'

describe('ContactQuality', () => {
  it('shows Good Contact when poorContact is false', () => {
    render(<ContactQuality poorContact={false} contactQuality={98} />)
    expect(screen.getByText(/good contact/i)).toBeTruthy()
  })

  it('shows Poor Contact when poorContact is true', () => {
    render(<ContactQuality poorContact={true} contactQuality={10} />)
    expect(screen.getByText(/poor contact/i)).toBeTruthy()
  })

  it('renders when contactQuality is exactly 0', () => {
    // 0 is falsy — component must handle it without crashing
    const { container } = render(<ContactQuality poorContact={false} contactQuality={0} />)
    expect(container.firstChild).toBeTruthy()
  })

  it('renders without crashing when contactQuality is null', () => {
    // Component shows only the good/poor badge when quality is null — no '--' placeholder
    const { container } = render(<ContactQuality poorContact={false} contactQuality={null} />)
    expect(container.firstChild).toBeTruthy()
    expect(screen.getByText(/good contact/i)).toBeTruthy()
  })
})
