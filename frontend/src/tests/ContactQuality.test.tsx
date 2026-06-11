import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import ContactQuality from '../components/ContactQuality'

describe('ContactQuality', () => {
  it('shows "Good Contact" when poorContact is false', () => {
    render(<ContactQuality poorContact={false} contactQuality={0.9} />)
    expect(screen.getByText('Good Contact')).toBeTruthy()
  })

  it('shows "Poor Contact" when poorContact is true', () => {
    render(<ContactQuality poorContact={true} contactQuality={0.2} />)
    expect(screen.getByText('Poor Contact')).toBeTruthy()
  })

  it('renders quality percentage when contactQuality is provided', () => {
    render(<ContactQuality poorContact={false} contactQuality={0.75} />)
    expect(screen.getByText('Quality: 75%')).toBeTruthy()
  })

  it('does not render quality bar when contactQuality is null', () => {
    render(<ContactQuality poorContact={false} contactQuality={null} />)
    expect(screen.queryByText(/Quality:/)).toBeNull()
  })
})
