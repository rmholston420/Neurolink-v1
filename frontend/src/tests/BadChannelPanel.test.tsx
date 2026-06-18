import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import BadChannelPanel from '../components/BadChannelPanel'

describe('BadChannelPanel', () => {
  it('shows all-good confirmation when no bad channels', () => {
    render(<BadChannelPanel badChannels={[]} />)
    expect(screen.getByText(/All channels active/i)).toBeTruthy()
  })

  it('renders all 4 default Muse channels as pills', () => {
    render(<BadChannelPanel badChannels={[]} />)
    for (const ch of ['TP9', 'AF7', 'AF8', 'TP10']) {
      expect(screen.getByText(ch)).toBeTruthy()
    }
  })

  it('shows amber banner when bad channels present', () => {
    render(<BadChannelPanel badChannels={['AF7']} />)
    expect(screen.getByText(/Stage 2 spherical-spline interpolation/i)).toBeTruthy()
  })

  it('lists bad channel names in banner', () => {
    render(<BadChannelPanel badChannels={['TP9', 'TP10']} />)
    expect(screen.getByText(/TP9.*TP10|TP10.*TP9/)).toBeTruthy()
  })

  it('summary shows correct good/total count', () => {
    render(<BadChannelPanel badChannels={['AF7']} />)
    // 3 good of 4 total
    expect(screen.getByText(/3 \/ 4 channels active/)).toBeTruthy()
  })

  it('summary shows flagged count when bad channels exist', () => {
    render(<BadChannelPanel badChannels={['AF7', 'TP9']} />)
    expect(screen.getByText(/2 flagged/)).toBeTruthy()
  })

  it('respects custom allChannels prop', () => {
    render(<BadChannelPanel badChannels={[]} allChannels={['Fz', 'Cz', 'Pz']} />)
    expect(screen.getByText('Fz')).toBeTruthy()
    expect(screen.getByText('Cz')).toBeTruthy()
    expect(screen.getByText('Pz')).toBeTruthy()
  })

  it('does not show the bad-channel banner when all channels are good', () => {
    render(<BadChannelPanel badChannels={[]} />)
    expect(screen.queryByText(/spherical-spline/i)).toBeNull()
  })

  it('is case-insensitive when matching bad channels', () => {
    // badChannels uses lowercase; allChannels uses uppercase
    render(<BadChannelPanel badChannels={['af7']} allChannels={['AF7', 'TP9']} />)
    expect(screen.getByText(/1 flagged/)).toBeTruthy()
  })
})
