import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import ArtifactStatsPanel from '../components/ArtifactStatsPanel'
import type { ArtifactStats } from '../hooks/useArtifactStats'

function makeStats(overrides: Partial<ArtifactStats> = {}): ArtifactStats {
  return {
    totalFrames:    0,
    rejectedFrames: 0,
    rejectRate:     0,
    causeCounts:    {},
    windowSize:     300,
    reset:          vi.fn(),
    ...overrides,
  }
}

describe('ArtifactStatsPanel', () => {
  it('shows placeholder when disconnected', () => {
    render(<ArtifactStatsPanel stats={makeStats()} connected={false} />)
    expect(screen.getByText(/Connect a device/i)).toBeTruthy()
  })

  it('renders rejection rate as percentage', () => {
    render(
      <ArtifactStatsPanel
        stats={makeStats({ rejectRate: 0.125, totalFrames: 8, rejectedFrames: 1 })}
        connected={true}
      />
    )
    expect(screen.getByText('12.5%')).toBeTruthy()
  })

  it('shows frame count in footer', () => {
    render(
      <ArtifactStatsPanel
        stats={makeStats({ totalFrames: 20, rejectedFrames: 4, rejectRate: 0.2 })}
        connected={true}
      />
    )
    expect(screen.getByText(/4 \/ 20 frames rejected/)).toBeTruthy()
  })

  it('renders cause pills when causeCounts is non-empty', () => {
    render(
      <ArtifactStatsPanel
        stats={makeStats({
          rejectRate: 0.3,
          totalFrames: 10,
          rejectedFrames: 3,
          causeCounts: { amplitude: 2, motion: 1 },
        })}
        connected={true}
      />
    )
    expect(screen.getByText('Amplitude spike')).toBeTruthy()
    expect(screen.getByText('Head movement')).toBeTruthy()
  })

  it('renders all-clean confirmation row when no causes and frames exist', () => {
    render(
      <ArtifactStatsPanel
        stats={makeStats({ totalFrames: 5, rejectedFrames: 0, rejectRate: 0, causeCounts: {} })}
        connected={true}
      />
    )
    expect(screen.getByText(/No artifacts detected/i)).toBeTruthy()
  })

  it('calls reset when Reset button clicked', () => {
    const resetFn = vi.fn()
    render(
      <ArtifactStatsPanel
        stats={makeStats({ reset: resetFn })}
        connected={true}
      />
    )
    fireEvent.click(screen.getByRole('button', { name: /Reset/i }))
    expect(resetFn).toHaveBeenCalledOnce()
  })

  it('window size appears in label', () => {
    render(
      <ArtifactStatsPanel
        stats={makeStats({ windowSize: 150 })}
        connected={true}
      />
    )
    expect(screen.getByText(/last 150 frames/i)).toBeTruthy()
  })
})
