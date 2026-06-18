import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import ArtifactBadge from '../components/ArtifactBadge'

describe('ArtifactBadge', () => {
  it('renders nothing when disconnected', () => {
    const { container } = render(
      <ArtifactBadge connected={false} artifactRejected={false} artifactReasons={[]} />
    )
    expect(container.firstChild).toBeNull()
  })

  it('renders a green dot for a clean frame', () => {
    const { container } = render(
      <ArtifactBadge connected={true} artifactRejected={false} artifactReasons={[]} />
    )
    const dot = container.firstChild as HTMLElement
    expect(dot).toBeTruthy()
    expect(dot.getAttribute('title')).toBe('Signal clean')
    expect(dot.getAttribute('aria-label')).toBe('EEG signal clean')
  })

  it('renders amber shield with "Artifact" label when frame rejected', () => {
    render(
      <ArtifactBadge connected={true} artifactRejected={true} artifactReasons={[]} />
    )
    expect(screen.getByText('Artifact')).toBeTruthy()
  })

  it('shows human-readable reason in title for amplitude', () => {
    const { container } = render(
      <ArtifactBadge connected={true} artifactRejected={true} artifactReasons={['amplitude']} />
    )
    const shield = container.querySelector('[role="status"]') as HTMLElement
    expect(shield.getAttribute('title')).toBe('Amplitude spike')
  })

  it('shows human-readable reason for motion', () => {
    const { container } = render(
      <ArtifactBadge connected={true} artifactRejected={true} artifactReasons={['motion']} />
    )
    const shield = container.querySelector('[role="status"]') as HTMLElement
    expect(shield.getAttribute('title')).toBe('Head movement')
  })

  it('shows human-readable reason for kurtosis', () => {
    const { container } = render(
      <ArtifactBadge connected={true} artifactRejected={true} artifactReasons={['kurtosis']} />
    )
    const shield = container.querySelector('[role="status"]') as HTMLElement
    expect(shield.getAttribute('title')).toBe('Muscle burst')
  })

  it('falls back to raw reason key for unknown causes', () => {
    const { container } = render(
      <ArtifactBadge connected={true} artifactRejected={true} artifactReasons={['unknown_cause']} />
    )
    const shield = container.querySelector('[role="status"]') as HTMLElement
    expect(shield.getAttribute('title')).toBe('unknown_cause')
  })

  it('joins multiple reasons with middle dot', () => {
    const { container } = render(
      <ArtifactBadge connected={true} artifactRejected={true} artifactReasons={['amplitude', 'motion']} />
    )
    const shield = container.querySelector('[role="status"]') as HTMLElement
    expect(shield.getAttribute('title')).toBe('Amplitude spike \u00b7 Head movement')
  })

  it('title is "Frame rejected" when reasons array is empty but frame is rejected', () => {
    const { container } = render(
      <ArtifactBadge connected={true} artifactRejected={true} artifactReasons={[]} />
    )
    const shield = container.querySelector('[role="status"]') as HTMLElement
    expect(shield.getAttribute('title')).toBe('Frame rejected')
  })

  it('aria-label encodes the reason on rejected frames', () => {
    const { container } = render(
      <ArtifactBadge connected={true} artifactRejected={true} artifactReasons={['kurtosis']} />
    )
    const shield = container.querySelector('[role="status"]') as HTMLElement
    expect(shield.getAttribute('aria-label')).toContain('Muscle burst')
  })
})
