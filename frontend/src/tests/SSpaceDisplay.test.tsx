import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import SSpaceDisplay from '../components/SSpaceDisplay'

describe('SSpaceDisplay', () => {
  const defaultProps = {
    region: 'B',
    stage: 'Albedo',
    regionV01: 'A',
    stageV01: 'Nigredo',
  }

  it('renders v2 region and stage', () => {
    render(<SSpaceDisplay {...defaultProps} />)
    expect(screen.getByText('B')).toBeTruthy()
    const albedoBadges = screen.getAllByText('Albedo')
    expect(albedoBadges.length).toBeGreaterThanOrEqual(1)
  })

  it('renders v0.1 region and stage', () => {
    render(<SSpaceDisplay {...defaultProps} />)
    expect(screen.getByText('A')).toBeTruthy()
    const nigredobadges = screen.getAllByText('Nigredo')
    expect(nigredobadges.length).toBeGreaterThanOrEqual(1)
  })

  it('renders section labels', () => {
    render(<SSpaceDisplay {...defaultProps} />)
    expect(screen.getByText('v2 Alchemical')).toBeTruthy()
    expect(screen.getByText('v0.1 S-Space')).toBeTruthy()
  })

  it('handles an unknown stage gracefully without crashing', () => {
    render(
      <SSpaceDisplay
        region="X"
        stage="UnknownStage"
        regionV01="Y"
        stageV01="AnotherUnknown"
      />
    )
    expect(screen.getByText('X')).toBeTruthy()
    expect(screen.getByText('Y')).toBeTruthy()
  })
})
