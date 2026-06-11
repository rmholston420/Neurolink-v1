import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import SSpaceDisplay from '../components/SSpaceDisplay'

describe('SSpaceDisplay', () => {
  it('renders the v2 region label', () => {
    render(
      <SSpaceDisplay region="B" stage="Albedo" regionV01="A" stageV01="Nigredo" />
    )
    expect(screen.getAllByText('B').length).toBeGreaterThan(0)
  })

  it('renders the v2 alchemical stage name', () => {
    render(
      <SSpaceDisplay region="B" stage="Albedo" regionV01="A" stageV01="Nigredo" />
    )
    expect(screen.getByText('Albedo')).toBeTruthy()
  })

  it('renders the v01 section label', () => {
    render(
      <SSpaceDisplay region="B" stage="Albedo" regionV01="C" stageV01="Citrinitas" />
    )
    expect(screen.getByText('Citrinitas')).toBeTruthy()
  })

  it('renders Nigredo stage', () => {
    render(
      <SSpaceDisplay region="A" stage="Nigredo" regionV01="A" stageV01="Nigredo" />
    )
    expect(screen.getAllByText('Nigredo').length).toBeGreaterThan(0)
  })

  it('renders Rubedo stage', () => {
    render(
      <SSpaceDisplay region="D" stage="Rubedo" regionV01="D" stageV01="Rubedo" />
    )
    expect(screen.getAllByText('Rubedo').length).toBeGreaterThan(0)
  })

  it('falls back gracefully for an unknown stage name', () => {
    // Unknown stage should not throw — falls back to #8b949e neutral colour
    const { container } = render(
      <SSpaceDisplay region="Z" stage="UnknownStage" regionV01="Z" stageV01="AlsoUnknown" />
    )
    expect(screen.getByText('UnknownStage')).toBeTruthy()
    expect(container.firstChild).toBeTruthy()
  })
})
