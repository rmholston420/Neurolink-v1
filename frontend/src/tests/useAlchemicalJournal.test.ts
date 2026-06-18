import { describe, it, expect, vi, beforeEach } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import { useAlchemicalJournal } from '../hooks/useAlchemicalJournal'
import type { NeurolinkState } from '../types'

// The hook uses a module-level STORE array, so we need to reset it between tests.
// We do this by deleting all entries before each test via the deleteEntry API.

function makeState(overrides: Partial<NeurolinkState> = {}): NeurolinkState {
  return {
    connected: true, source: 'mock', region: '', alchemical_stage: 'Albedo',
    integration_coverage: 0, engagement_index: 0,
    bands: { alpha: 0, theta: 0, beta: 0, delta: 0, gamma: 0 },
    s_space: null, ea1: {
      eligible: true, score: 0.8, criteria_met: 4, criteria_total: 5,
      label: '', overlay_mode: '', gates: {}, criteria: {},
      alchemical_stage: 'Albedo', s_space_coords: null,
      s_space_region: '', integration_coverage: 0,
    },
    last_ts: 0, frame_count: 0, poor_contact: false,
    region_v01: '', alchemical_stage_v01: '',
    faa: null, fmt: null, hr_bpm: null, hrv_rmssd: null, rr_bpm: null,
    pitch_deg: null, roll_deg: null, motion_rms: null,
    contact_quality: null, focus_state: 'high_focus', focus_score: 0.85,
    fatigue_score: 0.1, fnirs_oxy: null, fnirs_deoxy: null,
    eeg_samples: [], bad_channels: [],
    artifact_rejected: false, artifact_reasons: [],
    artifact_annotations: {}, artifact_correction_plan: '',
    channel_impedances: {}, baseline_phase: null,
    ...overrides,
  }
}

describe('useAlchemicalJournal', () => {
  let cleanup: (() => void) | null = null

  beforeEach(() => {
    // Clear any entries left in the module-level STORE from prior tests
    // by rendering the hook and deleting everything
    const { result, unmount } = renderHook(() => useAlchemicalJournal(null))
    act(() => {
      const ids = result.current.entries.map(e => e.id)
      ids.forEach(id => result.current.deleteEntry(id))
    })
    unmount()
    if (cleanup) { cleanup(); cleanup = null }
  })

  it('starts with empty entries', () => {
    const { result } = renderHook(() => useAlchemicalJournal(null))
    expect(result.current.entries).toHaveLength(0)
  })

  it('addEntry creates an entry with the given text', () => {
    const { result } = renderHook(() => useAlchemicalJournal(makeState()))
    act(() => result.current.addEntry('Test note'))
    expect(result.current.entries).toHaveLength(1)
    expect(result.current.entries[0].text).toBe('Test note')
  })

  it('auto-tags entry with alchemical stage', () => {
    const { result } = renderHook(() =>
      useAlchemicalJournal(makeState({ alchemical_stage: 'Nigredo' }))
    )
    act(() => result.current.addEntry('Darkness note'))
    expect(result.current.entries[0].tags).toContain('Nigredo')
  })

  it('auto-tags entry with EA-1 when eligible', () => {
    const { result } = renderHook(() =>
      useAlchemicalJournal(makeState())
    )
    act(() => result.current.addEntry('EA-1 note'))
    expect(result.current.entries[0].tags).toContain('EA-1')
  })

  it('includes extra tags passed to addEntry', () => {
    const { result } = renderHook(() => useAlchemicalJournal(makeState()))
    act(() => result.current.addEntry('Tagged note', ['custom_tag']))
    expect(result.current.entries[0].tags).toContain('custom_tag')
  })

  it('deleteEntry removes the entry by id', () => {
    const { result } = renderHook(() => useAlchemicalJournal(makeState()))
    act(() => result.current.addEntry('To delete'))
    const id = result.current.entries[0].id
    act(() => result.current.deleteEntry(id))
    expect(result.current.entries).toHaveLength(0)
  })

  it('setQuery filters entries by text', () => {
    const { result } = renderHook(() => useAlchemicalJournal(makeState()))
    act(() => {
      result.current.addEntry('First note about dragons')
      result.current.addEntry('Second note about clouds')
    })
    act(() => result.current.setQuery('dragons'))
    expect(result.current.filtered).toHaveLength(1)
    expect(result.current.filtered[0].text).toContain('dragons')
  })

  it('setQuery filters entries by tag', () => {
    const { result } = renderHook(() =>
      useAlchemicalJournal(makeState({ alchemical_stage: 'Rubedo' }))
    )
    act(() => result.current.addEntry('Red stage note'))
    act(() => result.current.setQuery('rubedo'))
    expect(result.current.filtered).toHaveLength(1)
  })

  it('empty query returns all entries', () => {
    const { result } = renderHook(() => useAlchemicalJournal(makeState()))
    act(() => {
      result.current.addEntry('A')
      result.current.addEntry('B')
    })
    act(() => result.current.setQuery(''))
    expect(result.current.filtered).toHaveLength(2)
  })

  it('records focusScore from state', () => {
    const { result } = renderHook(() =>
      useAlchemicalJournal(makeState({ focus_score: 0.77 }))
    )
    act(() => result.current.addEntry('Focus note'))
    expect(result.current.entries[0].focusScore).toBeCloseTo(0.77)
  })

  it('records ea1Score from state when eligible', () => {
    const { result } = renderHook(() =>
      useAlchemicalJournal(makeState({ ea1: {
        eligible: true, score: 0.91, criteria_met: 5, criteria_total: 5,
        label: '', overlay_mode: '', gates: {}, criteria: {},
        alchemical_stage: 'Albedo', s_space_coords: null,
        s_space_region: '', integration_coverage: 0,
      }}))
    )
    act(() => result.current.addEntry('Score note'))
    expect(result.current.entries[0].ea1Score).toBeCloseTo(0.91)
  })
})
