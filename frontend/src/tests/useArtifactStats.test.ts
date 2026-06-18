import { describe, it, expect } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import { useArtifactStats } from '../hooks/useArtifactStats'
import type { NeurolinkState } from '../types'

function makeState(overrides: Partial<NeurolinkState> = {}): NeurolinkState {
  return {
    connected: true,
    source: 'mock', region: '', alchemical_stage: '', integration_coverage: 0,
    engagement_index: 0, bands: { alpha: 0, theta: 0, beta: 0, delta: 0, gamma: 0 },
    s_space: null, ea1: {
      eligible: false, score: 0, criteria_met: 0, criteria_total: 0,
      label: '', overlay_mode: '', gates: {}, criteria: {},
      alchemical_stage: '', s_space_coords: null, s_space_region: '', integration_coverage: 0,
    },
    last_ts: 0, frame_count: 0, poor_contact: false,
    region_v01: '', alchemical_stage_v01: '',
    faa: null, fmt: null, hr_bpm: null, hrv_rmssd: null, rr_bpm: null,
    pitch_deg: null, roll_deg: null, motion_rms: null,
    contact_quality: null, focus_state: '', focus_score: 0, fatigue_score: 0,
    fnirs_oxy: null, fnirs_deoxy: null,
    eeg_samples: [], bad_channels: [],
    artifact_rejected: false, artifact_reasons: [],
    artifact_annotations: {}, artifact_correction_plan: '',
    channel_impedances: {}, baseline_phase: null,
    ...overrides,
  }
}

describe('useArtifactStats', () => {
  it('initialises with all-zero stats', () => {
    const { result } = renderHook(() => useArtifactStats(null))
    expect(result.current.totalFrames).toBe(0)
    expect(result.current.rejectedFrames).toBe(0)
    expect(result.current.rejectRate).toBe(0)
  })

  it('accumulates a clean frame', () => {
    const state = makeState({ frame_count: 1, artifact_rejected: false })
    const { result } = renderHook(() => useArtifactStats(state))
    expect(result.current.totalFrames).toBe(1)
    expect(result.current.rejectedFrames).toBe(0)
    expect(result.current.rejectRate).toBe(0)
  })

  it('accumulates a rejected frame and counts its reasons', () => {
    const state = makeState({
      frame_count: 1, artifact_rejected: true, artifact_reasons: ['amplitude'],
    })
    const { result } = renderHook(() => useArtifactStats(state))
    expect(result.current.rejectedFrames).toBe(1)
    expect(result.current.causeCounts['amplitude']).toBe(1)
  })

  it('does not double-count when state re-renders with same frame_count', () => {
    const state = makeState({ frame_count: 5, artifact_rejected: true, artifact_reasons: ['motion'] })
    const { result, rerender } = renderHook(({ s }) => useArtifactStats(s), {
      initialProps: { s: state },
    })
    // Re-render without changing frame_count
    rerender({ s: { ...state } })
    expect(result.current.totalFrames).toBe(1)
  })

  it('computes reject rate correctly', () => {
    let frame = 0
    // Render 4 frames: 1 rejected
    const states = [
      makeState({ frame_count: ++frame, artifact_rejected: false }),
      makeState({ frame_count: ++frame, artifact_rejected: true,  artifact_reasons: ['kurtosis'] }),
      makeState({ frame_count: ++frame, artifact_rejected: false }),
      makeState({ frame_count: ++frame, artifact_rejected: false }),
    ]
    let s = states[0]
    const { result, rerender } = renderHook(({ s }) => useArtifactStats(s), { initialProps: { s } })
    for (const next of states.slice(1)) {
      rerender({ s: next })
    }
    expect(result.current.rejectRate).toBeCloseTo(0.25)
  })

  it('reset() clears all accumulators', () => {
    const state = makeState({ frame_count: 1, artifact_rejected: true, artifact_reasons: ['amplitude'] })
    const { result } = renderHook(() => useArtifactStats(state))
    act(() => result.current.reset())
    expect(result.current.totalFrames).toBe(0)
    expect(result.current.rejectedFrames).toBe(0)
    expect(result.current.causeCounts).toEqual({})
  })

  it('does not accumulate when state is null', () => {
    const { result } = renderHook(() => useArtifactStats(null))
    expect(result.current.totalFrames).toBe(0)
  })

  it('does not accumulate when connected is false', () => {
    const state = makeState({ connected: false, frame_count: 1 })
    const { result } = renderHook(() => useArtifactStats(state))
    expect(result.current.totalFrames).toBe(0)
  })
})
