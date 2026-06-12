import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import { useNeurolinkSSE, isSettling, isBaselineComplete } from '../hooks/useNeurolinkSSE'
import type { NeurolinkState } from '../types'

// ---------------------------------------------------------------------------
// Minimal EventSource mock
// ---------------------------------------------------------------------------
type ESHandler = (event: MessageEvent) => void

class MockEventSource {
  static instances: MockEventSource[] = []
  url: string
  listeners: Record<string, ESHandler[]> = {}
  onmessage: ESHandler | null = null
  onerror: (() => void) | null = null
  closed = false

  constructor(url: string) {
    this.url = url
    MockEventSource.instances.push(this)
  }

  addEventListener(type: string, handler: ESHandler) {
    this.listeners[type] = this.listeners[type] ?? []
    this.listeners[type].push(handler)
  }

  removeEventListener(type: string, handler: ESHandler) {
    this.listeners[type] = (this.listeners[type] ?? []).filter(h => h !== handler)
  }

  close() { this.closed = true }

  /** Dispatch a named SSE event (event: <type>). */
  emit(type: string, data: string) {
    const evt = { data } as MessageEvent
    ;(this.listeners[type] ?? []).forEach(h => h(evt))
  }

  /** Dispatch via the generic onmessage fallback (no event: field). */
  emitMessage(data: string) {
    if (this.onmessage) this.onmessage({ data } as MessageEvent)
  }

  triggerError() {
    if (this.onerror) this.onerror()
  }
}

// ---------------------------------------------------------------------------
// Test fixtures
// ---------------------------------------------------------------------------

const minState = (): NeurolinkState => ({
  connected: true, source: 'mock', region: 'A', alchemical_stage: 'Nigredo',
  integration_coverage: 0.5, engagement_index: 0.5,
  bands: { alpha: 0.2, theta: 0.2, beta: 0.2, delta: 0.2, gamma: 0.2 },
  s_space: null,
  ea1: {
    eligible: false, score: 0, criteria_met: 0, criteria_total: 5,
    label: 'Not eligible', overlay_mode: '', gates: {}, criteria: {},
    alchemical_stage: 'Nigredo', s_space_coords: null,
    s_space_region: 'A', integration_coverage: 0,
  },
  last_ts: 0, frame_count: 1, poor_contact: false,
  region_v01: 'A', alchemical_stage_v01: 'Nigredo',
  faa: null, fmt: null, hr_bpm: null, hrv_rmssd: null, rr_bpm: null,
  pitch_deg: null, roll_deg: null, motion_rms: null, contact_quality: null,
  focus_state: 'unknown', focus_score: 0, fatigue_score: 0,
  fnirs_oxy: null, fnirs_deoxy: null,
  eeg_samples: [], bad_channels: [],
  artifact_rejected: false, artifact_reasons: [],
  artifact_annotations: {}, artifact_correction_plan: '',
  channel_impedances: {}, baseline_phase: null,
})

beforeEach(() => {
  MockEventSource.instances = []
  vi.stubGlobal('EventSource', MockEventSource)
  vi.useFakeTimers()
})

afterEach(() => {
  vi.useRealTimers()
  vi.unstubAllGlobals()
})

// ---------------------------------------------------------------------------
// Type guard unit tests
// ---------------------------------------------------------------------------

describe('isSettling', () => {
  it('returns true for a valid settling sentinel', () => {
    expect(isSettling({ event: 'settling', reason: 'impedance_unstable' })).toBe(true)
  })
  it('returns false for a NeurolinkState', () => {
    expect(isSettling(minState())).toBe(false)
  })
  it('returns false for null', () => {
    expect(isSettling(null)).toBe(false)
  })
})

describe('isBaselineComplete', () => {
  it('returns true for event-keyed sentinel', () => {
    expect(isBaselineComplete({ event: 'baseline_complete' })).toBe(true)
  })
  it('returns true for legacy type-keyed sentinel (backwards compat)', () => {
    expect(isBaselineComplete({ type: 'baseline_complete' })).toBe(true)
  })
  it('returns false for a state frame', () => {
    expect(isBaselineComplete(minState())).toBe(false)
  })
})

// ---------------------------------------------------------------------------
// Hook integration tests
// ---------------------------------------------------------------------------

describe('useNeurolinkSSE', () => {

  // ── Existing tests (unchanged) ────────────────────────────────────────────

  it('returns null state before any event arrives', () => {
    const { result } = renderHook(() => useNeurolinkSSE('http://test/stream'))
    expect(result.current.state).toBeNull()
  })

  it('updates state on named state event', () => {
    const { result } = renderHook(() => useNeurolinkSSE('http://test/stream'))
    const es = MockEventSource.instances[0]
    act(() => { es.emit('state', JSON.stringify(minState())) })
    expect(result.current.state?.frame_count).toBe(1)
    expect(result.current.state?.connected).toBe(true)
  })

  it('updates state via generic onmessage fallback', () => {
    const { result } = renderHook(() => useNeurolinkSSE('http://test/stream'))
    const es = MockEventSource.instances[0]
    act(() => { es.emitMessage(JSON.stringify({ ...minState(), frame_count: 42 })) })
    expect(result.current.state?.frame_count).toBe(42)
  })

  it('silently ignores malformed JSON on state event', () => {
    const { result } = renderHook(() => useNeurolinkSSE('http://test/stream'))
    const es = MockEventSource.instances[0]
    act(() => { es.emit('state', '{ bad json !!!') })
    expect(result.current.state).toBeNull()
  })

  it('schedules reconnect after error', () => {
    renderHook(() => useNeurolinkSSE('http://test/stream'))
    const es = MockEventSource.instances[0]
    act(() => { es.triggerError() })
    expect(es.closed).toBe(true)
    act(() => { vi.advanceTimersByTime(3100) })
    expect(MockEventSource.instances.length).toBe(2)
  })

  it('closes EventSource and cancels reconnect on unmount', () => {
    const { unmount } = renderHook(() => useNeurolinkSSE('http://test/stream'))
    const es = MockEventSource.instances[0]
    unmount()
    expect(es.closed).toBe(true)
    act(() => {
      es.triggerError()
      vi.advanceTimersByTime(4000)
    })
    expect(MockEventSource.instances.length).toBe(1)
  })

  it('reconnect timer is a no-op when component unmounts before timer fires', () => {
    const { unmount } = renderHook(() => useNeurolinkSSE('http://test/stream'))
    const es = MockEventSource.instances[0]
    act(() => { es.triggerError() })
    unmount()
    act(() => { vi.advanceTimersByTime(3100) })
    expect(MockEventSource.instances.length).toBe(1)
  })

  it('reconnects when url prop changes', () => {
    const { rerender } = renderHook(
      ({ url }: { url: string }) => useNeurolinkSSE(url),
      { initialProps: { url: 'http://test/stream' } },
    )
    rerender({ url: 'http://other/stream' })
    expect(MockEventSource.instances.length).toBe(2)
    expect(MockEventSource.instances[0].closed).toBe(true)
    expect(MockEventSource.instances[1].url).toBe('http://other/stream')
  })

  // ── baseline_complete tests ────────────────────────────────────────────────

  it('fires onBaselineComplete on named baseline_complete event', () => {
    const onBaselineComplete = vi.fn()
    renderHook(() => useNeurolinkSSE('http://test/stream', { onBaselineComplete }))
    const es = MockEventSource.instances[0]
    act(() => { es.emit('baseline_complete', '{}') })
    expect(onBaselineComplete).toHaveBeenCalledTimes(1)
  })

  it('does not mutate state on baseline_complete event', () => {
    const { result } = renderHook(() =>
      useNeurolinkSSE('http://test/stream', { onBaselineComplete: vi.fn() })
    )
    const es = MockEventSource.instances[0]
    act(() => { es.emit('baseline_complete', '{}') })
    // state should remain null — sentinel must not be fed into setState
    expect(result.current.state).toBeNull()
  })

  it('fires onBaselineComplete via onmessage fallback', () => {
    const onBaselineComplete = vi.fn()
    renderHook(() => useNeurolinkSSE('http://test/stream', { onBaselineComplete }))
    const es = MockEventSource.instances[0]
    act(() => { es.emitMessage(JSON.stringify({ event: 'baseline_complete' })) })
    expect(onBaselineComplete).toHaveBeenCalledTimes(1)
  })

  // ── settling tests ────────────────────────────────────────────────────────

  it('fires onSettling callback on named settling event', () => {
    const onSettling = vi.fn()
    renderHook(() => useNeurolinkSSE('http://test/stream', { onSettling }))
    const es = MockEventSource.instances[0]
    act(() => { es.emit('settling', JSON.stringify({ reason: 'impedance_unstable' })) })
    expect(onSettling).toHaveBeenCalledWith('impedance_unstable')
  })

  it('exposes settlingReason in return value', () => {
    const { result } = renderHook(() => useNeurolinkSSE('http://test/stream'))
    const es = MockEventSource.instances[0]
    expect(result.current.settlingReason).toBeNull()
    act(() => { es.emit('settling', JSON.stringify({ reason: 'motion_settling' })) })
    expect(result.current.settlingReason).toBe('motion_settling')
  })

  it('settlingReason clears to null after 2 s debounce', () => {
    const { result } = renderHook(() => useNeurolinkSSE('http://test/stream'))
    const es = MockEventSource.instances[0]
    act(() => { es.emit('settling', JSON.stringify({ reason: 'env_not_ready' })) })
    expect(result.current.settlingReason).toBe('env_not_ready')
    act(() => { vi.advanceTimersByTime(2100) })
    expect(result.current.settlingReason).toBeNull()
  })

  it('debounce timer resets on each new settling event', () => {
    const { result } = renderHook(() => useNeurolinkSSE('http://test/stream'))
    const es = MockEventSource.instances[0]
    act(() => { es.emit('settling', JSON.stringify({ reason: 'settling' })) })
    act(() => { vi.advanceTimersByTime(1500) })
    // emit again before 2 s expires — timer should reset
    act(() => { es.emit('settling', JSON.stringify({ reason: 'motion_settling' })) })
    act(() => { vi.advanceTimersByTime(1500) })
    // only 1.5 s has passed since last event; should still be set
    expect(result.current.settlingReason).toBe('motion_settling')
    act(() => { vi.advanceTimersByTime(600) })
    // now 2.1 s since last event; should be cleared
    expect(result.current.settlingReason).toBeNull()
  })

  it('handles all four SettlingReason codes', () => {
    const reasons = ['impedance_unstable', 'motion_settling', 'env_not_ready', 'settling'] as const
    for (const reason of reasons) {
      const { result, unmount } = renderHook(() => useNeurolinkSSE('http://test/stream'))
      const es = MockEventSource.instances[MockEventSource.instances.length - 1]
      act(() => { es.emit('settling', JSON.stringify({ reason })) })
      expect(result.current.settlingReason).toBe(reason)
      unmount()
    }
  })

  it('uses generic settling fallback reason for malformed settling data', () => {
    const { result } = renderHook(() => useNeurolinkSSE('http://test/stream'))
    const es = MockEventSource.instances[0]
    act(() => { es.emit('settling', '{ bad json') })
    expect(result.current.settlingReason).toBe('settling')
  })

  it('fires onSettling via onmessage fallback', () => {
    const onSettling = vi.fn()
    renderHook(() => useNeurolinkSSE('http://test/stream', { onSettling }))
    const es = MockEventSource.instances[0]
    act(() => { es.emitMessage(JSON.stringify({ event: 'settling', reason: 'env_not_ready' })) })
    expect(onSettling).toHaveBeenCalledWith('env_not_ready')
  })

  it('does not mutate state on settling event', () => {
    const { result } = renderHook(() => useNeurolinkSSE('http://test/stream'))
    const es = MockEventSource.instances[0]
    act(() => { es.emit('settling', JSON.stringify({ reason: 'settling' })) })
    expect(result.current.state).toBeNull()
  })

  it('ignores unknown named SSE event types', () => {
    const { result } = renderHook(() => useNeurolinkSSE('http://test/stream'))
    const es = MockEventSource.instances[0]
    // Emit a future/unknown event type; hook should not crash or mutate state
    act(() => { es.emit('unknown', JSON.stringify({ event: 'unknown', foo: 'bar' })) })
    expect(result.current.state).toBeNull()
    expect(result.current.settlingReason).toBeNull()
  })

  it('clears settling timer on unmount (no timer leak)', () => {
    const { unmount } = renderHook(() => useNeurolinkSSE('http://test/stream'))
    const es = MockEventSource.instances[0]
    act(() => { es.emit('settling', JSON.stringify({ reason: 'motion_settling' })) })
    // Unmount while the clear timer is still running
    unmount()
    // Advancing should not throw or call setState on an unmounted component
    expect(() => { act(() => { vi.advanceTimersByTime(3000) }) }).not.toThrow()
  })
})
