import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import { useNeurolinkSSE } from '../hooks/useNeurolinkSSE'
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

  emit(type: string, data: string) {
    const evt = { data } as MessageEvent
    ;(this.listeners[type] ?? []).forEach(h => h(evt))
  }

  emitMessage(data: string) {
    if (this.onmessage) this.onmessage({ data } as MessageEvent)
  }

  triggerError() {
    if (this.onerror) this.onerror()
  }
}

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

describe('useNeurolinkSSE', () => {
  it('returns null before any event arrives', () => {
    const { result } = renderHook(() => useNeurolinkSSE('http://test/stream'))
    expect(result.current).toBeNull()
  })

  it('updates state on named state event', () => {
    const { result } = renderHook(() => useNeurolinkSSE('http://test/stream'))
    const es = MockEventSource.instances[0]
    act(() => { es.emit('state', JSON.stringify(minState())) })
    expect(result.current?.frame_count).toBe(1)
    expect(result.current?.connected).toBe(true)
  })

  it('updates state via generic onmessage fallback', () => {
    const { result } = renderHook(() => useNeurolinkSSE('http://test/stream'))
    const es = MockEventSource.instances[0]
    act(() => { es.emitMessage(JSON.stringify({ ...minState(), frame_count: 42 })) })
    expect(result.current?.frame_count).toBe(42)
  })

  it('silently ignores malformed JSON', () => {
    const { result } = renderHook(() => useNeurolinkSSE('http://test/stream'))
    const es = MockEventSource.instances[0]
    act(() => { es.emit('state', '{ bad json !!!') })
    expect(result.current).toBeNull()
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

  it('reconnect timer is a no-op when component unmounts before timer fires (line 22 guard)', () => {
    // Exercises the cancelledRef.current early-return on line 22:
    // error fires -> setTimeout scheduled -> component unmounts -> timer fires
    // connect() should bail immediately without creating a new EventSource
    const { unmount } = renderHook(() => useNeurolinkSSE('http://test/stream'))
    const es = MockEventSource.instances[0]
    // Trigger error to schedule the reconnect timer (3 s)
    act(() => { es.triggerError() })
    // Unmount BEFORE the timer fires — sets cancelledRef.current = true
    unmount()
    // Now let the timer fire; connect() must hit the early-return guard
    act(() => { vi.advanceTimersByTime(3100) })
    // Still only 1 EventSource ever constructed
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
})
