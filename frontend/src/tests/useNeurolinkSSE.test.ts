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

  close() {
    this.closed = true
  }

  /** Test helper — dispatch a named event */
  emit(type: string, data: string) {
    const evt = { data } as MessageEvent
    ;(this.listeners[type] ?? []).forEach(h => h(evt))
  }

  /** Test helper — dispatch via onmessage */
  emitMessage(data: string) {
    if (this.onmessage) this.onmessage({ data } as MessageEvent)
  }

  /** Test helper — trigger the error handler */
  triggerError() {
    if (this.onerror) this.onerror()
  }
}

const minState = (): NeurolinkState => ({
  connected: true,
  source: 'mock',
  region: 'A',
  alchemical_stage: 'Nigredo',
  integration_coverage: 0.5,
  engagement_index: 0.5,
  bands: { alpha: 0.2, theta: 0.2, beta: 0.2, delta: 0.2, gamma: 0.2 },
  s_space: null,
  ea1: {
    eligible: false, score: 0, criteria_met: 0, criteria_total: 5,
    label: 'Not eligible', overlay_mode: '', gates: {}, criteria: {},
    alchemical_stage: 'Nigredo', s_space_coords: null,
    s_space_region: 'A', integration_coverage: 0,
  },
  last_ts: 0,
  frame_count: 1,
  poor_contact: false,
  region_v01: 'A',
  alchemical_stage_v01: 'Nigredo',
  faa: null, fmt: null,
  hr_bpm: null, hrv_rmssd: null, rr_bpm: null,
  pitch_deg: null, roll_deg: null, motion_rms: null,
  contact_quality: null,
  focus_state: 'unknown',
  focus_score: 0,
  fatigue_score: 0,
  fnirs_oxy: null,
  fnirs_deoxy: null,
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
    act(() => {
      es.emit('state', JSON.stringify(minState()))
    })
    expect(result.current?.frame_count).toBe(1)
    expect(result.current?.connected).toBe(true)
  })

  it('updates state via generic onmessage fallback', () => {
    const { result } = renderHook(() => useNeurolinkSSE('http://test/stream'))
    const es = MockEventSource.instances[0]
    const s = { ...minState(), frame_count: 42 }
    act(() => {
      es.emitMessage(JSON.stringify(s))
    })
    expect(result.current?.frame_count).toBe(42)
  })

  it('silently ignores malformed JSON', () => {
    const { result } = renderHook(() => useNeurolinkSSE('http://test/stream'))
    const es = MockEventSource.instances[0]
    act(() => {
      es.emit('state', '{ bad json !!!')
    })
    expect(result.current).toBeNull()
  })

  it('schedules reconnect after error', () => {
    renderHook(() => useNeurolinkSSE('http://test/stream'))
    const es = MockEventSource.instances[0]
    act(() => { es.triggerError() })
    expect(es.closed).toBe(true)
    // Advance past the 3-second back-off
    act(() => { vi.advanceTimersByTime(3100) })
    // A second EventSource should have been constructed
    expect(MockEventSource.instances.length).toBe(2)
  })

  it('closes EventSource and cancels reconnect on unmount', () => {
    const { unmount } = renderHook(() => useNeurolinkSSE('http://test/stream'))
    const es = MockEventSource.instances[0]
    unmount()
    expect(es.closed).toBe(true)
    // Error after unmount should NOT trigger a new connection
    act(() => {
      es.triggerError()
      vi.advanceTimersByTime(4000)
    })
    expect(MockEventSource.instances.length).toBe(1)
  })

  it('reconnects when url prop changes', () => {
    const { rerender } = renderHook(
      ({ url }: { url: string }) => useNeurolinkSSE(url),
      { initialProps: { url: 'http://test/stream' } },
    )
    rerender({ url: 'http://other/stream' })
    // Old ES closed, new one opened
    expect(MockEventSource.instances.length).toBe(2)
    expect(MockEventSource.instances[0].closed).toBe(true)
    expect(MockEventSource.instances[1].url).toBe('http://other/stream')
  })
})
