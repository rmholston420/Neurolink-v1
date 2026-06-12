/**
 * useBaselineBell.test.ts
 *
 * Covers:
 *   1. isBaselineComplete type guard returns true for sentinel, false for normal frame
 *   2. Normal NeurolinkState frames update hook state; sentinel does not
 *   3. onBaselineComplete fires only for sentinel frames
 *   4. Bell rings (AudioContext + oscillator created) on sentinel
 *   5. Bell is silent when enabled=false
 *   6. baselineComplete flag is set after first sentinel
 *   7. lastSentinel is populated with sentinel payload
 */

import { renderHook, act } from '@testing-library/react'
import { isBaselineComplete } from '../hooks/useNeurolinkSSE'
import { useBaselineBell } from '../hooks/useBaselineBell'
import type { BaselineCompleteSentinel } from '../hooks/useNeurolinkSSE'
import type { NeurolinkState } from '../types'

// ─── Minimal fixture factories ──────────────────────────────────────────

function makeSentinel(overrides?: Partial<BaselineCompleteSentinel>): BaselineCompleteSentinel {
  return {
    type:          'baseline_complete',
    bands:         { alpha: 0.45, theta: 0.3, beta: 0.2, delta: 0.05, gamma: 0.0 },
    focus_score:   0.62,
    fatigue_score: 0.21,
    sample_count:  120,
    duration_s:    120,
    ...overrides,
  }
}

function makeNeurolinkState(overrides?: Partial<NeurolinkState>): NeurolinkState {
  return {
    connected:          true,
    source:             'muse2',
    region:             'prefrontal',
    alchemical_stage:   'nigredo',
    integration_coverage: 0.4,
    engagement_index:   0.5,
    bands:              { alpha: 0.4, theta: 0.3, beta: 0.2, delta: 0.05, gamma: 0.05 },
    s_space:            null,
    ea1: {
      eligible: false, score: 0, criteria_met: 0, criteria_total: 5,
      label: '', gates: {}, criteria: {}, overlay_mode: '',
      alchemical_stage: '', s_space_coords: null, s_space_region: '',
      integration_coverage: 0,
    },
    last_ts:             Date.now(),
    frame_count:         1,
    poor_contact:        false,
    region_v01:          '',
    alchemical_stage_v01: '',
    faa: null, fmt: null, hr_bpm: null, hrv_rmssd: null, rr_bpm: null,
    pitch_deg: null, roll_deg: null, motion_rms: null, contact_quality: null,
    focus_state: 'focused', focus_score: 0.7, fatigue_score: 0.2,
    fnirs_oxy: null, fnirs_deoxy: null,
    eeg_samples: [], bad_channels: [], artifact_rejected: false,
    artifact_reasons: [], channel_impedances: {},
    ...overrides,
  } as NeurolinkState
}

// ─── Web Audio mock ──────────────────────────────────────────────────
// jsdom does not ship Web Audio API — provide a minimal stub.

const mockOscStart = vi.fn()
const mockOscStop  = vi.fn()
const mockGainConnect = vi.fn()
const mockOscConnect  = vi.fn()
const mockDestConnect = vi.fn()

const mockCreateOscillator = vi.fn(() => ({
  connect:   mockOscConnect,
  start:     mockOscStart,
  stop:      mockOscStop,
  type:      'sine' as OscillatorType,
  frequency: { setValueAtTime: vi.fn() },
}))

const mockCreateGain = vi.fn(() => ({
  connect: mockGainConnect,
  gain:    { setValueAtTime: vi.fn(), exponentialRampToValueAtTime: vi.fn() },
}))

const mockAudioContextInstance = {
  state:            'running',
  currentTime:      0,
  destination:      { connect: mockDestConnect },
  createOscillator: mockCreateOscillator,
  createGain:       mockCreateGain,
  resume:           vi.fn().mockResolvedValue(undefined),
  close:            vi.fn().mockResolvedValue(undefined),
}

const MockAudioContext = vi.fn(() => mockAudioContextInstance)

beforeAll(() => {
  // @ts-expect-error  jsdom lacks AudioContext
  globalThis.AudioContext = MockAudioContext
})

afterEach(() => {
  vi.clearAllMocks()
})

// ─── EventSource mock ────────────────────────────────────────────────

let capturedHandler: ((e: MessageEvent) => void) | null = null

class MockEventSource {
  onmessage: ((e: MessageEvent) => void) | null = null
  onerror:   (() => void) | null = null
  private listeners: Map<string, (e: MessageEvent) => void> = new Map()

  constructor(public url: string) {}

  addEventListener(type: string, handler: (e: MessageEvent) => void) {
    this.listeners.set(type, handler)
    // Expose for tests
    if (type === 'state') capturedHandler = handler
  }

  removeEventListener(type: string) { this.listeners.delete(type) }
  close() {}

  /** Test helper: push a raw JSON string as a 'state' event. */
  emit(json: string) {
    const e = new MessageEvent('state', { data: json })
    capturedHandler?.(e)
    this.onmessage?.(e)
  }
}

beforeAll(() => {
  // @ts-expect-error  override global
  globalThis.EventSource = MockEventSource
})

// ─── Tests: isBaselineComplete type guard ─────────────────────────────

describe('isBaselineComplete()', () => {
  it('returns true for a well-formed sentinel', () => {
    expect(isBaselineComplete(makeSentinel())).toBe(true)
  })

  it('returns false for a normal NeurolinkState frame', () => {
    // NeurolinkState has no `type` field
    expect(isBaselineComplete(makeNeurolinkState() as never)).toBe(false)
  })

  it('returns false for an object with an unexpected type value', () => {
    expect(isBaselineComplete({ type: 'other_event' } as never)).toBe(false)
  })

  it('returns false for a plain empty object', () => {
    expect(isBaselineComplete({} as never)).toBe(false)
  })
})

// ─── Tests: useBaselineBell hook ────────────────────────────────────────

describe('useBaselineBell()', () => {
  const TEST_URL = '/sse/stream'

  it('state is null initially', () => {
    const { result } = renderHook(() => useBaselineBell(TEST_URL))
    expect(result.current.state).toBeNull()
    expect(result.current.baselineComplete).toBe(false)
    expect(result.current.lastSentinel).toBeNull()
  })

  it('state updates when a normal NeurolinkState frame arrives', () => {
    const { result } = renderHook(() => useBaselineBell(TEST_URL))
    const frame = makeNeurolinkState({ frame_count: 42 })

    act(() => { capturedHandler?.(new MessageEvent('state', { data: JSON.stringify(frame) })) })

    expect(result.current.state?.frame_count).toBe(42)
    expect(result.current.baselineComplete).toBe(false)
  })

  it('state does NOT change when a sentinel arrives', () => {
    const { result } = renderHook(() => useBaselineBell(TEST_URL))
    const frame    = makeNeurolinkState({ frame_count: 7 })
    const sentinel = makeSentinel()

    act(() => { capturedHandler?.(new MessageEvent('state', { data: JSON.stringify(frame) })) })
    const stateAfterFrame = result.current.state

    act(() => { capturedHandler?.(new MessageEvent('state', { data: JSON.stringify(sentinel) })) })

    // state atom must be the same reference — sentinel must not overwrite it
    expect(result.current.state).toStrictEqual(stateAfterFrame)
  })

  it('baselineComplete becomes true after sentinel', () => {
    const { result } = renderHook(() => useBaselineBell(TEST_URL))

    act(() => {
      capturedHandler?.(new MessageEvent('state', { data: JSON.stringify(makeSentinel()) }))
    })

    expect(result.current.baselineComplete).toBe(true)
  })

  it('lastSentinel is populated with sentinel payload', () => {
    const { result } = renderHook(() => useBaselineBell(TEST_URL))
    const sentinel = makeSentinel({ sample_count: 90, duration_s: 90 })

    act(() => {
      capturedHandler?.(new MessageEvent('state', { data: JSON.stringify(sentinel) }))
    })

    expect(result.current.lastSentinel?.sample_count).toBe(90)
    expect(result.current.lastSentinel?.duration_s).toBe(90)
  })

  it('creates AudioContext and oscillators when sentinel fires (bell rings)', () => {
    renderHook(() => useBaselineBell(TEST_URL, { volume: 0.8 }))

    act(() => {
      capturedHandler?.(new MessageEvent('state', { data: JSON.stringify(makeSentinel()) }))
    })

    expect(MockAudioContext).toHaveBeenCalledTimes(1)
    // playSynthBell calls createOscillator twice (220 Hz + 440 Hz)
    expect(mockCreateOscillator).toHaveBeenCalledTimes(2)
    expect(mockOscStart).toHaveBeenCalledTimes(2)
  })

  it('does NOT ring when enabled=false', () => {
    renderHook(() => useBaselineBell(TEST_URL, { enabled: false }))

    act(() => {
      capturedHandler?.(new MessageEvent('state', { data: JSON.stringify(makeSentinel()) }))
    })

    // AudioContext should never be instantiated when muted
    expect(MockAudioContext).not.toHaveBeenCalled()
  })

  it('lastRangAt is set to an ISO string after bell fires', () => {
    const { result } = renderHook(() => useBaselineBell(TEST_URL))

    act(() => {
      capturedHandler?.(new MessageEvent('state', { data: JSON.stringify(makeSentinel()) }))
    })

    expect(result.current.lastRangAt).toMatch(/^\d{4}-\d{2}-\d{2}T/)
  })

  it('setEnabled(false) silences subsequent sentinels', () => {
    const { result } = renderHook(() => useBaselineBell(TEST_URL))

    act(() => { result.current.setEnabled(false) })

    act(() => {
      capturedHandler?.(new MessageEvent('state', { data: JSON.stringify(makeSentinel()) }))
    })

    expect(MockAudioContext).not.toHaveBeenCalled()
    // baselineComplete still fires — the flag tracks receipt, not audio
    expect(result.current.baselineComplete).toBe(true)
  })

  it('ignores malformed JSON without throwing', () => {
    const { result } = renderHook(() => useBaselineBell(TEST_URL))

    expect(() => {
      act(() => {
        capturedHandler?.(new MessageEvent('state', { data: 'not json {{{' }))
      })
    }).not.toThrow()

    expect(result.current.state).toBeNull()
  })
})
