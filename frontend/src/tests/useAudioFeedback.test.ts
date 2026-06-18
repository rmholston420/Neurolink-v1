import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import { useAudioFeedback } from '../hooks/useAudioFeedback'

// Stub AudioContext — not available in jsdom
const mockStop    = vi.fn()
const mockStart   = vi.fn()
const mockConnect = vi.fn()
const mockSetValueAtTime           = vi.fn()
const mockExponentialRampToValue   = vi.fn()
const mockLinearRampToValue        = vi.fn()
const mockResume  = vi.fn().mockResolvedValue(undefined)
const mockClose   = vi.fn().mockResolvedValue(undefined)

const makeOscillator = () => ({
  connect:   mockConnect,
  start:     mockStart,
  stop:      mockStop,
  type:      'sine',
  frequency: { setValueAtTime: mockSetValueAtTime },
})

const mockAudioContext = () => ({
  currentTime: 0,
  state: 'running',
  destination: {},
  createOscillator: vi.fn().mockImplementation(makeOscillator),
  createGain: vi.fn().mockReturnValue({
    connect: mockConnect,
    gain: {
      setValueAtTime: mockSetValueAtTime,
      exponentialRampToValueAtTime: mockExponentialRampToValue,
      linearRampToValueAtTime: mockLinearRampToValue,
    },
  }),
  resume: mockResume,
  close:  mockClose,
})

beforeEach(() => {
  vi.stubGlobal('AudioContext', vi.fn().mockImplementation(mockAudioContext))
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.clearAllMocks()
})

describe('useAudioFeedback', () => {
  it('initialises with enabled=false by default', () => {
    const { result } = renderHook(() => useAudioFeedback(null))
    expect(result.current.enabled).toBe(false)
  })

  it('toggle() flips enabled from false to true', () => {
    const { result } = renderHook(() => useAudioFeedback(null))
    act(() => result.current.toggle())
    expect(result.current.enabled).toBe(true)
  })

  it('toggle() flips enabled from true to false', () => {
    const { result } = renderHook(() => useAudioFeedback(null, { enabled: true }))
    act(() => result.current.toggle())
    expect(result.current.enabled).toBe(false)
  })

  it('setVolume() updates config.volume', () => {
    const { result } = renderHook(() => useAudioFeedback(null))
    act(() => result.current.setVolume(0.42))
    expect(result.current.config.volume).toBeCloseTo(0.42)
  })

  it('setSensitivity() updates config.sensitivity', () => {
    const { result } = renderHook(() => useAudioFeedback(null))
    act(() => result.current.setSensitivity('high'))
    expect(result.current.config.sensitivity).toBe('high')
  })

  it('initialises with provided config values', () => {
    const { result } = renderHook(() =>
      useAudioFeedback(null, { enabled: false, volume: 0.3, sensitivity: 'low' })
    )
    expect(result.current.config.volume).toBeCloseTo(0.3)
    expect(result.current.config.sensitivity).toBe('low')
  })

  it('lastEvent is null initially', () => {
    const { result } = renderHook(() => useAudioFeedback(null))
    expect(result.current.lastEvent).toBeNull()
  })

  it('does not throw when state is null', () => {
    expect(() => renderHook(() => useAudioFeedback(null, { enabled: true }))).not.toThrow()
  })

  it('config is returned in the result', () => {
    const { result } = renderHook(() =>
      useAudioFeedback(null, { volume: 0.5, sensitivity: 'medium' })
    )
    expect(result.current.config).toMatchObject({ volume: 0.5, sensitivity: 'medium' })
  })
})
