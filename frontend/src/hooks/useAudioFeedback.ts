/**
 * useAudioFeedback
 *
 * Adaptive audio neurofeedback hook.  Creates Web Audio API oscillators
 * that fire short tones when brain-state thresholds are crossed:
 *
 *   FOCUS LOSS   - soft 220 Hz sine bell when focus_score drops below threshold
 *   EA-1 ENTRY   - rising 528 Hz chime when ea1.eligible becomes true
 *   WANDERING    - gentle 110 Hz pulse when engagement_index spikes
 *
 * No external libraries required.  All audio synthesis is in-browser.
 *
 * Usage:
 *   const audio = useAudioFeedback(state, { enabled, volume, sensitivity })
 *   // audio.enabled, audio.lastEvent, audio.toggle()
 */
import { useRef, useEffect, useCallback, useState } from 'react'
import type { NeurolinkState } from '../types'

export type AudioSensitivity = 'low' | 'medium' | 'high'

export interface AudioFeedbackConfig {
  enabled:     boolean
  volume:      number          // 0.0 – 1.0
  sensitivity: AudioSensitivity
}

export interface AudioFeedbackReturn {
  enabled:     boolean
  lastEvent:   string | null
  toggle:      () => void
  setVolume:   (v: number) => void
  setSensitivity: (s: AudioSensitivity) => void
  config:      AudioFeedbackConfig
}

// Sensitivity multiplier on the threshold — higher = fires more easily
const SENSITIVITY_K: Record<AudioSensitivity, number> = {
  low:    0.65,
  medium: 0.80,
  high:   0.92,
}

function playTone(
  ctx: AudioContext,
  freq: number,
  type: OscillatorType,
  duration: number,
  volume: number,
  fadeOut = true,
) {
  const osc  = ctx.createOscillator()
  const gain = ctx.createGain()
  osc.connect(gain)
  gain.connect(ctx.destination)
  osc.type      = type
  osc.frequency.setValueAtTime(freq, ctx.currentTime)
  gain.gain.setValueAtTime(volume * 0.25, ctx.currentTime)
  if (fadeOut) {
    gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + duration)
  }
  osc.start(ctx.currentTime)
  osc.stop(ctx.currentTime + duration)
}

function playChime(ctx: AudioContext, volume: number) {
  // Rising two-tone chime for EA-1 entry
  playTone(ctx, 528, 'sine', 0.8, volume)
  setTimeout(() => playTone(ctx, 792, 'sine', 1.0, volume * 0.7), 300)
}

function playBell(ctx: AudioContext, volume: number) {
  // Soft bell for focus loss — 220 Hz with triangle harmonic
  playTone(ctx, 220, 'sine',     0.9, volume)
  playTone(ctx, 440, 'triangle', 0.4, volume * 0.3)
}

function playWanderPulse(ctx: AudioContext, volume: number) {
  // Subtle low pulse for mind-wandering event
  playTone(ctx, 110, 'sine', 0.3, volume * 0.5, false)
  const osc  = ctx.createOscillator()
  const gain = ctx.createGain()
  osc.connect(gain)
  gain.connect(ctx.destination)
  osc.frequency.value = 110
  gain.gain.setValueAtTime(volume * 0.12, ctx.currentTime)
  gain.gain.linearRampToValueAtTime(0, ctx.currentTime + 0.3)
  osc.start(); osc.stop(ctx.currentTime + 0.3)
}

export function useAudioFeedback(
  state: Partial<NeurolinkState> | null,
  initialConfig?: Partial<AudioFeedbackConfig>,
): AudioFeedbackReturn {
  const [config, setConfig] = useState<AudioFeedbackConfig>({
    enabled:     initialConfig?.enabled     ?? false,
    volume:      initialConfig?.volume      ?? 0.6,
    sensitivity: initialConfig?.sensitivity ?? 'medium',
  })
  const [lastEvent, setLastEvent] = useState<string | null>(null)

  const ctxRef        = useRef<AudioContext | null>(null)
  const prevFocusRef  = useRef<number>(0)
  const prevEA1Ref    = useRef<boolean>(false)
  const prevEngageRef = useRef<number>(0)
  const cooldownRef   = useRef<Record<string, number>>({})
  const engageHistRef = useRef<number[]>([])

  // Lazy-init AudioContext on first enable (browser autoplay policy)
  const getCtx = useCallback(() => {
    if (!ctxRef.current || ctxRef.current.state === 'closed') {
      ctxRef.current = new AudioContext()
    }
    if (ctxRef.current.state === 'suspended') {
      ctxRef.current.resume()
    }
    return ctxRef.current
  }, [])

  const isCoolingDown = (key: string, ms: number): boolean => {
    const last = cooldownRef.current[key] ?? 0
    if (Date.now() - last < ms) return true
    cooldownRef.current[key] = Date.now()
    return false
  }

  useEffect(() => {
    if (!state || !config.enabled) return

    const k     = SENSITIVITY_K[config.sensitivity]
    const focus = state.focus_score  ?? 0
    const ea1ok = state.ea1?.eligible ?? false
    const eng   = state.engagement_index ?? 0

    const ctx = getCtx()

    // ─ 1. Focus-loss bell: focus drops below k * 0.5 (tunable) ────────────
    if (prevFocusRef.current >= k * 0.5 && focus < k * 0.5) {
      if (!isCoolingDown('focus', 4000)) {
        playBell(ctx, config.volume)
        setLastEvent(`Focus bell — score dropped to ${focus.toFixed(2)}`)
      }
    }
    prevFocusRef.current = focus

    // ─ 2. EA-1 entry chime ──────────────────────────────────────
    if (!prevEA1Ref.current && ea1ok) {
      if (!isCoolingDown('ea1', 8000)) {
        playChime(ctx, config.volume)
        setLastEvent('EA-1 entry — eligibility threshold crossed ✨')
      }
    }
    prevEA1Ref.current = ea1ok

    // ─ 3. Wandering pulse: engagement spikes > rolling mean + k*sigma ────
    engageHistRef.current.push(eng)
    if (engageHistRef.current.length > 60) engageHistRef.current.shift()
    if (engageHistRef.current.length >= 10) {
      const mean  = engageHistRef.current.reduce((a, b) => a + b, 0) / engageHistRef.current.length
      const sigma = Math.sqrt(
        engageHistRef.current.reduce((a, b) => a + (b - mean) ** 2, 0) / engageHistRef.current.length
      )
      const threshold = mean + k * sigma
      if (eng > threshold && prevEngageRef.current <= threshold) {
        if (!isCoolingDown('wander', 5000)) {
          playWanderPulse(ctx, config.volume)
          setLastEvent(`Wandering pulse — engagement spike ${eng.toFixed(3)}`)
        }
      }
    }
    prevEngageRef.current = eng
  }, [state, config, getCtx])

  // Clean up AudioContext on unmount
  useEffect(() => () => { ctxRef.current?.close() }, [])

  const toggle      = useCallback(() => setConfig(c => ({ ...c, enabled: !c.enabled })), [])
  const setVolume   = useCallback((v: number) => setConfig(c => ({ ...c, volume: v })), [])
  const setSensitivity = useCallback((s: AudioSensitivity) => setConfig(c => ({ ...c, sensitivity: s })), [])

  return { enabled: config.enabled, lastEvent, toggle, setVolume, setSensitivity, config }
}
