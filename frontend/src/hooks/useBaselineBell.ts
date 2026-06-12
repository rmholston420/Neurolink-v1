/**
 * useBaselineBell
 *
 * Plays a single bell tone the moment the backend emits a baseline_complete
 * sentinel over the SSE stream.  Reuses the same Web Audio synthesis
 * pattern as useAudioFeedback (220 Hz sine + 440 Hz triangle harmonic) so
 * the timbre is consistent across all Neurolink audio events.
 *
 * Integration
 * ───────────
 * Replace your existing useNeurolinkSSE(url) call with:
 *
 *   const { state, baselineComplete, lastSentinel } = useBaselineBell(url)
 *
 * Or, if you are already calling useNeurolinkSSE elsewhere and only need the
 * bell, pass your existing onBaselineComplete callback directly:
 *
 *   const state = useNeurolinkSSE(url, { onBaselineComplete: ringBell })
 *   // where ringBell comes from the useBaselineBell standalone export below.
 *
 * Bell characteristics
 * ───────────────────
 *   220 Hz sine,     0.9 s, exponential fade-out  (fundamental)
 *   440 Hz triangle, 0.4 s, 30 % gain              (octave partial — adds bell shimmer)
 *
 * Volume
 * ──────
 *   Defaults to 0.6 (same default as useAudioFeedback).
 *   Pass { volume: 0.0–1.0 } to override.
 *
 * Asset fallback
 * ─────────────
 *   If you prefer a recorded bell sample, set { assetUrl: '/sounds/bell.mp3' }.
 *   The hook will decode the file once on first trigger and cache the buffer.
 *   Falls back to the synthesised tone if decoding fails.
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import {
  useNeurolinkSSE,
  type BaselineCompleteSentinel,
} from './useNeurolinkSSE'
import type { NeurolinkState } from '../types'

// ─── Audio helpers ───────────────────────────────────────────────────────

function playOscTone(
  ctx: AudioContext,
  freq: number,
  type: OscillatorType,
  duration: number,
  volume: number,
) {
  const osc  = ctx.createOscillator()
  const gain = ctx.createGain()
  osc.connect(gain)
  gain.connect(ctx.destination)
  osc.type = type
  osc.frequency.setValueAtTime(freq, ctx.currentTime)
  gain.gain.setValueAtTime(volume * 0.25, ctx.currentTime)
  gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + duration)
  osc.start(ctx.currentTime)
  osc.stop(ctx.currentTime + duration)
}

/** Synthesised bell: 220 Hz fundamental + 440 Hz shimmer partial. */
function playSynthBell(ctx: AudioContext, volume: number) {
  playOscTone(ctx, 220, 'sine',     0.9, volume)
  playOscTone(ctx, 440, 'triangle', 0.4, volume * 0.3)
}

/** Decoded-asset bell: plays a pre-loaded AudioBuffer. */
function playBufferBell(ctx: AudioContext, buffer: AudioBuffer, volume: number) {
  const source = ctx.createBufferSource()
  const gain   = ctx.createGain()
  source.buffer = buffer
  source.connect(gain)
  gain.connect(ctx.destination)
  gain.gain.setValueAtTime(volume, ctx.currentTime)
  source.start(ctx.currentTime)
}

// ─── Public API types ──────────────────────────────────────────────────

export interface BaselineBellOptions {
  /** Master gain for the bell, 0.0 – 1.0.  Default: 0.6 */
  volume?: number
  /**
   * Optional path to a recorded bell sample (e.g. '/sounds/bell.mp3').
   * When provided, the hook decodes the asset once and plays it instead of
   * the synthesised tone.  Falls back to synthesis if fetch/decode fails.
   */
  assetUrl?: string
  /**
   * If false, the bell is armed but silent.  Lets the parent component
   * temporarily mute without unmounting the hook.  Default: true
   */
  enabled?: boolean
}

export interface BaselineBellReturn {
  /** The live NeurolinkState stream (same as useNeurolinkSSE). */
  state: NeurolinkState | null
  /** True after the first baseline_complete sentinel has been received. */
  baselineComplete: boolean
  /** The most-recently received sentinel, or null if none yet. */
  lastSentinel: BaselineCompleteSentinel | null
  /** ISO timestamp of when the bell last rang, or null. */
  lastRangAt: string | null
  /** Programmatically mute/unmute the bell without changing options. */
  setEnabled: (enabled: boolean) => void
}

// ─── Hook ───────────────────────────────────────────────────────────

export function useBaselineBell(
  url: string,
  options: BaselineBellOptions = {},
): BaselineBellReturn {
  const volume  = options.volume  ?? 0.6
  const assetUrl = options.assetUrl

  const [enabled,          setEnabled]         = useState(options.enabled ?? true)
  const [baselineComplete, setBaselineComplete] = useState(false)
  const [lastSentinel,     setLastSentinel]     = useState<BaselineCompleteSentinel | null>(null)
  const [lastRangAt,       setLastRangAt]       = useState<string | null>(null)

  // AudioContext is created lazily on first bell ring (browser autoplay policy).
  const ctxRef    = useRef<AudioContext | null>(null)
  // Decoded asset buffer cache — only populated if assetUrl is provided.
  const bufferRef = useRef<AudioBuffer | null>(null)
  const enabledRef = useRef(enabled)
  enabledRef.current = enabled
  const volumeRef = useRef(volume)
  volumeRef.current = volume

  // ─ Ensure AudioContext is alive and resumed ───────────────────────────
  const getCtx = useCallback((): AudioContext => {
    if (!ctxRef.current || ctxRef.current.state === 'closed') {
      ctxRef.current = new AudioContext()
    }
    if (ctxRef.current.state === 'suspended') {
      void ctxRef.current.resume()
    }
    return ctxRef.current
  }, [])

  // ─ Optionally pre-fetch and decode the audio asset ────────────────────
  useEffect(() => {
    if (!assetUrl) return
    let cancelled = false
    // Decode lazily: we need an AudioContext to decode, so do it now.
    const ctx = getCtx()
    fetch(assetUrl)
      .then(r => r.arrayBuffer())
      .then(ab => ctx.decodeAudioData(ab))
      .then(buf => { if (!cancelled) bufferRef.current = buf })
      .catch(() => {
        // Asset unavailable — will fall through to synthesised bell.
        if (!cancelled) bufferRef.current = null
      })
    return () => { cancelled = true }
  }, [assetUrl, getCtx])

  // ─ The sentinel handler passed to useNeurolinkSSE ────────────────────
  const handleSentinel = useCallback((sentinel: BaselineCompleteSentinel) => {
    setBaselineComplete(true)
    setLastSentinel(sentinel)

    if (!enabledRef.current) return

    const ctx = getCtx()
    const vol = volumeRef.current

    if (bufferRef.current) {
      // Play decoded asset file
      playBufferBell(ctx, bufferRef.current, vol)
    } else {
      // Play synthesised bell
      playSynthBell(ctx, vol)
    }

    setLastRangAt(new Date().toISOString())
  }, [getCtx])

  // ─ Delegate stream consumption to the existing hook ──────────────────
  const state = useNeurolinkSSE(url, { onBaselineComplete: handleSentinel })

  // ─ Cleanup AudioContext on unmount ───────────────────────────────
  useEffect(() => () => { ctxRef.current?.close() }, [])

  return {
    state,
    baselineComplete,
    lastSentinel,
    lastRangAt,
    setEnabled,
  }
}
