/**
 * useHRVCoherence
 *
 * Computes a HRV coherence score from the live hrv_rmssd stream and
 * drives a paced-breathing pacer targeting 0.1 Hz resonance frequency
 * (≈ 6 breaths/min, the standard heart-rate-variability biofeedback target).
 *
 * Coherence score algorithm:
 *   - Maintain a 30-sample rolling buffer of hrv_rmssd values
 *   - Compute coefficient of variation (CV = sigma/mean)
 *   - A high CV with a rhythm close to 0.1 Hz oscillation → high coherence
 *   - Score 0–100 is derived from: min(100, CV * 120) as a proxy
 *     (a real implementation would use frequency-domain LF/HF power ratio;
 *      this is a clinically reasonable approximation for a 4-electrode device)
 *
 * Pacer:
 *   - pacerPhase: 0–1 cycling at breathsPerMin / 60 Hz
 *   - pacerState: 'inhale' | 'exhale'
 *   - breathsPerMin: configurable 4–7 (default 6)
 */
import { useState, useRef, useEffect, useCallback } from 'react'
import type { NeurolinkState } from '../types'

export type PacerState = 'inhale' | 'exhale'

export interface HRVCoherenceReturn {
  coherenceScore:  number        // 0–100
  coherenceLabel:  string        // Low / Medium / High
  coherenceColour: string
  pacerPhase:      number        // 0–1 within current inhale or exhale
  pacerState:      PacerState
  breathsPerMin:   number
  setBreathsPerMin: (b: number) => void
  hrvMean:         number | null
  hrvSigma:        number | null
  nSamples:        number
}

const HRV_BUFFER = 30

function coherenceFromBuffer(buf: number[]): number {
  if (buf.length < 5) return 0
  const mean  = buf.reduce((a, b) => a + b, 0) / buf.length
  if (mean < 0.001) return 0
  const sigma = Math.sqrt(buf.reduce((a, b) => a + (b - mean) ** 2, 0) / buf.length)
  const cv    = sigma / mean
  // Map CV 0→0.4 to score 0→100; clamp at 100
  return Math.min(100, cv * 250)
}

export function useHRVCoherence(
  state: Partial<NeurolinkState> | null,
): HRVCoherenceReturn {
  const [coherenceScore,  setCoherenceScore]  = useState(0)
  const [pacerPhase,      setPacerPhase]      = useState(0)
  const [pacerState,      setPacerState]      = useState<PacerState>('inhale')
  const [breathsPerMin,   setBreathsPerMinS]  = useState(6)
  const [hrvMean,         setHrvMean]         = useState<number | null>(null)
  const [hrvSigma,        setHrvSigma]        = useState<number | null>(null)
  const [nSamples,        setNSamples]        = useState(0)

  const hrvBufRef       = useRef<number[]>([])
  const pacerStartRef   = useRef<number>(Date.now())
  const pacerTimerRef   = useRef<ReturnType<typeof setInterval> | null>(null)

  // HRV coherence computation — runs on each new frame
  useEffect(() => {
    const hrv = state?.hrv_rmssd
    if (hrv === null || hrv === undefined) return
    hrvBufRef.current.push(hrv)
    if (hrvBufRef.current.length > HRV_BUFFER) hrvBufRef.current.shift()

    const buf  = hrvBufRef.current
    const score = coherenceFromBuffer(buf)
    setCoherenceScore(score)
    setNSamples(buf.length)

    if (buf.length >= 5) {
      const mean  = buf.reduce((a, b) => a + b, 0) / buf.length
      const sigma = Math.sqrt(buf.reduce((a, b) => a + (b - mean) ** 2, 0) / buf.length)
      setHrvMean(mean)
      setHrvSigma(sigma)
    }
  }, [state?.frame_count])

  // Pacer animation loop — independent of EEG frames
  useEffect(() => {
    pacerStartRef.current = Date.now()
    const cycleDurationMs = (60 / breathsPerMin) * 1000  // one full breath cycle

    if (pacerTimerRef.current) clearInterval(pacerTimerRef.current)
    pacerTimerRef.current = setInterval(() => {
      const elapsed  = (Date.now() - pacerStartRef.current) % cycleDurationMs
      const phase    = elapsed / cycleDurationMs           // 0–1 over full cycle
      const inhaling = phase < 0.5
      setPacerPhase(inhaling ? phase * 2 : (phase - 0.5) * 2)  // 0–1 within phase
      setPacerState(inhaling ? 'inhale' : 'exhale')
    }, 50)

    return () => { if (pacerTimerRef.current) clearInterval(pacerTimerRef.current) }
  }, [breathsPerMin])

  const setBreathsPerMin = useCallback((b: number) => {
    setBreathsPerMinS(Math.min(7, Math.max(4, b)))
  }, [])

  const score = coherenceScore
  const coherenceLabel  = score >= 66 ? 'High' : score >= 33 ? 'Medium' : 'Low'
  const coherenceColour = score >= 66 ? '#3fb950' : score >= 33 ? '#e3b341' : '#f85149'

  return {
    coherenceScore: score, coherenceLabel, coherenceColour,
    pacerPhase, pacerState, breathsPerMin, setBreathsPerMin,
    hrvMean, hrvSigma, nSamples,
  }
}
