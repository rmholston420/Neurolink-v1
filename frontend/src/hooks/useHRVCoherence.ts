/**
 * useHRVCoherence — Tier 2
 * Estimates real-time HRV coherence ratio targeting the 0.1 Hz resonance band.
 * Uses a circular buffer of R-R intervals derived from hr_bpm to compute power
 * in the low-frequency coherence band (0.04–0.15 Hz) vs. total HRV power.
 *
 * Also drives a paced-breathing pacer: inhale/exhale phase at ~5.5 breath/min (0.092 Hz).
 */
import { useRef, useState, useEffect } from 'react'
import type { NeurolinkState } from '../types'

export type BreathPhase = 'inhale' | 'exhale'

export interface HRVCoherenceResult {
  coherenceRatio:  number       // 0–1; ≥ 0.6 considered coherent
  coherenceLabel:  string
  rrBufferLen:     number
  breathPhase:     BreathPhase
  breathProgress:  number       // 0–1 within current phase
  pacerBpm:        number       // configurable resonance breathing rate
  setPacerBpm:     (bpm: number) => void
  isCoherent:      boolean
}

const BUFFER_SIZE = 30   // ~30 s at ~1 sample/s derived from HR
const SAMPLE_RATE = 1    // Hz (we derive one RR per second from hr_bpm)

function lfPower(rr: number[]): number {
  // Approximate LF band (0.04–0.15 Hz) power via Goertzel-like sum
  // on the RR series. Simple DFT energy sum for target bins.
  if (rr.length < 4) return 0
  const N = rr.length
  let lf = 0, total = 0
  for (let k = 0; k < N; k++) {
    const freq = (k * SAMPLE_RATE) / N
    const power = rr.reduce((acc, x, n) => {
      return acc + x * Math.cos(2 * Math.PI * k * n / N)
    }, 0) ** 2 + rr.reduce((acc, x, n) => {
      return acc + x * Math.sin(2 * Math.PI * k * n / N)
    }, 0) ** 2
    total += power
    if (freq >= 0.04 && freq <= 0.15) lf += power
  }
  return total > 0 ? lf / total : 0
}

export function useHRVCoherence(state: NeurolinkState | null): HRVCoherenceResult {
  const rrBuffer   = useRef<number[]>([])
  const [coherenceRatio, setCoherenceRatio] = useState(0)
  const [pacerBpm, setPacerBpm] = useState(5.5)  // resonance frequency target
  const [breathPhase, setBreathPhase] = useState<BreathPhase>('inhale')
  const [breathProgress, setBreathProgress] = useState(0)

  // Build RR buffer from hr_bpm
  const hrRef = useRef<number | null>(null)
  hrRef.current = state?.hr_bpm ?? null

  useEffect(() => {
    const id = setInterval(() => {
      const hr = hrRef.current
      if (hr && hr > 30 && hr < 220) {
        const rr = 60 / hr  // seconds per beat
        rrBuffer.current.push(rr)
        if (rrBuffer.current.length > BUFFER_SIZE) rrBuffer.current.shift()
        const ratio = lfPower(rrBuffer.current)
        setCoherenceRatio(Math.min(1, ratio * 4))  // scale to 0–1 display
      }
    }, 1000)
    return () => clearInterval(id)
  }, [])

  // Breathing pacer at pacerBpm
  useEffect(() => {
    const cycleSec  = 60 / pacerBpm         // total seconds per breath cycle
    const halfCycle = cycleSec / 2 * 1000   // ms per phase
    let startTime = performance.now()
    let phase: BreathPhase = 'inhale'

    const id = setInterval(() => {
      const now     = performance.now()
      const elapsed = now - startTime
      const progress = Math.min(1, elapsed / halfCycle)
      setBreathProgress(progress)
      setBreathPhase(phase)
      if (progress >= 1) {
        phase = phase === 'inhale' ? 'exhale' : 'inhale'
        startTime = now
      }
    }, 50)
    return () => clearInterval(id)
  }, [pacerBpm])

  const isCoherent = coherenceRatio >= 0.6
  const coherenceLabel =
    coherenceRatio >= 0.8 ? 'High Coherence' :
    coherenceRatio >= 0.6 ? 'Coherent' :
    coherenceRatio >= 0.3 ? 'Building' : 'Low Coherence'

  return {
    coherenceRatio,
    coherenceLabel,
    rrBufferLen: rrBuffer.current.length,
    breathPhase,
    breathProgress,
    pacerBpm,
    setPacerBpm,
    isCoherent,
  }
}
