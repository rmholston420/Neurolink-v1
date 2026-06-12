/**
 * usePersonalBaseline — Tier 2
 * Maintains a rolling baseline (exponential moving average) for alpha, theta,
 * beta, focus_score, and fatigue_score. Returns per-metric deviation from norm.
 */
import { useRef, useState, useEffect } from 'react'
import type { NeurolinkState } from '../types'

export interface BaselineMetrics {
  alpha:        number | null
  theta:        number | null
  beta:         number | null
  focusScore:   number | null
  fatigueScore: number | null
}

export interface BaselineDeviation {
  alpha:        number | null  // (current - baseline) / baseline * 100  [%]
  theta:        number | null
  beta:         number | null
  focusScore:   number | null
  fatigueScore: number | null
}

export interface PersonalBaselineResult {
  baseline:   BaselineMetrics
  deviation:  BaselineDeviation
  sampleCount: number
  resetBaseline: () => void
}

const ALPHA = 0.03  // EMA smoothing — ~33-sample half-life at 1 fps

function ema(prev: number | null, next: number): number {
  if (prev === null) return next
  return prev * (1 - ALPHA) + next * ALPHA
}

function deviation(current: number, base: number | null): number | null {
  if (base === null || base === 0) return null
  return ((current - base) / base) * 100
}

export function usePersonalBaseline(state: NeurolinkState | null): PersonalBaselineResult {
  const [baseline, setBaseline] = useState<BaselineMetrics>({
    alpha: null, theta: null, beta: null, focusScore: null, fatigueScore: null,
  })
  const [sampleCount, setSampleCount] = useState(0)
  const stateRef = useRef(state)
  stateRef.current = state

  // Tick at 1 Hz (state updates arrive faster; we sample periodically)
  useEffect(() => {
    const id = setInterval(() => {
      const s = stateRef.current
      if (!s?.connected || !s.bands) return
      setBaseline(prev => ({
        alpha:        ema(prev.alpha,        s.bands.alpha),
        theta:        ema(prev.theta,        s.bands.theta),
        beta:         ema(prev.beta,         s.bands.beta),
        focusScore:   ema(prev.focusScore,   s.focus_score),
        fatigueScore: ema(prev.fatigueScore, s.fatigue_score),
      }))
      setSampleCount(n => n + 1)
    }, 1000)
    return () => clearInterval(id)
  }, [])

  const resetBaseline = () => {
    setBaseline({ alpha: null, theta: null, beta: null, focusScore: null, fatigueScore: null })
    setSampleCount(0)
  }

  const bands = state?.bands ?? null
  const dev: BaselineDeviation = {
    alpha:        bands ? deviation(bands.alpha,        baseline.alpha)        : null,
    theta:        bands ? deviation(bands.theta,        baseline.theta)        : null,
    beta:         bands ? deviation(bands.beta,         baseline.beta)         : null,
    focusScore:   state  ? deviation(state.focus_score,  baseline.focusScore)  : null,
    fatigueScore: state  ? deviation(state.fatigue_score, baseline.fatigueScore) : null,
  }

  return { baseline, deviation: dev, sampleCount, resetBaseline }
}
