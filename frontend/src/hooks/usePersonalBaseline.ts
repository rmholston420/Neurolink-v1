/**
 * usePersonalBaseline
 *
 * Maintains a rolling personal baseline for alpha, theta, and focus_score
 * computed from the last N completed sessions.
 *
 * Each frame, computes a z-score deviation from the baseline mean:
 *   z = (current - mean) / sigma
 *
 * Exports:
 *   baseline        rolling mean ± sigma for each metric
 *   deviation       current z-scores { alpha, theta, focus }
 *   nSessions       number of sessions used for baseline
 *   isCalibrated    true once >= MIN_SESSIONS available
 *   recordSession   call at end of each session with mean values
 *   resetBaseline   clears accumulated data
 */
import { useState, useRef, useCallback, useEffect } from 'react'
import type { NeurolinkState } from '../types'

const MIN_SESSIONS = 3   // need at least 3 sessions for meaningful baseline
const MAX_SESSIONS = 20  // keep last 20

export interface MetricBaseline {
  mean:  number
  sigma: number
}

export interface BaselineSnapshot {
  alpha: MetricBaseline
  theta: MetricBaseline
  focus: MetricBaseline
}

export interface BaselineDeviation {
  alpha: number | null   // z-score, null if not calibrated
  theta: number | null
  focus: number | null
}

export interface SessionSample {
  meanAlpha: number
  meanTheta: number
  meanFocus: number
}

export interface PersonalBaselineReturn {
  baseline:     BaselineSnapshot | null
  deviation:    BaselineDeviation
  nSessions:    number
  isCalibrated: boolean
  recordSession: (sample: SessionSample) => void
  resetBaseline: () => void
}

function stats(vals: number[]): MetricBaseline {
  const mean  = vals.reduce((a, b) => a + b, 0) / vals.length
  const sigma = Math.sqrt(
    vals.reduce((a, b) => a + (b - mean) ** 2, 0) / vals.length
  ) || 0.001
  return { mean, sigma }
}

export function usePersonalBaseline(
  state: Partial<NeurolinkState> | null,
): PersonalBaselineReturn {
  const [sessions,  setSessions]  = useState<SessionSample[]>([])
  const [baseline,  setBaseline]  = useState<BaselineSnapshot | null>(null)
  const [deviation, setDeviation] = useState<BaselineDeviation>({ alpha: null, theta: null, focus: null })

  // Recompute baseline whenever sessions change
  useEffect(() => {
    if (sessions.length < MIN_SESSIONS) {
      setBaseline(null)
      return
    }
    setBaseline({
      alpha: stats(sessions.map(s => s.meanAlpha)),
      theta: stats(sessions.map(s => s.meanTheta)),
      focus: stats(sessions.map(s => s.meanFocus)),
    })
  }, [sessions])

  // Recompute deviation on every new frame
  useEffect(() => {
    if (!baseline || !state) {
      setDeviation({ alpha: null, theta: null, focus: null })
      return
    }
    const alpha = state.bands?.alpha ?? 0
    const theta = state.bands?.theta ?? 0
    const focus = state.focus_score  ?? 0
    setDeviation({
      alpha: (alpha - baseline.alpha.mean) / baseline.alpha.sigma,
      theta: (theta - baseline.theta.mean) / baseline.theta.sigma,
      focus: (focus - baseline.focus.mean) / baseline.focus.sigma,
    })
  }, [state?.frame_count, baseline])

  const recordSession = useCallback((sample: SessionSample) => {
    setSessions(prev => [sample, ...prev].slice(0, MAX_SESSIONS))
  }, [])

  const resetBaseline = useCallback(() => {
    setSessions([])
    setBaseline(null)
    setDeviation({ alpha: null, theta: null, focus: null })
  }, [])

  return {
    baseline,
    deviation,
    nSessions:    sessions.length,
    isCalibrated: sessions.length >= MIN_SESSIONS,
    recordSession,
    resetBaseline,
  }
}
