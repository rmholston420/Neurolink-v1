/**
 * useMeditationTimer
 *
 * Configurable multi-stage meditation timer.
 *
 * Features:
 *   - Define N stages, each with a name and duration in seconds
 *   - Auto-advances through stages; fires a bell tone at each transition
 *   - Exposes elapsed, remaining, currentStage, totalProgress (0–1)
 *   - Controls: start, pause, resume, reset
 *   - onComplete callback fired when all stages finish
 *
 * Audio uses Web Audio API (same pattern as useAudioFeedback — no deps).
 */
import { useState, useRef, useEffect, useCallback } from 'react'

export interface TimerStage {
  name:       string
  durationS:  number   // seconds
}

export type TimerStatus = 'idle' | 'running' | 'paused' | 'complete'

export interface MeditationTimerReturn {
  status:         TimerStatus
  stageIndex:     number
  currentStage:   TimerStage
  stageElapsed:   number   // seconds elapsed in current stage
  stageRemaining: number   // seconds remaining in current stage
  totalElapsed:   number   // seconds elapsed across all stages
  totalDuration:  number   // total seconds across all stages
  totalProgress:  number   // 0–1
  stageProgress:  number   // 0–1
  stages:         TimerStage[]
  start:    () => void
  pause:    () => void
  resume:   () => void
  reset:    () => void
  setStages: (s: TimerStage[]) => void
}

const DEFAULT_STAGES: TimerStage[] = [
  { name: 'Settle',  durationS: 3 * 60  },
  { name: 'Deepen',  durationS: 10 * 60 },
  { name: 'Rest',    durationS: 3 * 60  },
]

function playBell(freq = 432, duration = 1.5) {
  try {
    const ctx  = new AudioContext()
    const osc  = ctx.createOscillator()
    const gain = ctx.createGain()
    osc.connect(gain); gain.connect(ctx.destination)
    osc.type = 'sine'
    osc.frequency.setValueAtTime(freq, ctx.currentTime)
    gain.gain.setValueAtTime(0.35, ctx.currentTime)
    gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + duration)
    osc.start(); osc.stop(ctx.currentTime + duration)
    setTimeout(() => ctx.close(), (duration + 0.2) * 1000)
  } catch (_) { /* audio not available */ }
}

function playCompletionBells() {
  playBell(432, 2.0)
  setTimeout(() => playBell(528, 2.0), 600)
  setTimeout(() => playBell(639, 2.5), 1300)
}

export function useMeditationTimer(
  onComplete?: () => void,
): MeditationTimerReturn {
  const [stages,     setStages]     = useState<TimerStage[]>(DEFAULT_STAGES)
  const [status,     setStatus]     = useState<TimerStatus>('idle')
  const [stageIndex, setStageIndex] = useState(0)
  const [elapsed,    setElapsed]    = useState(0)  // total elapsed seconds

  const intervalRef    = useRef<ReturnType<typeof setInterval> | null>(null)
  const startTimeRef   = useRef<number>(0)    // wall-clock when last resumed
  const accumulatedRef = useRef<number>(0)    // elapsed before last pause

  const totalDuration = stages.reduce((a, s) => a + s.durationS, 0)

  // Derive per-stage position from total elapsed
  const getStagePos = useCallback((totalElapsed: number, stgs: TimerStage[]) => {
    let remaining = totalElapsed
    for (let i = 0; i < stgs.length; i++) {
      if (remaining < stgs[i].durationS) {
        return { stageIndex: i, stageElapsed: remaining }
      }
      remaining -= stgs[i].durationS
    }
    return { stageIndex: stgs.length - 1, stageElapsed: stgs[stgs.length - 1].durationS }
  }, [])

  const tick = useCallback(() => {
    const now          = Date.now()
    const totalElapsed = accumulatedRef.current + (now - startTimeRef.current) / 1000

    if (totalElapsed >= totalDuration) {
      // Complete
      accumulatedRef.current = totalDuration
      setElapsed(totalDuration)
      setStageIndex(stages.length - 1)
      setStatus('complete')
      if (intervalRef.current) clearInterval(intervalRef.current)
      playCompletionBells()
      onComplete?.()
      return
    }

    const pos = getStagePos(totalElapsed, stages)

    // Bell on stage advance
    setStageIndex(prev => {
      if (pos.stageIndex > prev) {
        playBell(528 + pos.stageIndex * 111, 1.8)
      }
      return pos.stageIndex
    })

    setElapsed(totalElapsed)
  }, [stages, totalDuration, getStagePos, onComplete])

  const start = useCallback(() => {
    if (status !== 'idle') return
    accumulatedRef.current = 0
    startTimeRef.current   = Date.now()
    setElapsed(0)
    setStageIndex(0)
    setStatus('running')
    intervalRef.current = setInterval(tick, 250)
    playBell(432, 1.5)
  }, [status, tick])

  const pause = useCallback(() => {
    if (status !== 'running') return
    accumulatedRef.current += (Date.now() - startTimeRef.current) / 1000
    if (intervalRef.current) clearInterval(intervalRef.current)
    setStatus('paused')
  }, [status])

  const resume = useCallback(() => {
    if (status !== 'paused') return
    startTimeRef.current = Date.now()
    setStatus('running')
    intervalRef.current = setInterval(tick, 250)
  }, [status, tick])

  const reset = useCallback(() => {
    if (intervalRef.current) clearInterval(intervalRef.current)
    accumulatedRef.current = 0
    setElapsed(0)
    setStageIndex(0)
    setStatus('idle')
  }, [])

  // Re-wire interval when tick changes (stages updated)
  useEffect(() => {
    if (status === 'running') {
      if (intervalRef.current) clearInterval(intervalRef.current)
      intervalRef.current = setInterval(tick, 250)
    }
    return () => { if (intervalRef.current) clearInterval(intervalRef.current) }
  }, [tick, status])

  const pos           = getStagePos(elapsed, stages)
  const currentStage  = stages[pos.stageIndex] ?? stages[0]
  const stageElapsed  = pos.stageElapsed
  const stageRemaining = Math.max(0, currentStage.durationS - stageElapsed)
  const totalProgress  = totalDuration > 0 ? Math.min(1, elapsed / totalDuration) : 0
  const stageProgress  = currentStage.durationS > 0
    ? Math.min(1, stageElapsed / currentStage.durationS)
    : 0

  return {
    status, stageIndex: pos.stageIndex, currentStage,
    stageElapsed, stageRemaining,
    totalElapsed: elapsed, totalDuration,
    totalProgress, stageProgress,
    stages, start, pause, resume, reset, setStages,
  }
}
