/**
 * useSessionGoals
 *
 * Tracks session goals and awards achievement badges.
 *
 * Goals (all optional, user-configurable):
 *   - Target duration in minutes
 *   - EA-1 eligibility % target
 *   - Focus % target
 *
 * Progress is computed live from the current session.
 *
 * Achievements (earned in-memory, persist across sessions this browser session):
 *   FIRST_EA1        First time ea1.eligible becomes true
 *   DEEP_FOCUS       focus_score >= 0.8 sustained for 60 consecutive frames
 *   LONG_SIT         Session duration >= 20 minutes
 *   COHERENCE_MASTER HRV coherence score >= 66 for 30 consecutive frames
 *   NIGREDO_ESCAPE   First stage transition out of Nigredo
 *   RUBEDO_TOUCH     alchemical_stage === 'Rubedo' for the first time
 */
import { useState, useRef, useEffect, useCallback } from 'react'
import type { NeurolinkState } from '../types'

export interface SessionGoalConfig {
  durationMinTarget:  number | null   // minutes
  ea1PctTarget:       number | null   // 0–100
  focusPctTarget:     number | null   // 0–100
}

export interface GoalProgress {
  durationMin:    number
  durationPct:    number   // 0–1 toward target
  ea1Pct:         number
  ea1ProgressPct: number   // 0–1 toward target
  focusPct:       number
  focusProgressPct: number // 0–1 toward target
  allGoalsMet:    boolean
}

export type AchievementId =
  | 'FIRST_EA1'
  | 'DEEP_FOCUS'
  | 'LONG_SIT'
  | 'COHERENCE_MASTER'
  | 'NIGREDO_ESCAPE'
  | 'RUBEDO_TOUCH'

export interface Achievement {
  id:          AchievementId
  label:       string
  description: string
  icon:        string
  earnedAt:    number | null
}

const ACHIEVEMENT_DEFS: Omit<Achievement, 'earnedAt'>[] = [
  { id: 'FIRST_EA1',        icon: '✨', label: 'First Light',       description: 'EA-1 eligibility reached for the first time' },
  { id: 'DEEP_FOCUS',       icon: '🎯', label: 'Deep Focus',        description: 'Maintained focus ≥ 80% for 60 consecutive frames' },
  { id: 'LONG_SIT',         icon: '🪷', label: 'Long Sit',          description: 'Completed a session of 20+ minutes' },
  { id: 'COHERENCE_MASTER', icon: '💚', label: 'Coherence Master',  description: 'Sustained HRV coherence ≥ 66 for 30 consecutive frames' },
  { id: 'NIGREDO_ESCAPE',   icon: '🌅', label: 'Dawn Breaking',     description: 'First transition out of Nigredo stage' },
  { id: 'RUBEDO_TOUCH',     icon: '🔴', label: 'Rubedo Touch',      description: 'Reached the Rubedo alchemical stage' },
]

export interface SessionGoalsReturn {
  goals:          SessionGoalConfig
  setGoals:       (g: Partial<SessionGoalConfig>) => void
  progress:       GoalProgress
  achievements:   Achievement[]
  newAchievement: Achievement | null   // most recently earned, shown as toast
  clearToast:     () => void
}

export function useSessionGoals(
  state: Partial<NeurolinkState> | null,
  coherenceScore?: number,
): SessionGoalsReturn {
  const [goals, setGoalsState] = useState<SessionGoalConfig>({
    durationMinTarget: 20,
    ea1PctTarget:      30,
    focusPctTarget:    50,
  })

  const [achievements, setAchievements] = useState<Achievement[]>(
    ACHIEVEMENT_DEFS.map(a => ({ ...a, earnedAt: null }))
  )
  const [newAchievement, setNewAchievement] = useState<Achievement | null>(null)

  // Session accumulators
  const sessionStartRef   = useRef<number>(Date.now())
  const ea1FramesRef      = useRef<number>(0)
  const totalFramesRef    = useRef<number>(0)
  const focusSumRef       = useRef<number>(0)
  const deepFocusRunRef   = useRef<number>(0)
  const coherenceRunRef   = useRef<number>(0)
  const prevStageRef      = useRef<string>('')
  const prevFrameRef      = useRef<number>(0)

  const earn = useCallback((id: AchievementId) => {
    setAchievements(prev => {
      const existing = prev.find(a => a.id === id)
      if (existing?.earnedAt !== null) return prev  // already earned
      const updated = prev.map(a =>
        a.id === id ? { ...a, earnedAt: Date.now() } : a
      )
      const earned = updated.find(a => a.id === id)!
      setNewAchievement(earned)
      return updated
    })
  }, [])

  useEffect(() => {
    if (!state) return
    const frame     = state.frame_count  ?? 0
    const ea1ok     = state.ea1?.eligible ?? false
    const focus     = state.focus_score   ?? 0
    const stage     = state.alchemical_stage ?? ''
    const connected = state.connected ?? false

    if (!connected) return

    // Detect session reset
    if (frame < prevFrameRef.current && prevFrameRef.current > 10) {
      sessionStartRef.current  = Date.now()
      ea1FramesRef.current     = 0
      totalFramesRef.current   = 0
      focusSumRef.current      = 0
      deepFocusRunRef.current  = 0
      coherenceRunRef.current  = 0
    }
    prevFrameRef.current = frame

    totalFramesRef.current++
    if (ea1ok) ea1FramesRef.current++
    focusSumRef.current += focus

    // Deep focus run
    if (focus >= 0.8) {
      deepFocusRunRef.current++
      if (deepFocusRunRef.current >= 60) earn('DEEP_FOCUS')
    } else {
      deepFocusRunRef.current = 0
    }

    // Coherence run
    if ((coherenceScore ?? 0) >= 66) {
      coherenceRunRef.current++
      if (coherenceRunRef.current >= 30) earn('COHERENCE_MASTER')
    } else {
      coherenceRunRef.current = 0
    }

    // EA-1 first time
    if (ea1ok) earn('FIRST_EA1')

    // Long sit (20 min)
    const durationMin = (Date.now() - sessionStartRef.current) / 60000
    if (durationMin >= 20) earn('LONG_SIT')

    // Stage transitions
    if (stage && stage !== prevStageRef.current) {
      if (prevStageRef.current === 'Nigredo') earn('NIGREDO_ESCAPE')
      if (stage === 'Rubedo') earn('RUBEDO_TOUCH')
      prevStageRef.current = stage
    }
  }, [state?.frame_count, coherenceScore, earn])

  const setGoals = useCallback((g: Partial<SessionGoalConfig>) => {
    setGoalsState(prev => ({ ...prev, ...g }))
  }, [])

  const clearToast = useCallback(() => setNewAchievement(null), [])

  // Live progress
  const durationMin    = (Date.now() - sessionStartRef.current) / 60000
  const ea1Pct         = totalFramesRef.current > 0
    ? (ea1FramesRef.current / totalFramesRef.current) * 100 : 0
  const focusPct       = totalFramesRef.current > 0
    ? (focusSumRef.current / totalFramesRef.current) * 100 : 0

  const durTarget   = goals.durationMinTarget ?? 0
  const ea1Target   = goals.ea1PctTarget      ?? 0
  const focusTarget = goals.focusPctTarget    ?? 0

  const progress: GoalProgress = {
    durationMin,
    durationPct:     durTarget   > 0 ? Math.min(1, durationMin / durTarget)    : 0,
    ea1Pct,
    ea1ProgressPct:  ea1Target   > 0 ? Math.min(1, ea1Pct      / ea1Target)    : 0,
    focusPct,
    focusProgressPct: focusTarget > 0 ? Math.min(1, focusPct    / focusTarget)  : 0,
    allGoalsMet: (
      (durTarget   === 0 || durationMin >= durTarget) &&
      (ea1Target   === 0 || ea1Pct      >= ea1Target) &&
      (focusTarget === 0 || focusPct    >= focusTarget)
    ),
  }

  return { goals, setGoals, progress, achievements, newAchievement, clearToast }
}
