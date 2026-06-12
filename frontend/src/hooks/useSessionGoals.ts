/**
 * useSessionGoals — Tier 2
 * Manages a pre-session goal (target duration + EA-1%), tracks live progress,
 * awards achievement badges on completion, persists badge list across mounts.
 */
import { useState, useRef, useEffect } from 'react'
import type { NeurolinkState } from '../types'

export interface SessionGoal {
  targetMinutes:  number    // desired session duration
  targetEA1Pct:   number    // desired % of time in EA-1 eligibility (0–100)
}

export interface Badge {
  id:        string
  label:     string
  emoji:     string
  earnedAt:  number
  detail:    string
}

export interface SessionGoalsResult {
  goal:            SessionGoal | null
  setGoal:         (g: SessionGoal) => void
  clearGoal:       () => void
  elapsedSec:      number
  ea1Frames:       number
  totalFrames:     number
  ea1Pct:          number    // live % of frames that were EA-1 eligible
  durationProgress: number   // 0–1
  ea1Progress:     number    // 0–1
  badges:          Badge[]
  isRunning:       boolean
  startSession:    () => void
  endSession:      () => void
}

const BADGE_STORE: Badge[] = []

const BADGE_DEFS: Array<{ id: string; label: string; emoji: string; detail: string;
  test: (el: number, ea1Pct: number, goal: SessionGoal) => boolean }> = [
  { id: 'first_session',  label: 'First Session',  emoji: '🌱', detail: 'Completed your first timed session',
    test: (el) => el >= 60 },
  { id: 'five_min',       label: '5-Minute Sit',   emoji: '⏱',  detail: 'Meditated for 5 continuous minutes',
    test: (el) => el >= 300 },
  { id: 'twenty_min',     label: '20-Minute Sit',  emoji: '🧘', detail: 'Meditated for 20 continuous minutes',
    test: (el) => el >= 1200 },
  { id: 'ea1_50',         label: 'EA-1 Achiever',  emoji: '⚡', detail: '50%+ of session time in EA-1',
    test: (_e, ea1) => ea1 >= 50 },
  { id: 'ea1_75',         label: 'EA-1 Master',    emoji: '🔮', detail: '75%+ of session time in EA-1',
    test: (_e, ea1) => ea1 >= 75 },
  { id: 'goal_met',       label: 'Goal Achieved',  emoji: '🏆', detail: 'Met both duration and EA-1 targets',
    test: (el, ea1, g) => el >= g.targetMinutes * 60 && ea1 >= g.targetEA1Pct },
]

export function useSessionGoals(state: NeurolinkState | null): SessionGoalsResult {
  const [goal,       setGoalState]   = useState<SessionGoal | null>(null)
  const [isRunning,  setIsRunning]   = useState(false)
  const [elapsedSec, setElapsedSec]  = useState(0)
  const [ea1Frames,  setEa1Frames]   = useState(0)
  const [totalFrames, setTotalFrames] = useState(0)
  const [badges,     setBadges]      = useState<Badge[]>(BADGE_STORE)

  const stateRef   = useRef(state)
  stateRef.current = state

  const goalRef = useRef(goal)
  goalRef.current = goal

  // Tick every second when running
  useEffect(() => {
    if (!isRunning) return
    const id = setInterval(() => {
      const s = stateRef.current
      setElapsedSec(t => t + 1)
      setTotalFrames(n => n + 1)
      if (s?.ea1?.eligible) setEa1Frames(n => n + 1)
    }, 1000)
    return () => clearInterval(id)
  }, [isRunning])

  // Badge evaluation on every state update
  useEffect(() => {
    if (!isRunning || !goalRef.current) return
    const ea1Pct = totalFrames > 0 ? (ea1Frames / totalFrames) * 100 : 0
    const g = goalRef.current
    BADGE_DEFS.forEach(def => {
      const alreadyEarned = BADGE_STORE.some(b => b.id === def.id)
      if (!alreadyEarned && def.test(elapsedSec, ea1Pct, g)) {
        const badge: Badge = {
          id: def.id, label: def.label, emoji: def.emoji,
          earnedAt: Date.now(), detail: def.detail,
        }
        BADGE_STORE.push(badge)
        setBadges([...BADGE_STORE])
      }
    })
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [elapsedSec, isRunning])

  const ea1Pct = totalFrames > 0 ? (ea1Frames / totalFrames) * 100 : 0
  const g = goal
  const durationProgress = g ? Math.min(1, elapsedSec / (g.targetMinutes * 60)) : 0
  const ea1Progress      = g ? Math.min(1, ea1Pct / g.targetEA1Pct)              : 0

  function startSession() {
    setElapsedSec(0)
    setEa1Frames(0)
    setTotalFrames(0)
    setIsRunning(true)
  }

  function endSession() {
    setIsRunning(false)
  }

  function setGoal(g: SessionGoal) {
    setGoalState(g)
  }

  function clearGoal() {
    setGoalState(null)
    setIsRunning(false)
  }

  return {
    goal, setGoal, clearGoal,
    elapsedSec, ea1Frames, totalFrames, ea1Pct,
    durationProgress, ea1Progress,
    badges, isRunning, startSession, endSession,
  }
}
