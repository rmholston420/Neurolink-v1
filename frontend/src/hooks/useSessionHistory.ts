/**
 * useSessionHistory
 *
 * Accumulates per-session records in memory (no localStorage —
 * sandboxed iframe restriction).
 *
 * A session record is written when:
 *   1. The device disconnects (connected transitions false → true then back)
 *   2. Or explicitly via flushSession() (called by SessionHistoryPanel)
 *
 * Each record captures:
 *   date, durationS, meanFocus, meanAlpha, ea1EligiblePct,
 *   meanEngagement, peakAlpha, alchemicalStage
 *
 * Also exposes a 7-day sparkline array and a heatmap grid (52w × 7d)
 * compatible with a GitHub-style contribution calendar.
 */
import { useRef, useState, useEffect, useCallback } from 'react'
import type { NeurolinkState } from '../types'

export interface SessionRecord {
  id:              string
  date:            string    // ISO date YYYY-MM-DD
  startTs:         number
  durationS:       number
  meanFocus:       number
  meanAlpha:       number
  peakAlpha:       number
  meanEngagement:  number
  ea1EligiblePct:  number
  alchemicalStage: string
  frameCount:      number
}

export interface SessionHistoryReturn {
  sessions:      SessionRecord[]
  sparkline7d:   number[]    // last 7 days mean focus, one entry per day
  heatmap:       number[][]  // [week][day] = session count, 52 weeks x 7 days
  flushSession:  () => void
  exportCSV:     () => void
}

function todayISO(): string {
  return new Date().toISOString().slice(0, 10)
}

function buildSparkline(sessions: SessionRecord[]): number[] {
  const days: Record<string, number[]> = {}
  sessions.forEach(s => {
    if (!days[s.date]) days[s.date] = []
    days[s.date].push(s.meanFocus)
  })
  return Array.from({ length: 7 }, (_, i) => {
    const d = new Date()
    d.setDate(d.getDate() - (6 - i))
    const key = d.toISOString().slice(0, 10)
    const vals = days[key]
    return vals ? vals.reduce((a, b) => a + b, 0) / vals.length : 0
  })
}

function buildHeatmap(sessions: SessionRecord[]): number[][] {
  const counts: Record<string, number> = {}
  sessions.forEach(s => { counts[s.date] = (counts[s.date] ?? 0) + 1 })
  // 52 weeks x 7 days, ending today
  const today = new Date()
  return Array.from({ length: 52 }, (_, w) =>
    Array.from({ length: 7 }, (__, d) => {
      const date = new Date(today)
      date.setDate(date.getDate() - ((51 - w) * 7 + (6 - d)))
      return counts[date.toISOString().slice(0, 10)] ?? 0
    })
  )
}

export function useSessionHistory(
  state: Partial<NeurolinkState> | null,
): SessionHistoryReturn {
  const [sessions,    setSessions]    = useState<SessionRecord[]>([])
  const [sparkline7d, setSparkline7d] = useState<number[]>(Array(7).fill(0))
  const [heatmap,     setHeatmap]     = useState<number[][]>(Array.from({ length: 52 }, () => Array(7).fill(0)))

  // In-session accumulators
  const focusAccRef  = useRef<number[]>([])
  const alphaAccRef  = useRef<number[]>([])
  const engageAccRef = useRef<number[]>([])
  const ea1AccRef    = useRef<boolean[]>([])
  const stageRef     = useRef<string>('Nigredo')
  const sessionStartRef = useRef<number>(Date.now())
  const prevConnRef  = useRef<boolean>(false)
  const prevFrameRef = useRef<number>(0)

  const commitSession = useCallback((frames: number) => {
    const focusArr  = focusAccRef.current
    const alphaArr  = alphaAccRef.current
    if (focusArr.length < 5) return   // too short to record

    const meanFocus      = focusArr.reduce((a, b) => a + b, 0) / focusArr.length
    const meanAlpha      = alphaArr.reduce((a, b) => a + b, 0) / alphaArr.length
    const peakAlpha      = Math.max(...alphaArr)
    const meanEngagement = engageAccRef.current.reduce((a, b) => a + b, 0) / Math.max(1, engageAccRef.current.length)
    const ea1Pct         = ea1AccRef.current.filter(Boolean).length / Math.max(1, ea1AccRef.current.length) * 100
    const durationS      = (Date.now() - sessionStartRef.current) / 1000

    const record: SessionRecord = {
      id:              `${Date.now()}`,
      date:            todayISO(),
      startTs:         sessionStartRef.current,
      durationS,
      meanFocus,
      meanAlpha,
      peakAlpha,
      meanEngagement,
      ea1EligiblePct:  ea1Pct,
      alchemicalStage: stageRef.current,
      frameCount:      frames,
    }

    setSessions(prev => {
      const updated = [record, ...prev].slice(0, 200) // keep last 200
      setSparkline7d(buildSparkline(updated))
      setHeatmap(buildHeatmap(updated))
      return updated
    })

    // Reset accumulators
    focusAccRef.current  = []
    alphaAccRef.current  = []
    engageAccRef.current = []
    ea1AccRef.current    = []
    sessionStartRef.current = Date.now()
  }, [])

  useEffect(() => {
    if (!state) return

    const connected = state.connected ?? false
    const frame     = state.frame_count ?? 0

    // Accumulate
    if (connected) {
      focusAccRef.current.push(state.focus_score ?? 0)
      alphaAccRef.current.push(state.bands?.alpha ?? 0)
      engageAccRef.current.push(state.engagement_index ?? 0)
      ea1AccRef.current.push(state.ea1?.eligible ?? false)
      stageRef.current = state.alchemical_stage ?? stageRef.current
    }

    // Commit on disconnect
    if (prevConnRef.current && !connected) {
      commitSession(frame)
    }

    // Commit on session reset (frame_count goes back to 0)
    if (frame < prevFrameRef.current && prevFrameRef.current > 30) {
      commitSession(prevFrameRef.current)
    }

    prevConnRef.current  = connected
    prevFrameRef.current = frame
  }, [state?.frame_count, state?.connected, commitSession])

  const flushSession = useCallback(() => {
    commitSession(prevFrameRef.current)
  }, [commitSession])

  const exportCSV = useCallback(() => {
    if (sessions.length === 0) return
    const header = 'Date,Duration(s),MeanFocus,MeanAlpha,PeakAlpha,EA1%,Stage,Frames\n'
    const rows = sessions.map(s =>
      `${s.date},${s.durationS.toFixed(0)},${s.meanFocus.toFixed(3)},${s.meanAlpha.toFixed(4)},${s.peakAlpha.toFixed(4)},${s.ea1EligiblePct.toFixed(1)},${s.alchemicalStage},${s.frameCount}`
    ).join('\n')
    const blob = new Blob([header + rows], { type: 'text/csv' })
    const a    = document.createElement('a')
    a.href     = URL.createObjectURL(blob)
    a.download = `neurolink-sessions-${todayISO()}.csv`
    a.click()
    URL.revokeObjectURL(a.href)
  }, [sessions])

  return { sessions, sparkline7d, heatmap, flushSession, exportCSV }
}
