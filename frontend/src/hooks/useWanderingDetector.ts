/**
 * useWanderingDetector
 *
 * Detects mind-wandering events from the live SSE stream.
 *
 * A wandering event fires when:
 *   engagement_index > rolling_mean + k * rolling_sigma
 * where k = 1.2 by default (configurable via sensitivity).
 *
 * Tracks:
 *   events[]          timestamped list of wandering events with recovery time
 *   sessionStats      aggregate stats for the current session
 *   timeline          50-bucket array of normalised focus (0–1) for the session bar
 *
 * Resets automatically when frame_count resets (new session detected).
 */
import { useRef, useState, useEffect } from 'react'
import type { NeurolinkState } from '../types'

export interface WanderEvent {
  timestamp:    number      // ms since epoch
  engageValue:  number      // engagement_index at spike
  focusAtEvent: number      // focus_score at event
  recoveryMs:   number | null  // ms until engagement returned below threshold
}

export interface SessionStats {
  eventCount:       number
  meanRecoveryMs:   number | null
  longestFocusRunS: number        // longest uninterrupted focus period in seconds
  sessionDurationS: number
  ea1EligiblePct:   number        // % of frames where ea1.eligible was true
}

export interface WanderingDetectorReturn {
  events:       WanderEvent[]
  sessionStats: SessionStats
  timeline:     number[]   // 50 buckets, each = mean focus_score for that segment
  isWandering:  boolean
}

const HISTORY_LEN  = 60   // frames for rolling stats
const TIMELINE_BUCKETS = 50
const K_SENSITIVITY   = 1.2
const COOLDOWN_FRAMES = 15  // min frames between events

export function useWanderingDetector(
  state: Partial<NeurolinkState> | null,
): WanderingDetectorReturn {
  const [events,       setEvents]       = useState<WanderEvent[]>([])
  const [sessionStats, setSessionStats] = useState<SessionStats>({
    eventCount: 0, meanRecoveryMs: null,
    longestFocusRunS: 0, sessionDurationS: 0, ea1EligiblePct: 0,
  })
  const [timeline,     setTimeline]     = useState<number[]>(Array(TIMELINE_BUCKETS).fill(0))
  const [isWandering,  setIsWandering]  = useState(false)

  // Internal refs (no re-renders)
  const engHistRef      = useRef<number[]>([])
  const focusHistRef    = useRef<number[]>([])
  const ea1HistRef      = useRef<boolean[]>([])
  const timelineAccRef  = useRef<number[]>([])
  const prevFrameRef    = useRef<number>(0)
  const wanderStartRef  = useRef<number | null>(null)
  const cooldownRef     = useRef<number>(0)
  const sessionStartRef = useRef<number>(Date.now())
  const longestRunRef   = useRef<number>(0)
  const currentRunRef   = useRef<number>(0)
  const framesSinceWanderRef = useRef<number>(0)

  useEffect(() => {
    if (!state) return

    const frame = state.frame_count ?? 0
    const eng   = state.engagement_index ?? 0
    const focus = state.focus_score      ?? 0
    const ea1ok = state.ea1?.eligible    ?? false

    // Detect new session (frame_count reset)
    if (frame < prevFrameRef.current) {
      setEvents([])
      engHistRef.current     = []
      focusHistRef.current   = []
      ea1HistRef.current     = []
      timelineAccRef.current = []
      wanderStartRef.current = null
      cooldownRef.current    = 0
      longestRunRef.current  = 0
      currentRunRef.current  = 0
      sessionStartRef.current = Date.now()
      setTimeline(Array(TIMELINE_BUCKETS).fill(0))
      setSessionStats({ eventCount:0, meanRecoveryMs:null, longestFocusRunS:0, sessionDurationS:0, ea1EligiblePct:0 })
    }
    prevFrameRef.current = frame

    // Accumulate history
    engHistRef.current.push(eng)
    focusHistRef.current.push(focus)
    ea1HistRef.current.push(ea1ok)
    if (engHistRef.current.length > HISTORY_LEN) {
      engHistRef.current.shift()
      focusHistRef.current.shift()
      ea1HistRef.current.shift()
    }

    // Timeline accumulator (map each frame to a bucket)
    timelineAccRef.current.push(focus)
    const totalFrames = timelineAccRef.current.length
    const bucketSize  = Math.max(1, Math.floor(totalFrames / TIMELINE_BUCKETS))
    const tl = Array.from({ length: TIMELINE_BUCKETS }, (_, i) => {
      const start = i * bucketSize
      const slice = timelineAccRef.current.slice(start, start + bucketSize)
      if (slice.length === 0) return 0
      return slice.reduce((a, b) => a + b, 0) / slice.length
    })
    setTimeline(tl)

    // Rolling mean + sigma
    if (engHistRef.current.length < 10) return
    const mean  = engHistRef.current.reduce((a, b) => a + b, 0) / engHistRef.current.length
    const sigma = Math.sqrt(
      engHistRef.current.reduce((a, b) => a + (b - mean) ** 2, 0) / engHistRef.current.length
    )
    const threshold = mean + K_SENSITIVITY * sigma
    const wandering = eng > threshold

    setIsWandering(wandering)

    if (wandering) {
      if (wanderStartRef.current === null) {
        wanderStartRef.current = Date.now()
      }
      currentRunRef.current = 0
      framesSinceWanderRef.current = 0
      cooldownRef.current = COOLDOWN_FRAMES
    } else {
      if (wanderStartRef.current !== null) {
        // Recovered
        const recoveryMs = Date.now() - wanderStartRef.current
        const newEvent: WanderEvent = {
          timestamp:    wanderStartRef.current,
          engageValue:  eng,
          focusAtEvent: focus,
          recoveryMs,
        }
        setEvents(prev => {
          const updated = [...prev, newEvent]
          // Update session stats
          const recs = updated
            .filter(e => e.recoveryMs !== null)
            .map(e => e.recoveryMs as number)
          const meanRec = recs.length > 0
            ? recs.reduce((a, b) => a + b, 0) / recs.length
            : null
          const sessionDurationS = (Date.now() - sessionStartRef.current) / 1000
          const ea1Pct = ea1HistRef.current.filter(Boolean).length / Math.max(1, ea1HistRef.current.length) * 100
          setSessionStats({
            eventCount: updated.length,
            meanRecoveryMs: meanRec,
            longestFocusRunS: longestRunRef.current / 10, // approx 10 fps
            sessionDurationS,
            ea1EligiblePct: ea1Pct,
          })
          return updated
        })
        wanderStartRef.current = null
      }
      currentRunRef.current++
      if (currentRunRef.current > longestRunRef.current) {
        longestRunRef.current = currentRunRef.current
      }
    }
  }, [state?.frame_count])

  return { events, sessionStats, timeline, isWandering }
}
