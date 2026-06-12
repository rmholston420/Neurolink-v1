/**
 * useArtifactStats
 *
 * Accumulates per-frame Stage 3 artifact decisions over a rolling window
 * (default 300 frames ≈ 75 s at 4 Hz) and returns live statistics.
 *
 * Returns:
 *   totalFrames     — frames observed since mount (or last reset)
 *   rejectedFrames  — frames where artifact_rejected === true
 *   rejectRate      — 0-1 fraction of rejected frames in window
 *   causeCounts     — { amplitude: n, motion: n, kurtosis: n, ... }
 *   windowSize      — configurable rolling window length in frames
 *   reset           — call to clear all accumulators
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import type { NeurolinkState } from '../types'

export interface ArtifactStats {
  totalFrames:    number
  rejectedFrames: number
  rejectRate:     number          // 0-1
  causeCounts:    Record<string, number>
  windowSize:     number
  reset:          () => void
}

const DEFAULT_WINDOW = 300  // frames

export function useArtifactStats(
  state: NeurolinkState | null,
  windowSize = DEFAULT_WINDOW,
): ArtifactStats {
  // Ring-buffer of { rejected, reasons } per frame
  const bufRef = useRef<Array<{ rejected: boolean; reasons: string[] }>>([])

  const [stats, setStats] = useState<Omit<ArtifactStats, 'reset'>>(() => ({
    totalFrames: 0,
    rejectedFrames: 0,
    rejectRate: 0,
    causeCounts: {},
    windowSize,
  }))

  // Track last frame_count to avoid double-counting on re-renders
  const lastFrameRef = useRef<number>(-1)

  useEffect(() => {
    if (!state || !state.connected) return
    if (state.frame_count === lastFrameRef.current) return
    lastFrameRef.current = state.frame_count

    const buf = bufRef.current
    buf.push({ rejected: state.artifact_rejected, reasons: state.artifact_reasons ?? [] })

    // Trim to rolling window
    if (buf.length > windowSize) buf.splice(0, buf.length - windowSize)

    // Recompute stats over window
    let rejected = 0
    const causeCounts: Record<string, number> = {}
    for (const f of buf) {
      if (f.rejected) {
        rejected++
        for (const r of f.reasons) {
          causeCounts[r] = (causeCounts[r] ?? 0) + 1
        }
      }
    }

    setStats({
      totalFrames:    buf.length,
      rejectedFrames: rejected,
      rejectRate:     buf.length > 0 ? rejected / buf.length : 0,
      causeCounts,
      windowSize,
    })
  }, [state, windowSize])

  const reset = useCallback(() => {
    bufRef.current = []
    lastFrameRef.current = -1
    setStats({ totalFrames: 0, rejectedFrames: 0, rejectRate: 0, causeCounts: {}, windowSize })
  }, [windowSize])

  return { ...stats, reset }
}
