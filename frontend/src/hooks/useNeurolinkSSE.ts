import { useEffect, useRef, useState } from 'react'
import type { NeurolinkState } from '../types'

// ─── Sentinel shape emitted by the backend once baseline is ready ────────────
export interface BaselineCompleteSentinel {
  type:         'baseline_complete'
  bands:        Record<string, number>   // mean band-powers captured during baseline
  focus_score:  number
  fatigue_score: number
  sample_count: number
  duration_s:   number
}

// ─── Union of everything the SSE stream can emit ────────────────────────────
type SSEFrame = NeurolinkState | BaselineCompleteSentinel

/**
 * Type guard: true only for the baseline_complete sentinel dict.
 *
 * All normal NeurolinkState frames lack a `type` field, so checking for its
 * presence and value is the correct discriminator — no backend schema change
 * is required.
 */
export function isBaselineComplete(frame: SSEFrame): frame is BaselineCompleteSentinel {
  return (
    typeof (frame as BaselineCompleteSentinel).type === 'string' &&
    (frame as BaselineCompleteSentinel).type === 'baseline_complete'
  )
}

export interface UseNeurolinkSSEOptions {
  /**
   * Called once, the first time a baseline_complete sentinel arrives.
   * Subsequent sentinels (re-calibration) also fire this callback.
   */
  onBaselineComplete?: (sentinel: BaselineCompleteSentinel) => void
}

/**
 * SSE consumer hook for Neurolink stream.
 *
 * The backend emits named SSE events:  event: state\ndata: <json>\n\n
 * EventSource.onmessage only fires for un-named events (event: message).
 * We must use addEventListener('state', handler) to receive named events.
 *
 * Frame taxonomy
 * ──────────────
 *   Normal frame  →  NeurolinkState  →  returned as hook state
 *   Sentinel      →  BaselineCompleteSentinel  →  onBaselineComplete() called,
 *                                                   hook state is NOT mutated
 *
 * Auto-reconnects after disconnection with 3-second back-off.
 */
export function useNeurolinkSSE(
  url: string,
  options?: UseNeurolinkSSEOptions,
): NeurolinkState | null {
  const [state, setState] = useState<NeurolinkState | null>(null)
  const esRef        = useRef<EventSource | null>(null)
  const cancelledRef = useRef(false)
  // Stable ref so the SSE handler can call the latest callback without
  // needing to be re-registered whenever the parent re-renders.
  const onBaselineRef = useRef(options?.onBaselineComplete)
  onBaselineRef.current = options?.onBaselineComplete

  useEffect(() => {
    cancelledRef.current = false

    function connect() {
      if (cancelledRef.current) return

      const es = new EventSource(url)
      esRef.current = es

      const stateHandler = (event: MessageEvent) => {
        let frame: SSEFrame
        try {
          frame = JSON.parse(event.data) as SSEFrame
        } catch {
          return // discard malformed JSON
        }

        if (isBaselineComplete(frame)) {
          // Route sentinel to the callback; never touch NeurolinkState atom.
          onBaselineRef.current?.(frame)
        } else {
          // Normal frame — set app state as before.
          setState(frame)
        }
      }

      es.addEventListener('state', stateHandler)
      // Fallback: some SSE proxies strip the event: field
      es.onmessage = stateHandler

      es.onerror = () => {
        es.removeEventListener('state', stateHandler)
        es.close()
        esRef.current = null
        if (!cancelledRef.current) {
          setTimeout(connect, 3000)
        }
      }
    }

    connect()

    return () => {
      cancelledRef.current = true
      const es = esRef.current
      if (es) {
        es.close()
        esRef.current = null
      }
    }
  }, [url]) // options object intentionally excluded — use the ref

  return state
}
