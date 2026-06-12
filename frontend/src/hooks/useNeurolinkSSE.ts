import { useCallback, useEffect, useRef, useState } from 'react'
import type { NeurolinkState, SettlingReason, SettlingSentinel, BaselineCompleteSentinel } from '../types'

export type { SettlingReason, SettlingSentinel, BaselineCompleteSentinel }

// ─── Type guards ─────────────────────────────────────────────────────────────

/**
 * True when the SSE data payload is a settling sentinel.
 * Used internally and re-exported for consumers that subscribe via onmessage
 * fallback paths.
 */
export function isSettling(data: unknown): data is SettlingSentinel {
  return (
    typeof data === 'object' &&
    data !== null &&
    (data as SettlingSentinel).event === 'settling'
  )
}

/**
 * True when the SSE data payload is a baseline_complete sentinel.
 * Checks the 'event' field (backend shape) as primary discriminator.
 * Also accepts the legacy 'type' field for backwards compatibility with
 * clients that serialised the sentinel with a 'type' key.
 */
export function isBaselineComplete(data: unknown): data is BaselineCompleteSentinel {
  if (typeof data !== 'object' || data === null) return false
  const d = data as Record<string, unknown>
  return d['event'] === 'baseline_complete' || d['type'] === 'baseline_complete'
}

// ─── Hook options ─────────────────────────────────────────────────────────────

export interface UseNeurolinkSSEOptions {
  /**
   * Called once each time a baseline_complete sentinel arrives.
   * Fired on the named 'baseline_complete' SSE event.
   */
  onBaselineComplete?: () => void
  /**
   * Called on every frame the Stage 0 acquisition guard holds.
   * Reason codes:
   *   'impedance_unstable'  — electrode contact not stable
   *   'motion_settling'     — movement detected; waiting for rest
   *   'env_not_ready'       — environment calibration pending
   *   'settling'            — generic fallback
   */
  onSettling?: (reason: SettlingReason) => void
}

// ─── Return type ─────────────────────────────────────────────────────────────

export interface UseNeurolinkSSEResult {
  /** Latest NeurolinkState tick, or null before the first frame arrives. */
  state: NeurolinkState | null
  /**
   * Most-recent settling reason if the device is currently in Stage 0
   * acquisition hold, otherwise null.  Auto-clears 2 s after the last
   * settling event so the indicator fades out once acquisition resumes.
   */
  settlingReason: SettlingReason | null
}

// How long to keep settlingReason non-null after the last settling event.
const SETTLING_CLEAR_MS = 2_000

/**
 * SSE consumer hook for the Neurolink /neurolink/stream endpoint.
 *
 * The backend emits three named SSE event types on the same stream:
 *
 *   event: state             →  NeurolinkState JSON
 *   event: baseline_complete →  {} (one-shot; bell + unlock UI)
 *   event: settling          →  {"reason": "<SettlingReason>"}
 *
 * EventSource.onmessage only fires for un-named (event: message) events.
 * This hook registers named listeners for all three types and falls back to
 * onmessage (data-level 'event' key discrimination) for proxies that strip
 * the SSE event field.
 *
 * Auto-reconnects after disconnection with a 3-second back-off.
 *
 * @param url     Full URL of the SSE stream endpoint.
 * @param options Callbacks for sentinel events.
 * @returns       { state, settlingReason }
 */
export function useNeurolinkSSE(
  url: string,
  options?: UseNeurolinkSSEOptions,
): UseNeurolinkSSEResult {
  const [state, setState]                       = useState<NeurolinkState | null>(null)
  const [settlingReason, setSettlingReason]     = useState<SettlingReason | null>(null)

  const esRef             = useRef<EventSource | null>(null)
  const cancelledRef      = useRef(false)
  const clearTimerRef     = useRef<ReturnType<typeof setTimeout> | null>(null)

  // Stable refs so event handlers always call the latest callbacks without
  // needing to be re-registered whenever the parent re-renders.
  const onBaselineRef = useRef(options?.onBaselineComplete)
  const onSettlingRef = useRef(options?.onSettling)
  onBaselineRef.current = options?.onBaselineComplete
  onSettlingRef.current = options?.onSettling

  // Debounced settling-reason clear: resets the 2 s timer on each event.
  const scheduleSettlingClear = useCallback(() => {
    if (clearTimerRef.current !== null) clearTimeout(clearTimerRef.current)
    clearTimerRef.current = setTimeout(() => {
      setSettlingReason(null)
      clearTimerRef.current = null
    }, SETTLING_CLEAR_MS)
  }, [])

  useEffect(() => {
    cancelledRef.current = false

    function connect() {
      if (cancelledRef.current) return

      const es = new EventSource(url)
      esRef.current = es

      // ── Named event handlers ─────────────────────────────────────────────

      const stateHandler = (event: MessageEvent) => {
        try {
          setState(JSON.parse(event.data) as NeurolinkState)
        } catch {
          // discard malformed JSON
        }
      }

      const baselineCompleteHandler = (_event: MessageEvent) => {
        onBaselineRef.current?.()
      }

      const settlingHandler = (event: MessageEvent) => {
        let reason: SettlingReason = 'settling'
        try {
          const parsed = JSON.parse(event.data) as { reason?: SettlingReason }
          if (parsed.reason) reason = parsed.reason
        } catch {
          // use fallback reason
        }
        setSettlingReason(reason)
        scheduleSettlingClear()
        onSettlingRef.current?.(reason)
      }

      es.addEventListener('state', stateHandler)
      es.addEventListener('baseline_complete', baselineCompleteHandler)
      es.addEventListener('settling', settlingHandler)

      // ── Fallback: proxies that strip the event: field ────────────────────
      // Discriminate on the 'event' key in the data payload.
      es.onmessage = (event: MessageEvent) => {
        let parsed: Record<string, unknown>
        try {
          parsed = JSON.parse(event.data) as Record<string, unknown>
        } catch {
          return
        }
        if (parsed['event'] === 'baseline_complete') {
          onBaselineRef.current?.()
        } else if (parsed['event'] === 'settling') {
          const reason = (parsed['reason'] as SettlingReason | undefined) ?? 'settling'
          setSettlingReason(reason)
          scheduleSettlingClear()
          onSettlingRef.current?.(reason)
        } else {
          // Assume NeurolinkState (normal frame or proxy-stripped named event)
          setState(parsed as unknown as NeurolinkState)
        }
      }

      es.onerror = () => {
        es.removeEventListener('state', stateHandler)
        es.removeEventListener('baseline_complete', baselineCompleteHandler)
        es.removeEventListener('settling', settlingHandler)
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
      if (clearTimerRef.current !== null) {
        clearTimeout(clearTimerRef.current)
        clearTimerRef.current = null
      }
      const es = esRef.current
      if (es) {
        es.close()
        esRef.current = null
      }
    }
  }, [url, scheduleSettlingClear])

  return { state, settlingReason }
}
