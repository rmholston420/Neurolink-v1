import { useEffect, useRef, useState } from 'react'
import type { NeurolinkState } from '../types'

/**
 * SSE consumer hook for Neurolink stream.
 *
 * The backend emits named SSE events:  event: state\ndata: <json>\n\n
 * EventSource.onmessage only fires for un-named events (event: message).
 * We must use addEventListener('state', handler) to receive named events.
 *
 * Auto-reconnects after disconnection with 3-second back-off.
 */
export function useNeurolinkSSE(url: string): NeurolinkState | null {
  const [state, setState] = useState<NeurolinkState | null>(null)
  const esRef = useRef<EventSource | null>(null)
  const cancelledRef = useRef(false)

  useEffect(() => {
    cancelledRef.current = false

    function connect() {
      if (cancelledRef.current) return

      const es = new EventSource(url)
      esRef.current = es

      // Handle named 'state' events from the backend SSE stream
      const stateHandler = (event: MessageEvent) => {
        try {
          const data: NeurolinkState = JSON.parse(event.data)
          setState(data)
        } catch {
          // Ignore malformed JSON frames
        }
      }

      // Also handle generic message events as a fallback
      // (some SSE proxies strip the event: field)
      es.addEventListener('state', stateHandler)
      es.onmessage = stateHandler

      es.onerror = () => {
        es.removeEventListener('state', stateHandler)
        es.close()
        esRef.current = null
        if (!cancelledRef.current) {
          // Exponential-ish back-off capped at 3 s
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
  }, [url])

  return state
}
