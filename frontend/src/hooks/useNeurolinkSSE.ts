import { useEffect, useRef, useState } from 'react'

export interface BandPowers {
  alpha: number
  theta: number
  beta: number
  delta: number
  gamma: number
}

export interface EA1Result {
  eligible: boolean
  score: number
  criteria_met: number
  criteria_total: number
  label: string
  overlay_mode: string
}

export interface NeurolinkState {
  connected: boolean
  source: string
  region: string
  alchemical_stage: string
  region_v01: string
  alchemical_stage_v01: string
  integration_coverage: number
  engagement_index: number
  bands: BandPowers
  ea1: EA1Result
  frame_count: number
  poor_contact: boolean
  contact_quality: number | null
  hr_bpm: number | null
  hrv_rmssd: number | null
  rr_bpm: number | null
  pitch_deg: number | null
  roll_deg: number | null
  focus_state: string
  focus_score: number
  fatigue_score: number
  fnirs_oxy: number | null
  fnirs_deoxy: number | null
}

/**
 * SSE consumer hook for Neurolink stream.
 * Reconnects automatically after disconnection.
 */
export function useNeurolinkSSE(url: string): NeurolinkState | null {
  const [state, setState] = useState<NeurolinkState | null>(null)
  const esRef = useRef<EventSource | null>(null)

  useEffect(() => {
    let cancelled = false

    function connect() {
      if (cancelled) return
      const es = new EventSource(url)
      esRef.current = es

      es.onmessage = (event) => {
        try {
          const data: NeurolinkState = JSON.parse(event.data)
          setState(data)
        } catch {
          // Ignore parse errors
        }
      }

      es.onerror = () => {
        es.close()
        esRef.current = null
        if (!cancelled) {
          // Reconnect after 3 seconds
          setTimeout(connect, 3000)
        }
      }
    }

    connect()

    return () => {
      cancelled = true
      esRef.current?.close()
    }
  }, [url])

  return state
}
