/**
 * useArtifactConfig
 *
 * Fetches and manages the live Stage 3 artifact-gate thresholds
 * from /api/v1/stage3/config.
 *
 * Returns:
 *   config        — current GateConfig (or null while loading)
 *   loading       — true on initial fetch
 *   error         — last fetch/update error string, or null
 *   updateConfig  — send a partial update; optimistic UI, server-confirmed
 *   resetConfig   — POST /stage3/reset to zero the frame counters
 *
 * Mirrors the shape and pattern of useFilters.ts.
 */
import { useState, useEffect, useCallback } from 'react'

export interface ArtifactGateConfig {
  pk2pk_uv:            number   // amplitude threshold µV
  accel_rms_g:         number   // IMU motion threshold g
  kurtosis_threshold:  number   // kurtosis burst threshold
  enable_amplitude:    boolean  // toggle amplitude gate pass
  enable_imu:          boolean  // toggle IMU gate pass
  enable_kurtosis:     boolean  // toggle kurtosis gate pass
}

export type PartialArtifactGateConfig = Partial<ArtifactGateConfig>

export function useArtifactConfig(apiUrl: string) {
  const [config, setConfig] = useState<ArtifactGateConfig | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError]   = useState<string | null>(null)

  // ── Fetch on mount ──────────────────────────────────────────────────
  useEffect(() => {
    let cancelled = false
    fetch(`${apiUrl}/api/v1/stage3/config`)
      .then(r => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json() as Promise<ArtifactGateConfig>
      })
      .then(data => { if (!cancelled) { setConfig(data); setLoading(false) } })
      .catch(e  => { if (!cancelled) { setError(String(e)); setLoading(false) } })
    return () => { cancelled = true }
  }, [apiUrl])

  // ── Optimistic partial update ───────────────────────────────────────
  const updateConfig = useCallback(
    (patch: PartialArtifactGateConfig) => {
      // Apply optimistically
      setConfig(prev => prev ? { ...prev, ...patch } : prev)

      fetch(`${apiUrl}/api/v1/stage3/config`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(patch),
      })
        .then(r => {
          if (!r.ok) throw new Error(`HTTP ${r.status}`)
          return r.json() as Promise<ArtifactGateConfig>
        })
        .then(data => setConfig(data))  // confirm with server state
        .catch(e  => setError(String(e)))
    },
    [apiUrl],
  )

  // ── Reset frame counters ────────────────────────────────────────────
  const resetStats = useCallback(() => {
    fetch(`${apiUrl}/api/v1/stage3/reset`, { method: 'POST' })
      .catch(e => setError(String(e)))
  }, [apiUrl])

  return { config, loading, error, updateConfig, resetStats }
}
