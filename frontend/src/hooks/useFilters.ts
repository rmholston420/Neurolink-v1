import { useState, useEffect, useCallback } from 'react'

export interface FilterToggles {
  stage1_fir: boolean
  stage2_bad_channels: boolean
  stage3_artifact_gate: boolean
  stage4_asr: boolean
  stage4b_baseline: boolean
  stage5_ocular: boolean
  imu_gate: boolean
}

const DEFAULT_TOGGLES: FilterToggles = {
  stage1_fir: true,
  stage2_bad_channels: true,
  stage3_artifact_gate: true,
  stage4_asr: true,
  stage4b_baseline: true,
  stage5_ocular: true,
  imu_gate: true,
}

export function useFilters(apiUrl: string) {
  const [toggles, setToggles] = useState<FilterToggles>(DEFAULT_TOGGLES)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Fetch current state on mount
  useEffect(() => {
    let cancelled = false
    fetch(`${apiUrl}/api/v1/filters`)
      .then(r => r.json())
      .then(data => { if (!cancelled) { setToggles(data); setLoading(false) } })
      .catch(e => { if (!cancelled) { setError(String(e)); setLoading(false) } })
    return () => { cancelled = true }
  }, [apiUrl])

  // Optimistic toggle: update UI immediately, then confirm with server
  const toggle = useCallback((key: keyof FilterToggles) => {
    setToggles(prev => {
      const next = { ...prev, [key]: !prev[key] }
      fetch(`${apiUrl}/api/v1/filters`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ [key]: next[key] }),
      })
        .then(r => r.json())
        .then(data => setToggles(data))
        .catch(e => setError(String(e)))
      return next
    })
  }, [apiUrl])

  const resetAll = useCallback(() => {
    fetch(`${apiUrl}/api/v1/filters`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(DEFAULT_TOGGLES),
    })
      .then(r => r.json())
      .then(data => setToggles(data))
      .catch(e => setError(String(e)))
  }, [apiUrl])

  return { toggles, loading, error, toggle, resetAll }
}
