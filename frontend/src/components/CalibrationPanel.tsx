import React, { useState } from 'react'

interface Props {
  apiUrl: string
}

type CalibStatus = 'idle' | 'running' | 'complete' | 'error'

const styles: Record<string, React.CSSProperties> = {
  container: { display: 'flex', flexDirection: 'column', gap: 14 },
  description: { fontSize: 13, color: '#8b949e', lineHeight: 1.5 },
  button: {
    padding: '8px 18px',
    borderRadius: 8,
    border: '1px solid #388bfd',
    background: 'rgba(56,139,253,0.12)',
    color: '#58a6ff',
    fontWeight: 600,
    fontSize: 14,
    cursor: 'pointer',
    alignSelf: 'flex-start',
    transition: 'background 0.15s ease',
  },
  buttonDisabled: {
    opacity: 0.5,
    cursor: 'not-allowed',
  },
  status: { fontSize: 13, fontWeight: 600 },
  baseline: { fontSize: 13, color: '#8b949e' },
}

const STATUS_COLOR: Record<CalibStatus, string> = {
  idle: '#8b949e',
  running: '#d29922',
  complete: '#3fb950',
  error: '#f85149',
}

const STATUS_LABEL: Record<CalibStatus, string> = {
  idle: 'Not calibrated',
  running: 'Calibrating… (30 s)',
  complete: 'Calibration complete',
  error: 'Calibration failed',
}

export default function CalibrationPanel({ apiUrl }: Props) {
  const [status, setStatus] = useState<CalibStatus>('idle')
  const [baselineAlpha, setBaselineAlpha] = useState<number | null>(null)
  const busy = status === 'running'

  async function handleCalibrate() {
    setStatus('running')
    try {
      const resp = await fetch(`${apiUrl}/api/v1/neurolink/calibrate`, { method: 'POST' })
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
      const data = await resp.json()
      setStatus('complete')
      if (data.baseline_alpha != null) setBaselineAlpha(data.baseline_alpha)
    } catch {
      setStatus('error')
    }
  }

  return (
    <div style={styles.container}>
      <p style={styles.description}>
        Captures a 30-second personal alpha baseline. Keep eyes closed and
        remain still during calibration.
      </p>
      <button
        style={{ ...styles.button, ...(busy ? styles.buttonDisabled : {}) }}
        onClick={handleCalibrate}
        disabled={busy}
        aria-disabled={busy}
      >
        {busy ? 'Calibrating…' : 'Start Calibration'}
      </button>
      <div style={{ ...styles.status, color: STATUS_COLOR[status] }}>
        {STATUS_LABEL[status]}
      </div>
      {baselineAlpha !== null && (
        <div style={styles.baseline}>
          Baseline α: {baselineAlpha.toFixed(4)}
        </div>
      )}
    </div>
  )
}
