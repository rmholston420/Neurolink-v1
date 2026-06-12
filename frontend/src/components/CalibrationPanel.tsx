import React, { useState, useEffect, useRef } from 'react'

interface Props {
  apiUrl: string
}

type CalibStatus = 'idle' | 'running' | 'complete' | 'error'

const TOTAL_SEC = 90
const WARMUP_SEC = 30

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
  phase: { fontSize: 12, color: '#8b949e', marginTop: 2 },
  progressTrack: {
    height: 6,
    borderRadius: 3,
    background: '#21262d',
    overflow: 'hidden',
  },
  baseline: { fontSize: 13, color: '#8b949e' },
}

const STATUS_COLOR: Record<CalibStatus, string> = {
  idle: '#8b949e',
  running: '#d29922',
  complete: '#3fb950',
  error: '#f85149',
}

function formatCountdown(sec: number): string {
  const s = Math.max(0, Math.ceil(sec))
  return `${s}s`
}

export default function CalibrationPanel({ apiUrl }: Props) {
  const [status, setStatus] = useState<CalibStatus>('idle')
  const [baselineAlpha, setBaselineAlpha] = useState<number | null>(null)
  const [elapsed, setElapsed] = useState(0)
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const startTimeRef = useRef<number>(0)

  const busy = status === 'running'
  const remaining = Math.max(0, TOTAL_SEC - elapsed)
  const progress = Math.min(1, elapsed / TOTAL_SEC)
  const inWarmup = elapsed < WARMUP_SEC
  const warmupRemaining = Math.max(0, WARMUP_SEC - elapsed)
  const baselineRemaining = Math.max(0, TOTAL_SEC - elapsed)

  function startCountdown() {
    startTimeRef.current = performance.now()
    intervalRef.current = setInterval(() => {
      const secs = (performance.now() - startTimeRef.current) / 1000
      setElapsed(secs)
    }, 250)
  }

  function stopCountdown() {
    if (intervalRef.current) {
      clearInterval(intervalRef.current)
      intervalRef.current = null
    }
  }

  useEffect(() => () => stopCountdown(), [])

  async function handleCalibrate() {
    setStatus('running')
    setElapsed(0)
    startCountdown()
    try {
      const resp = await fetch(`${apiUrl}/api/v1/neurolink/calibrate`, { method: 'POST' })
      stopCountdown()
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
      const data = await resp.json()
      setStatus('complete')
      setElapsed(TOTAL_SEC)
      if (data.baseline_alpha != null) setBaselineAlpha(data.baseline_alpha)
    } catch {
      stopCountdown()
      setStatus('error')
    }
  }

  const phaseLabel = inWarmup
    ? `Warming up — discard window (${formatCountdown(warmupRemaining)} remaining)`
    : `Capturing baseline (${formatCountdown(baselineRemaining)} remaining)`

  const statusLabel = (() => {
    switch (status) {
      case 'idle':     return 'Not calibrated'
      case 'running':  return inWarmup
                         ? `Warming up… ${formatCountdown(warmupRemaining)}`
                         : `Capturing baseline… ${formatCountdown(baselineRemaining)}`
      case 'complete': return 'Calibration complete'
      case 'error':    return 'Calibration failed'
    }
  })()

  return (
    <div style={styles.container}>
      <p style={styles.description}>
        Captures a 90-second personal alpha baseline. Keep eyes closed and
        remain still. The first 30 seconds are a warmup discard window;
        baseline is computed from seconds 30–90.
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
        {statusLabel}
      </div>

      {busy && (
        <>
          <div style={styles.phase}>{phaseLabel}</div>
          <div style={styles.progressTrack}>
            <div
              style={{
                height: '100%',
                width: `${(progress * 100).toFixed(1)}%`,
                background: inWarmup ? '#d29922' : '#388bfd',
                borderRadius: 3,
                transition: 'width 0.25s linear, background 0.4s ease',
              }}
            />
          </div>
          <div style={{ fontSize: 11, color: '#484f58', textAlign: 'right' }}>
            {Math.round(elapsed)}s / {TOTAL_SEC}s
          </div>
        </>
      )}

      {baselineAlpha !== null && (
        <div style={styles.baseline}>
          Baseline α: {baselineAlpha.toFixed(4)}
        </div>
      )}
    </div>
  )
}
