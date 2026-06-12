/**
 * BaselinePanel
 *
 * Displays personal baseline calibration status and live z-score deviations.
 *
 * States:
 *   Uncalibrated  → shows how many sessions remain before baseline is ready
 *   Calibrated    → shows baseline mean ± sigma and current z-score for
 *                   alpha, theta, and focus with colour-coded delta arrows
 */
import React from 'react'
import type { PersonalBaselineReturn } from '../hooks/usePersonalBaseline'

interface Props {
  bl: PersonalBaselineReturn
}

function zColour(z: number | null): string {
  if (z === null) return '#484f58'
  if (z > 1.5)  return '#3fb950'
  if (z > 0.5)  return '#6fdd8b'
  if (z > -0.5) return '#e6edf3'
  if (z > -1.5) return '#e3b341'
  return '#f85149'
}

function zLabel(z: number | null): string {
  if (z === null) return '—'
  if (z > 1.5)  return '▲▲'
  if (z > 0.5)  return '▲'
  if (z > -0.5) return '◆'
  if (z > -1.5) return '▼'
  return '▼▼'
}

function zText(z: number | null): string {
  if (z === null) return 'No data'
  const sign = z >= 0 ? '+' : ''
  return `${sign}${z.toFixed(2)} σ`
}

const st: Record<string, React.CSSProperties> = {
  root:   { display: 'flex', flexDirection: 'column', gap: 14 },
  uncal:  {
    padding: '12px 14px', borderRadius: 8,
    background: 'rgba(225,162,66,0.08)', border: '1px solid rgba(225,162,66,0.25)',
    fontSize: 12, color: '#e3b341', lineHeight: 1.5,
  },
  metricGrid: {
    display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 10,
  },
  metricBox: {
    display: 'flex', flexDirection: 'column', gap: 3,
    padding: '10px 12px',
    background: 'rgba(22,27,34,0.6)',
    border: '1px solid #21262d', borderRadius: 8,
  },
  metricLabel: { fontSize: 10, fontWeight: 700, color: '#484f58', textTransform: 'uppercase', letterSpacing: 0.8 },
  metricZ:     (z: number | null): React.CSSProperties => ({
    fontSize: 22, fontWeight: 700, color: zColour(z),
    lineHeight: 1.1, fontVariantNumeric: 'tabular-nums',
  }),
  metricArrow: (z: number | null): React.CSSProperties => ({
    fontSize: 13, color: zColour(z),
  }),
  metricBase: {
    fontSize: 10, color: '#484f58', marginTop: 2,
  },
  footer: { display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 8 },
  nLabel: { fontSize: 11, color: '#484f58' },
  resetBtn: {
    fontSize: 11, color: '#f85149', cursor: 'pointer', background: 'none',
    border: '1px solid rgba(248,81,73,0.3)', borderRadius: 5,
    padding: '3px 8px', transition: 'all 150ms',
  },
}

const METRICS: { key: 'alpha' | 'theta' | 'focus'; label: string }[] = [
  { key: 'alpha', label: 'Alpha' },
  { key: 'theta', label: 'Theta' },
  { key: 'focus', label: 'Focus' },
]

export default function BaselinePanel({ bl }: Props) {
  if (!bl.isCalibrated) {
    const needed = Math.max(0, 3 - bl.nSessions)
    return (
      <div style={st.root}>
        <div style={st.uncal}>
          <strong>Baseline not yet calibrated.</strong><br />
          Complete <strong>{needed}</strong> more session{needed !== 1 ? 's' : ''} to establish your personal baseline.
          {bl.nSessions > 0 && ` (${bl.nSessions} recorded so far)`}
        </div>
      </div>
    )
  }

  return (
    <div style={st.root}>
      <div style={st.metricGrid}>
        {METRICS.map(({ key, label }) => {
          const z  = bl.deviation[key]
          const bm = bl.baseline![key]
          return (
            <div key={key} style={st.metricBox}>
              <span style={st.metricLabel}>{label}</span>
              <span style={st.metricZ(z)}>{zText(z)}</span>
              <span style={st.metricArrow(z)}>{zLabel(z)}</span>
              <span style={st.metricBase}>
                baseline {bm.mean.toFixed(key === 'focus' ? 2 : 4)} ± {bm.sigma.toFixed(key === 'focus' ? 2 : 4)}
              </span>
            </div>
          )
        })}
      </div>

      <div style={st.footer}>
        <span style={st.nLabel}>{bl.nSessions} sessions in baseline (max 20)</span>
        <button style={st.resetBtn} onClick={bl.resetBaseline}>↺ Reset baseline</button>
      </div>
    </div>
  )
}
