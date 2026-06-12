/**
 * ImpedancePanel — per-channel electrode impedance display.
 *
 * Outlier detection: a channel is flagged when its impedance lies more than
 * 1.5 × IQR above the session median (upper fence only — we never flag a
 * channel for being TOO well-connected).
 *
 * Clinical colour thresholds:
 *   good   < 20 kΩ   green
 *   warn  20–50 kΩ   amber
 *   bad    > 50 kΩ   red
 *   outlier: ⚠ badge + pulsing amber border
 */
import React, { useMemo, useRef } from 'react'

interface Props {
  /** Map of channel name → impedance in kΩ. */
  impedances: Record<string, number>
}

const GOOD_KO = 20
const BAD_KO  = 50
const MAX_BAR = 100

function impedanceColor(kOhm: number): string {
  if (kOhm < GOOD_KO) return '#3fb950'
  if (kOhm < BAD_KO)  return '#d29922'
  return '#f85149'
}

function median(vals: number[]): number {
  const s = [...vals].sort((a, b) => a - b)
  const mid = Math.floor(s.length / 2)
  return s.length % 2 === 1 ? s[mid] : (s[mid - 1] + s[mid]) / 2
}

function iqr(vals: number[]): number {
  const s = [...vals].sort((a, b) => a - b)
  const q1 = median(s.slice(0, Math.floor(s.length / 2)))
  const q3 = median(s.slice(Math.ceil(s.length / 2)))
  return q3 - q1
}

const KEYFRAMES = `
  @keyframes imp-pulse {
    0%, 100% { border-color: #d29922; }
    50%       { border-color: #f0883e; }
  }
`

const S: Record<string, React.CSSProperties> = {
  root: { width: '100%' },
  empty: {
    color: '#484f58', fontSize: 12, fontStyle: 'italic',
    textAlign: 'center', padding: '18px 0',
  },
  legend: { display: 'flex', gap: 14, marginBottom: 10, flexWrap: 'wrap' as const },
  legendItem: { display: 'flex', alignItems: 'center', gap: 5, fontSize: 11 },
  grid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fill, minmax(130px, 1fr))',
    gap: 8,
  },
  channelName: {
    fontSize: 11, fontWeight: 700, color: '#8b949e',
    textTransform: 'uppercase' as const, letterSpacing: 0.5, marginBottom: 5,
  },
  valueRow: { display: 'flex', alignItems: 'baseline', gap: 4, marginBottom: 6 },
  unit: { fontSize: 11, color: '#484f58' },
  barTrack: { height: 4, background: '#21262d', borderRadius: 2, overflow: 'hidden' },
  badge: { position: 'absolute', top: 6, right: 7, fontSize: 10, color: '#d29922' },
  medianRow: {
    marginTop: 12, paddingTop: 10, borderTop: '1px solid #21262d',
    fontSize: 11, color: '#8b949e',
    display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap' as const, gap: 6,
  },
  medianVal: { color: '#cdd9e5', fontVariantNumeric: 'tabular-nums' as const },
}

export default function ImpedancePanel({ impedances }: Props) {
  const injected = useRef(false)
  if (!injected.current) {
    const style = document.createElement('style')
    style.textContent = KEYFRAMES
    document.head.appendChild(style)
    injected.current = true
  }

  const entries = useMemo(
    () => Object.entries(impedances).sort(([a], [b]) => a.localeCompare(b)),
    [impedances],
  )

  const values = entries.map(([, v]) => v)

  const { fence, med } = useMemo(() => {
    if (values.length < 2) return { fence: Infinity, med: values[0] ?? 0 }
    const m = median(values)
    return { fence: m + 1.5 * iqr(values), med: m }
  }, [values])

  if (entries.length === 0) {
    return (
      <div style={S.root}>
        <p style={S.empty}>
          No impedance data — adapter does not expose per-channel impedance.
        </p>
      </div>
    )
  }

  return (
    <div style={S.root}>
      {/* Legend */}
      <div style={S.legend}>
        {([
          { color: '#3fb950', label: `Good < ${GOOD_KO} kΩ` },
          { color: '#d29922', label: `Warn ${GOOD_KO}–${BAD_KO} kΩ` },
          { color: '#f85149', label: `Poor > ${BAD_KO} kΩ` },
        ] as const).map(({ color, label }) => (
          <span key={label} style={S.legendItem}>
            <span style={{ width: 8, height: 8, borderRadius: '50%', background: color, display: 'inline-block', flexShrink: 0 }} />
            <span style={{ color: '#8b949e' }}>{label}</span>
          </span>
        ))}
        <span style={S.legendItem}>
          <span style={{ color: '#d29922', fontSize: 11 }}>⚠ Outlier (&gt; median + 1.5×IQR)</span>
        </span>
      </div>

      {/* Per-channel tiles */}
      <div style={S.grid}>
        {entries.map(([ch, kOhm]) => {
          const color   = impedanceColor(kOhm)
          const outlier = kOhm > fence
          const pct     = Math.min((kOhm / MAX_BAR) * 100, 100)
          return (
            <div
              key={ch}
              style={{
                background: '#0d1117',
                border: `1px solid ${outlier ? '#d29922' : '#21262d'}`,
                borderRadius: 8,
                padding: '8px 10px',
                position: 'relative',
                animation: outlier ? 'imp-pulse 1.8s ease-in-out infinite' : 'none',
              }}
            >
              {outlier && <span style={S.badge}>⚠</span>}
              <div style={S.channelName}>{ch}</div>
              <div style={S.valueRow}>
                <span style={{ fontSize: 18, fontWeight: 700, fontVariantNumeric: 'tabular-nums', color }}>
                  {kOhm.toFixed(1)}
                </span>
                <span style={S.unit}>kΩ</span>
              </div>
              <div style={S.barTrack}>
                <div style={{
                  height: '100%', width: `${pct}%`, background: color,
                  borderRadius: 2, transition: 'width 0.35s ease',
                }} />
              </div>
            </div>
          )
        })}
      </div>

      {/* Session summary */}
      <div style={S.medianRow}>
        <span>Session median</span>
        <span style={S.medianVal}>{med.toFixed(1)} kΩ</span>
        <span>Outlier fence</span>
        <span style={S.medianVal}>
          {fence === Infinity ? '—' : `${fence.toFixed(1)} kΩ`}
        </span>
      </div>
    </div>
  )
}
