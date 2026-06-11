import React from 'react'

interface Props {
  pitchDeg: number | null
  rollDeg: number | null
  motionRms: number | null
}

const styles: Record<string, React.CSSProperties> = {
  container: { display: 'flex', flexDirection: 'column', gap: 10 },
  metric: { display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' },
  metricLabel: { fontSize: 13, color: '#8b949e' },
  metricValue: { fontSize: 22, fontWeight: 700, color: '#e6edf3' },
  metricUnit: { fontSize: 12, color: '#484f58', marginLeft: 4 },
  divider: { height: 1, background: '#21262d' },
  motionBar: {
    height: 8,
    background: '#21262d',
    borderRadius: 4,
    overflow: 'hidden',
    marginTop: 4,
  },
}

function MetricRow({
  label,
  value,
  unit,
}: {
  label: string
  value: number | null
  unit: string
}) {
  return (
    <div style={styles.metric}>
      <span style={styles.metricLabel}>{label}</span>
      <span>
        <span style={styles.metricValue}>
          {value !== null ? value.toFixed(1) : '—'}
        </span>
        <span style={styles.metricUnit}>{unit}</span>
      </span>
    </div>
  )
}

export default function IMUPanel({ pitchDeg, rollDeg, motionRms }: Props) {
  const motionPct = motionRms !== null ? Math.min(motionRms * 100, 100) : 0
  const motionColor = motionPct > 60 ? '#f85149' : motionPct > 30 ? '#d29922' : '#3fb950'

  return (
    <div style={styles.container}>
      <MetricRow label="Pitch" value={pitchDeg} unit="°" />
      <div style={styles.divider} />
      <MetricRow label="Roll" value={rollDeg} unit="°" />
      <div style={styles.divider} />
      <div>
        <div style={styles.metric}>
          <span style={styles.metricLabel}>Motion RMS</span>
          <span>
            <span style={{ ...styles.metricValue, fontSize: 16 }}>
              {motionRms !== null ? motionRms.toFixed(3) : '—'}
            </span>
          </span>
        </div>
        <div style={styles.motionBar}>
          <div
            style={{
              height: '100%',
              width: `${motionPct}%`,
              background: motionColor,
              borderRadius: 4,
              transition: 'width 0.25s ease',
            }}
          />
        </div>
      </div>
    </div>
  )
}
