import React from 'react'

interface Props {
  rrBpm: number | null
  rrPpg?: number | null
  rrAccel?: number | null
}

const styles: Record<string, React.CSSProperties> = {
  container: { display: 'flex', flexDirection: 'column', gap: 10 },
  metric: { display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' },
  metricLabel: { fontSize: 13, color: '#8b949e' },
  metricValue: { fontSize: 22, fontWeight: 700, color: '#e6edf3' },
  metricUnit: { fontSize: 12, color: '#484f58', marginLeft: 4 },
  divider: { height: 1, background: '#21262d' },
}

function MetricRow({
  label,
  value,
  unit,
}: {
  label: string
  value: number | null | undefined
  unit: string
}) {
  return (
    <div style={styles.metric}>
      <span style={styles.metricLabel}>{label}</span>
      <span>
        <span style={styles.metricValue}>
          {value != null ? value.toFixed(1) : '—'}
        </span>
        <span style={styles.metricUnit}>{unit}</span>
      </span>
    </div>
  )
}

export default function BreathingPanel({ rrBpm, rrPpg, rrAccel }: Props) {
  return (
    <div style={styles.container}>
      <MetricRow label="Fused Rate" value={rrBpm} unit="bpm" />
      <div style={styles.divider} />
      <MetricRow label="PPG-derived" value={rrPpg} unit="bpm" />
      <div style={styles.divider} />
      <MetricRow label="Accel-derived" value={rrAccel} unit="bpm" />
    </div>
  )
}
