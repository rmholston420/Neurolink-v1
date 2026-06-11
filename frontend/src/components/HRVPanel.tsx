import React from 'react'

interface Props {
  hrBpm: number | null
  hrv: number | null
  rrBpm: number | null
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
  value: number | null
  unit: string
}) {
  return (
    <div style={styles.metric}>
      <span style={styles.metricLabel}>{label}</span>
      <span>
        <span style={styles.metricValue}>{value !== null ? value.toFixed(1) : '—'}</span>
        <span style={styles.metricUnit}>{unit}</span>
      </span>
    </div>
  )
}

export default function HRVPanel({ hrBpm, hrv, rrBpm }: Props) {
  return (
    <div style={styles.container}>
      <MetricRow label="Heart Rate" value={hrBpm} unit="bpm" />
      <div style={styles.divider} />
      <MetricRow label="HRV RMSSD" value={hrv} unit="ms" />
      <div style={styles.divider} />
      <MetricRow label="Breathing Rate" value={rrBpm} unit="bpm" />
    </div>
  )
}
