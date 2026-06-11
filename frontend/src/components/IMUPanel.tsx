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
}

function MetricRow({
  label,
  value,
  unit,
  decimals = 1,
}: {
  label: string
  value: number | null
  unit: string
  decimals?: number
}) {
  return (
    <div style={styles.metric}>
      <span style={styles.metricLabel}>{label}</span>
      <span>
        <span style={styles.metricValue}>
          {value !== null ? value.toFixed(decimals) : '\u2014'}
        </span>
        {value !== null && <span style={styles.metricUnit}>{unit}</span>}
      </span>
    </div>
  )
}

export default function IMUPanel({ pitchDeg, rollDeg, motionRms }: Props) {
  return (
    <div style={styles.container}>
      <MetricRow label="Pitch" value={pitchDeg} unit="\u00b0" />
      <div style={styles.divider} />
      <MetricRow label="Roll" value={rollDeg} unit="\u00b0" />
      <div style={styles.divider} />
      <MetricRow label="Motion RMS" value={motionRms} unit="g" decimals={3} />
    </div>
  )
}
