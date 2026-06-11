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

/**
 * Degree rows (Pitch, Roll): value and ° are combined into a single text
 * node so that getAllByText('°') can locate the symbol directly.
 */
function DegreeRow({ label, value }: { label: string; value: number | null }) {
  return (
    <div style={styles.metric}>
      <span style={styles.metricLabel}>{label}</span>
      <span style={styles.metricValue}>
        {value !== null ? `${value.toFixed(1)}\u00b0` : '\u2014'}
      </span>
    </div>
  )
}

/**
 * Generic metric row — value and unit in separate spans.
 * Used for Motion RMS where the unit is not degree-symbol.
 */
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
      <DegreeRow label="Pitch" value={pitchDeg} />
      <div style={styles.divider} />
      <DegreeRow label="Roll" value={rollDeg} />
      <div style={styles.divider} />
      <MetricRow label="Motion RMS" value={motionRms} unit="g" decimals={3} />
    </div>
  )
}
