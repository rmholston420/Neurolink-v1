/**
 * IMUPanel
 *
 * Shows pitch, roll and motion RMS from the headset IMU.
 * Upgraded for Stage 3: highlights motion_rms in amber/red when it
 * approaches or exceeds the artifact gate threshold (~0.12 g default).
 */
import React from 'react'

interface Props {
  pitchDeg:  number | null
  rollDeg:   number | null
  motionRms: number | null
  /** Stage 3 artifact gate motion threshold in g (default 0.12) */
  motionGateThreshold?: number
}

// Fraction of threshold at which we start showing a warning colour
const WARN_FRACTION = 0.7   // amber at 70% of threshold
const GATE_DEFAULT  = 0.12  // g — mirrors artifact_gate.py default

const styles: Record<string, React.CSSProperties> = {
  container: { display: 'flex', flexDirection: 'column', gap: 10 },
  metric: { display: 'flex', justifyContent: 'space-between', alignItems: 'center' },
  metricLabel: { fontSize: 13, color: '#8b949e' },
  metricValue: { fontSize: 22, fontWeight: 700, color: '#e6edf3' },
  metricUnit: { fontSize: 12, color: '#484f58', marginLeft: 2 },
  divider: { height: 1, background: '#21262d' },
}

function DegreeRow({ label, value }: { label: string; value: number | null }) {
  return (
    <div style={styles.metric}>
      <span style={styles.metricLabel}>{label}</span>
      <span>
        {value !== null ? (
          <>
            <span style={styles.metricValue}>{value.toFixed(1)}</span>
            <span style={styles.metricUnit}>{'°'}</span>
          </>
        ) : (
          <span style={styles.metricValue}>{'\u2014'}</span>
        )}
      </span>
    </div>
  )
}

function MotionRow({
  value,
  threshold,
}: {
  value: number | null
  threshold: number
}) {
  let valueColour = '#e6edf3'
  let gateLabel: string | null = null

  if (value !== null) {
    if (value >= threshold) {
      valueColour = '#f85149'
      gateLabel   = 'GATE TRIGGERED'
    } else if (value >= threshold * WARN_FRACTION) {
      valueColour = '#e3b341'
      gateLabel   = 'Near threshold'
    }
  }

  return (
    <div style={{ ...styles.metric, flexWrap: 'wrap', gap: 4 }}>
      <span style={styles.metricLabel}>Motion RMS</span>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
        <span>
          <span style={{ ...styles.metricValue, color: valueColour }}>
            {value !== null ? value.toFixed(3) : '\u2014'}
          </span>
          {value !== null && <span style={styles.metricUnit}>g</span>}
        </span>
        {gateLabel && (
          <span style={{
            fontSize: 10, fontWeight: 700,
            color: value !== null && value >= threshold ? '#f85149' : '#e3b341',
            letterSpacing: 0.5,
            textTransform: 'uppercase',
          }}>
            {gateLabel}
          </span>
        )}
      </div>
      {/* Mini threshold bar */}
      {value !== null && (
        <div style={{ width: '100%', marginTop: 4 }}>
          <div style={{ height: 4, background: '#21262d', borderRadius: 2, overflow: 'hidden' }}>
            <div style={{
              height: '100%',
              width: `${Math.min((value / threshold) * 100, 100)}%`,
              background: valueColour,
              borderRadius: 2,
              transition: 'width 0.2s ease, background 0.3s ease',
            }} />
          </div>
          <div style={{
            display: 'flex', justifyContent: 'flex-end',
            fontSize: 10, color: '#484f58', marginTop: 2,
          }}>
            gate: {threshold} g
          </div>
        </div>
      )}
    </div>
  )
}

export default function IMUPanel({
  pitchDeg, rollDeg, motionRms,
  motionGateThreshold = GATE_DEFAULT,
}: Props) {
  return (
    <div style={styles.container}>
      <DegreeRow label="Pitch" value={pitchDeg} />
      <div style={styles.divider} />
      <DegreeRow label="Roll"  value={rollDeg} />
      <div style={styles.divider} />
      <MotionRow value={motionRms} threshold={motionGateThreshold} />
    </div>
  )
}
