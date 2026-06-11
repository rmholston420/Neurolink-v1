import React from 'react'
import type { EA1Result } from '../hooks/useNeurolinkSSE'

interface Props {
  ea1: EA1Result | null
}

const styles: Record<string, React.CSSProperties> = {
  container: { display: 'flex', flexDirection: 'column', gap: 12 },
  scoreRow: { display: 'flex', alignItems: 'baseline', gap: 8 },
  scoreNum: { fontSize: 42, fontWeight: 700, lineHeight: 1 },
  scoreLabel: { fontSize: 14, color: '#8b949e' },
  criteria: { fontSize: 13, color: '#8b949e' },
  overlay: { fontSize: 12, color: '#484f58' },
}

const badgeStyle = (eligible: boolean): React.CSSProperties => ({
  display: 'inline-block',
  padding: '4px 14px',
  borderRadius: 20,
  fontWeight: 600,
  fontSize: 14,
  background: eligible ? 'rgba(46,160,67,0.15)' : 'rgba(248,81,73,0.15)',
  color: eligible ? '#3fb950' : '#f85149',
  border: `1px solid ${eligible ? '#238636' : '#da3633'}`,
})

export default function EA1Score({ ea1 }: Props) {
  if (!ea1) {
    return <div style={{ color: '#484f58', fontSize: 14 }}>No data</div>
  }

  const pct = (ea1.score * 100).toFixed(0)
  const color = ea1.eligible ? '#3fb950' : '#f85149'

  return (
    <div style={styles.container}>
      <div style={styles.scoreRow}>
        <span style={{ ...styles.scoreNum, color }}>{pct}%</span>
        <span style={styles.scoreLabel}>EA-1 score</span>
      </div>
      <span style={badgeStyle(ea1.eligible)}>{ea1.label}</span>
      <div style={styles.criteria}>
        {ea1.criteria_met} / {ea1.criteria_total} criteria met
      </div>
      <div style={styles.overlay}>Mode: {ea1.overlay_mode}</div>
    </div>
  )
}
