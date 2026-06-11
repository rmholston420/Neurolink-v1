import React from 'react'

interface Props {
  focusState: string
  focusScore: number
  fatigueScore: number
}

const FOCUS_COLORS: Record<string, string> = {
  HIGH_FOCUS: '#3fb950',
  MODERATE_FOCUS: '#d29922',
  LOW_FOCUS: '#f85149',
  DISTRACTED: '#da3633',
  unknown: '#484f58',
}

const styles: Record<string, React.CSSProperties> = {
  container: { display: 'flex', flexDirection: 'column', gap: 14 },
  label: { fontSize: 13, color: '#8b949e', marginBottom: 4 },
  stateBadge: (color: string): React.CSSProperties => ({
    display: 'inline-block',
    padding: '4px 14px',
    borderRadius: 20,
    fontWeight: 600,
    fontSize: 14,
    background: `${color}22`,
    color,
    border: `1px solid ${color}66`,
    marginBottom: 6,
  }),
  barBg: {
    height: 12,
    background: '#21262d',
    borderRadius: 6,
    overflow: 'hidden',
  },
}

export default function FocusFatigueGauge({ focusState, focusScore, fatigueScore }: Props) {
  const focusColor = FOCUS_COLORS[focusState] ?? '#484f58'
  const fatigueColor = fatigueScore > 0.7 ? '#f85149' : fatigueScore > 0.4 ? '#d29922' : '#3fb950'

  return (
    <div style={styles.container}>
      <div>
        <div style={styles.label}>Focus State</div>
        <span style={styles.stateBadge(focusColor)}>{focusState.replace('_', ' ')}</span>
        <div style={styles.barBg}>
          <div
            style={{
              height: '100%',
              width: `${(focusScore * 100).toFixed(1)}%`,
              background: focusColor,
              borderRadius: 6,
              transition: 'width 0.25s ease',
            }}
          />
        </div>
      </div>
      <div>
        <div style={styles.label}>Fatigue Score: {(fatigueScore * 100).toFixed(0)}%</div>
        <div style={styles.barBg}>
          <div
            style={{
              height: '100%',
              width: `${(fatigueScore * 100).toFixed(1)}%`,
              background: fatigueColor,
              borderRadius: 6,
              transition: 'width 0.25s ease',
            }}
          />
        </div>
      </div>
    </div>
  )
}
