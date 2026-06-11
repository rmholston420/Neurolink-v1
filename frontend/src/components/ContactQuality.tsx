import React from 'react'

interface Props {
  poorContact: boolean
  contactQuality: number | null
}

const styles: Record<string, React.CSSProperties> = {
  container: { display: 'flex', flexDirection: 'column', gap: 12 },
  barBg: {
    height: 14,
    background: '#21262d',
    borderRadius: 7,
    overflow: 'hidden',
  },
  label: { fontSize: 12, color: '#8b949e' },
}

const indicatorStyle = (poor: boolean): React.CSSProperties => ({
  display: 'inline-flex',
  alignItems: 'center',
  gap: 8,
  padding: '8px 16px',
  borderRadius: 10,
  background: poor ? 'rgba(248,81,73,0.1)' : 'rgba(46,160,67,0.1)',
  border: `1px solid ${poor ? '#da3633' : '#238636'}`,
  color: poor ? '#f85149' : '#3fb950',
  fontWeight: 600,
  fontSize: 15,
})

const dotStyle = (poor: boolean): React.CSSProperties => ({
  width: 10,
  height: 10,
  borderRadius: '50%',
  background: poor ? '#f85149' : '#3fb950',
})

export default function ContactQuality({ poorContact, contactQuality }: Props) {
  const pct = contactQuality !== null ? (contactQuality * 100).toFixed(0) : null
  const color = poorContact ? '#f85149' : '#3fb950'

  return (
    <div style={styles.container}>
      <span style={indicatorStyle(poorContact)}>
        <span style={dotStyle(poorContact)} />
        {poorContact ? 'Poor Contact' : 'Good Contact'}
      </span>
      {contactQuality !== null && (
        <div>
          <div style={styles.label}>Quality: {pct}%</div>
          <div style={styles.barBg}>
            <div
              style={{
                height: '100%',
                width: `${pct}%`,
                background: color,
                borderRadius: 7,
                transition: 'width 0.25s ease',
              }}
            />
          </div>
        </div>
      )}
    </div>
  )
}
