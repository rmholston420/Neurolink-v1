/**
 * ContactQuality
 *
 * Upgraded for Stage 0/2: also renders per-channel bad_channel list
 * when provided, so the card gives a complete picture of electrode
 * contact at the channel level.
 */
import React from 'react'

interface Props {
  poorContact:    boolean
  contactQuality: number | null
  /** Stage 2 bad channel names — optional; omit to keep original behaviour */
  badChannels?:   string[]
}

const MUSE_CH = ['TP9', 'AF7', 'AF8', 'TP10']

const styles: Record<string, React.CSSProperties> = {
  container:  { display: 'flex', flexDirection: 'column', gap: 12 },
  barBg:      { height: 14, background: '#21262d', borderRadius: 7, overflow: 'hidden' },
  label:      { fontSize: 12, color: '#8b949e' },
  grid:       { display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 2 },
  divider:    { height: 1, background: '#21262d' },
}

const indicatorStyle = (poor: boolean): React.CSSProperties => ({
  display: 'inline-flex', alignItems: 'center', gap: 8,
  padding: '8px 16px', borderRadius: 10,
  background: poor ? 'rgba(248,81,73,0.1)' : 'rgba(46,160,67,0.1)',
  border: `1px solid ${poor ? '#da3633' : '#238636'}`,
  color: poor ? '#f85149' : '#3fb950',
  fontWeight: 600, fontSize: 15,
})

const dotStyle = (poor: boolean): React.CSSProperties => ({
  width: 10, height: 10, borderRadius: '50%',
  background: poor ? '#f85149' : '#3fb950',
})

export default function ContactQuality({ poorContact, contactQuality, badChannels }: Props) {
  const pct   = contactQuality !== null ? (contactQuality * 100).toFixed(0) : null
  const color = poorContact ? '#f85149' : '#3fb950'
  const badSet = new Set((badChannels ?? []).map(c => c.toUpperCase()))

  return (
    <div style={styles.container}>
      {/* Overall indicator */}
      <span style={indicatorStyle(poorContact)}>
        <span style={dotStyle(poorContact)} />
        {poorContact ? 'Poor Contact' : 'Good Contact'}
      </span>

      {/* Quality bar */}
      {contactQuality !== null && (
        <div>
          <div style={styles.label}>Quality: {pct}%</div>
          <div style={styles.barBg}>
            <div style={{
              height: '100%', width: `${pct}%`,
              background: color, borderRadius: 7,
              transition: 'width 0.25s ease',
            }} />
          </div>
        </div>
      )}

      {/* Per-channel pills (Stage 2) */}
      {badChannels !== undefined && (
        <>
          <div style={styles.divider} />
          <div style={styles.label}>Channels</div>
          <div style={styles.grid}>
            {MUSE_CH.map(ch => {
              const bad = badSet.has(ch)
              return (
                <span key={ch} style={{
                  display: 'inline-flex', alignItems: 'center', gap: 5,
                  padding: '3px 10px', borderRadius: 20, fontSize: 12, fontWeight: 600,
                  background: bad ? 'rgba(248,81,73,0.1)' : 'rgba(46,160,67,0.08)',
                  border: `1px solid ${bad ? 'rgba(248,81,73,0.3)' : 'rgba(46,160,67,0.25)'}`,
                  color: bad ? '#f85149' : '#3fb950',
                }}>
                  <span style={{
                    width: 5, height: 5, borderRadius: '50%',
                    background: bad ? '#f85149' : '#3fb950',
                  }} />
                  {ch}
                  {bad && <span style={{ fontSize: 10 }}>↯</span>}
                </span>
              )
            })}
          </div>
          {badSet.size > 0 && (
            <div style={{ fontSize: 11, color: '#e3b341', marginTop: -4 }}>
              ↻ Spherical-spline interpolation active for flagged channels
            </div>
          )}
        </>
      )}
    </div>
  )
}
