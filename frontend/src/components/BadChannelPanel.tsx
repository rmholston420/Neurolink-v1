/**
 * BadChannelPanel
 *
 * Displays the current Stage 0/2 bad-channel state:
 *   • Per-electrode pill grid — green (good) or red (bad) for each channel name
 *   • If bad channels exist: amber banner listing them + interpolation note
 *   • Counts good vs bad at a glance
 *
 * Props:
 *   badChannels   string[]   channel names flagged bad this frame (from NeurolinkState)
 *   allChannels   string[]   ordered list of all channels the device reports
 *                            (defaults to Muse-S 4-channel layout if omitted)
 */
import React from 'react'

const MUSE_CHANNELS = ['TP9', 'AF7', 'AF8', 'TP10']

interface Props {
  badChannels: string[]
  allChannels?: string[]
}

const S: Record<string, React.CSSProperties> = {
  container: { display: 'flex', flexDirection: 'column', gap: 10 },
  grid: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: 6,
  },
  banner: {
    display: 'flex',
    alignItems: 'flex-start',
    gap: 8,
    padding: '8px 12px',
    borderRadius: 8,
    background: 'rgba(227,179,65,0.08)',
    border: '1px solid rgba(227,179,65,0.3)',
  },
  bannerIcon: { fontSize: 14, flexShrink: 0, marginTop: 1 },
  bannerText: { fontSize: 12, color: '#e3b341', lineHeight: 1.5 },
  bannerNote: { fontSize: 11, color: '#8b949e', marginTop: 3 },
  good: {
    display: 'inline-flex', alignItems: 'center', gap: 5,
    padding: '4px 10px', borderRadius: 20, fontSize: 12, fontWeight: 600,
    background: 'rgba(46,160,67,0.1)',
    border: '1px solid rgba(46,160,67,0.3)',
    color: '#3fb950',
  },
  bad: {
    display: 'inline-flex', alignItems: 'center', gap: 5,
    padding: '4px 10px', borderRadius: 20, fontSize: 12, fontWeight: 600,
    background: 'rgba(248,81,73,0.1)',
    border: '1px solid rgba(248,81,73,0.3)',
    color: '#f85149',
  },
  dot: (bad: boolean): React.CSSProperties => ({
    width: 6, height: 6, borderRadius: '50%',
    background: bad ? '#f85149' : '#3fb950',
    flexShrink: 0,
  }),
  summary: { fontSize: 12, color: '#8b949e' },
}

export default function BadChannelPanel({ badChannels, allChannels }: Props) {
  const channels = allChannels ?? MUSE_CHANNELS
  const badSet = new Set(badChannels.map(c => c.toUpperCase()))
  const goodCount = channels.filter(c => !badSet.has(c.toUpperCase())).length
  const badCount  = badChannels.length

  return (
    <div style={S.container}>
      {/* Per-channel pill grid */}
      <div style={S.grid}>
        {channels.map(ch => {
          const isBad = badSet.has(ch.toUpperCase())
          return (
            <span key={ch} style={isBad ? S.bad : S.good}>
              <span style={S.dot(isBad)} />
              {ch}
            </span>
          )
        })}
      </div>

      {/* Summary line */}
      <div style={S.summary}>
        {goodCount} / {channels.length} channels active
        {badCount > 0 && ` · ${badCount} flagged`}
      </div>

      {/* Bad channel banner */}
      {badCount > 0 && (
        <div style={S.banner}>
          <span style={S.bannerIcon}>⚠️</span>
          <div>
            <div style={S.bannerText}>
              Bad: {badChannels.join(', ')}
            </div>
            <div style={S.bannerNote}>
              Stage 2 spherical-spline interpolation applied this frame.
            </div>
          </div>
        </div>
      )}

      {/* All-good confirmation */}
      {badCount === 0 && (
        <div style={{
          fontSize: 12, color: '#3fb950',
          display: 'flex', alignItems: 'center', gap: 6,
        }}>
          <span style={{ fontSize: 13 }}>✓</span>
          All channels active — no interpolation needed
        </div>
      )}
    </div>
  )
}
