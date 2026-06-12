/**
 * ArtifactStatsPanel
 *
 * Displays live rolling Stage 3 artifact rejection statistics:
 *   • Rejection rate gauge bar (% of last N frames)
 *   • Per-cause count pills (Amplitude spike / Head movement / Muscle burst)
 *   • Total frame count + window size note
 *   • Reset button to clear accumulators
 *
 * Props: pass the return value of useArtifactStats() directly.
 */
import React from 'react'
import type { ArtifactStats } from '../hooks/useArtifactStats'

interface Props {
  stats: ArtifactStats
  connected: boolean
}

const CAUSE_LABELS: Record<string, string> = {
  amplitude: 'Amplitude spike',
  motion:    'Head movement',
  kurtosis:  'Muscle burst',
}

const CAUSE_COLOURS: Record<string, { bg: string; border: string; text: string }> = {
  amplitude: { bg: 'rgba(248,81,73,0.1)',   border: 'rgba(248,81,73,0.3)',   text: '#f85149' },
  motion:    { bg: 'rgba(227,179,65,0.1)',  border: 'rgba(227,179,65,0.3)',  text: '#e3b341' },
  kurtosis:  { bg: 'rgba(210,118,255,0.1)', border: 'rgba(210,118,255,0.3)', text: '#d27bff' },
}

const DEFAULT_COLOUR = { bg: 'rgba(139,148,158,0.1)', border: 'rgba(139,148,158,0.25)', text: '#8b949e' }

function rateColour(rate: number): string {
  if (rate >= 0.30) return '#f85149'
  if (rate >= 0.10) return '#e3b341'
  return '#3fb950'
}

const S: Record<string, React.CSSProperties> = {
  container: { display: 'flex', flexDirection: 'column', gap: 12 },
  rateRow: { display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' },
  rateLabel: { fontSize: 12, color: '#8b949e' },
  rateValue: { fontSize: 26, fontWeight: 700 },
  barBg: { height: 8, background: '#21262d', borderRadius: 4, overflow: 'hidden' },
  causes: { display: 'flex', flexWrap: 'wrap', gap: 6 },
  footer: { fontSize: 11, color: '#484f58', display: 'flex', justifyContent: 'space-between', alignItems: 'center' },
  resetBtn: {
    fontSize: 11, fontWeight: 600, color: '#8b949e',
    background: 'none', border: '1px solid #30363d',
    borderRadius: 6, padding: '2px 10px', cursor: 'pointer',
  },
}

export default function ArtifactStatsPanel({ stats, connected }: Props) {
  const { totalFrames, rejectedFrames, rejectRate, causeCounts, windowSize, reset } = stats
  const ratePct = (rejectRate * 100).toFixed(1)
  const colour  = rateColour(rejectRate)

  if (!connected) {
    return (
      <div style={{ color: '#484f58', fontSize: 13, padding: '8px 0' }}>
        Connect a device to begin artifact tracking.
      </div>
    )
  }

  return (
    <div style={S.container}>
      {/* Rate gauge */}
      <div>
        <div style={S.rateRow}>
          <span style={S.rateLabel}>Rejection rate (last {windowSize} frames)</span>
          <span style={{ ...S.rateValue, color: colour }}>{ratePct}%</span>
        </div>
        <div style={S.barBg}>
          <div style={{
            height: '100%',
            width: `${Math.min(rejectRate * 100, 100)}%`,
            background: colour,
            borderRadius: 4,
            transition: 'width 0.3s ease, background 0.3s ease',
          }} />
        </div>
      </div>

      {/* Per-cause pills */}
      {Object.keys(causeCounts).length > 0 && (
        <div style={S.causes}>
          {Object.entries(causeCounts)
            .sort((a, b) => b[1] - a[1])
            .map(([cause, count]) => {
              const c = CAUSE_COLOURS[cause] ?? DEFAULT_COLOUR
              return (
                <span key={cause} style={{
                  display: 'inline-flex', alignItems: 'center', gap: 5,
                  padding: '3px 10px', borderRadius: 20, fontSize: 12, fontWeight: 600,
                  background: c.bg, border: `1px solid ${c.border}`, color: c.text,
                }}>
                  {CAUSE_LABELS[cause] ?? cause}
                  <span style={{
                    marginLeft: 2, fontSize: 11, fontWeight: 700,
                    background: c.border, borderRadius: 10,
                    padding: '0 5px', color: c.text,
                  }}>{count}</span>
                </span>
              )
            })}
        </div>
      )}

      {Object.keys(causeCounts).length === 0 && totalFrames > 0 && (
        <div style={{ fontSize: 12, color: '#3fb950', display: 'flex', alignItems: 'center', gap: 6 }}>
          <span>✓</span> No artifacts detected in this window
        </div>
      )}

      {/* Footer: frame counts + reset */}
      <div style={S.footer}>
        <span>
          {rejectedFrames} / {totalFrames} frames rejected
        </span>
        <button style={S.resetBtn} onClick={reset} title="Reset artifact statistics">
          Reset
        </button>
      </div>
    </div>
  )
}
