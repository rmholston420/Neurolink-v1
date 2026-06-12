/**
 * HRVCoherenceTrainer
 *
 * Visual interface for useHRVCoherence.
 *
 * Layout:
 *   - Animated SVG breathing pacer ring (expands on inhale, contracts on exhale)
 *   - Coherence score gauge bar 0–100 with colour gradient
 *   - Breaths/min rate selector (4–7)
 *   - HRV stats: mean RMSSD, sigma, n samples
 */
import React from 'react'
import type { HRVCoherenceReturn } from '../hooks/useHRVCoherence'

interface Props {
  hrv: HRVCoherenceReturn
}

const PACER_MAX = 80
const PACER_MIN = 42
const CX = 100, CY = 100

const st: Record<string, React.CSSProperties> = {
  root:     { display: 'flex', flexDirection: 'column', gap: 16 },
  topRow:   { display: 'flex', gap: 16, alignItems: 'center', flexWrap: 'wrap' },
  pacerWrap:{ flexShrink: 0 },
  right:    { flex: 1, display: 'flex', flexDirection: 'column', gap: 10, minWidth: 120 },
  scoreLabel: { fontSize: 11, fontWeight: 700, color: '#8b949e', textTransform: 'uppercase', letterSpacing: 0.8 },
  scoreVal: (colour: string): React.CSSProperties => ({
    fontSize: 32, fontWeight: 700, color: colour, lineHeight: 1,
    fontVariantNumeric: 'tabular-nums',
  }),
  scoreSub:   { fontSize: 11, color: '#484f58' },
  gaugeTrack: {
    height: 8, borderRadius: 4,
    background: '#21262d', overflow: 'hidden',
  },
  gaugeFill: (score: number, colour: string): React.CSSProperties => ({
    height: '100%', borderRadius: 4,
    width: `${score}%`,
    background: colour,
    transition: 'width 600ms ease, background 600ms ease',
  }),
  stateLabel: (inhale: boolean): React.CSSProperties => ({
    fontSize: 12, fontWeight: 700,
    color: inhale ? '#58a6ff' : '#3fb950',
    textAlign: 'center', letterSpacing: 1,
    textTransform: 'uppercase',
  }),
  rateRow:  { display: 'flex', alignItems: 'center', gap: 8 },
  rateLabel:{ fontSize: 11, color: '#8b949e', minWidth: 80 },
  rateBtn:  (active: boolean): React.CSSProperties => ({
    width: 28, height: 28, borderRadius: 6, fontSize: 13, fontWeight: 700,
    cursor: 'pointer', border: `1px solid ${active ? '#58a6ff' : '#30363d'}`,
    background: active ? 'rgba(88,166,255,0.12)' : 'transparent',
    color: active ? '#58a6ff' : '#484f58',
    transition: 'all 150ms',
  }),
  statsRow: {
    display: 'flex', gap: 12, flexWrap: 'wrap',
    fontSize: 12, color: '#8b949e',
    padding: '8px 10px',
    background: 'rgba(22,27,34,0.5)',
    border: '1px solid #21262d', borderRadius: 6,
  },
  statItem: { display: 'flex', flexDirection: 'column', gap: 1 },
  statKey:  { fontSize: 10, color: '#484f58', textTransform: 'uppercase', letterSpacing: 0.6 },
  statVal:  { color: '#e6edf3', fontVariantNumeric: 'tabular-nums' },
  noHRV: { fontSize: 12, color: '#484f58', fontStyle: 'italic' },
}

export default function HRVCoherenceTrainer({ hrv }: Props) {
  const inhaling = hrv.pacerState === 'inhale'
  const r = PACER_MIN + (PACER_MAX - PACER_MIN) * (inhaling ? hrv.pacerPhase : 1 - hrv.pacerPhase)

  return (
    <div style={st.root}>
      <div style={st.topRow}>
        {/* Breathing pacer ring */}
        <div style={st.pacerWrap}>
          <svg width={200} height={200} viewBox="0 0 200 200">
            {/* Guide circles */}
            <circle cx={CX} cy={CY} r={PACER_MAX} fill="none" stroke="#21262d" strokeWidth={1} strokeDasharray="4 4" />
            <circle cx={CX} cy={CY} r={PACER_MIN} fill="none" stroke="#21262d" strokeWidth={1} strokeDasharray="4 4" />
            {/* Animated pacer */}
            <circle
              cx={CX} cy={CY} r={r}
              fill={inhaling ? 'rgba(88,166,255,0.08)' : 'rgba(63,185,80,0.08)'}
              stroke={inhaling ? '#388bfd' : '#3fb950'}
              strokeWidth={3}
              style={{ transition: 'r 0.1s ease, fill 0.3s, stroke 0.3s' }}
            />
            {/* Centre label */}
            <text x={CX} y={CY - 6}  textAnchor="middle" fontSize={13} fontWeight={700}
              fill={inhaling ? '#58a6ff' : '#3fb950'}>
              {inhaling ? 'Inhale' : 'Exhale'}
            </text>
            <text x={CX} y={CY + 14} textAnchor="middle" fontSize={11} fill="#484f58">
              {hrv.breathsPerMin} /min
            </text>
          </svg>
        </div>

        {/* Coherence score + gauge */}
        <div style={st.right}>
          <span style={st.scoreLabel}>HRV Coherence</span>
          <span style={st.scoreVal(hrv.coherenceColour)}>
            {Math.round(hrv.coherenceScore)}
            <span style={{ fontSize: 14, color: '#484f58' }}>/100</span>
          </span>
          <span style={{ ...st.scoreSub, color: hrv.coherenceColour }}>{hrv.coherenceLabel}</span>
          <div style={st.gaugeTrack}>
            <div style={st.gaugeFill(hrv.coherenceScore, hrv.coherenceColour)} />
          </div>
          <span style={st.scoreSub}>Target: ≥ 66 for High coherence</span>
        </div>
      </div>

      {/* Rate selector */}
      <div style={st.rateRow}>
        <span style={st.rateLabel}>Breaths / min</span>
        {[4, 5, 6, 7].map(b => (
          <button key={b} style={st.rateBtn(hrv.breathsPerMin === b)}
            onClick={() => hrv.setBreathsPerMin(b)}>
            {b}
          </button>
        ))}
        <span style={{ fontSize: 11, color: '#484f58' }}>({(60 / hrv.breathsPerMin).toFixed(0)}s cycle)</span>
      </div>

      {/* HRV stats */}
      {hrv.hrvMean !== null ? (
        <div style={st.statsRow}>
          <div style={st.statItem}>
            <span style={st.statKey}>RMSSD mean</span>
            <span style={st.statVal}>{hrv.hrvMean.toFixed(1)} ms</span>
          </div>
          <div style={st.statItem}>
            <span style={st.statKey}>RMSSD σ</span>
            <span style={st.statVal}>{hrv.hrvSigma?.toFixed(1)} ms</span>
          </div>
          <div style={st.statItem}>
            <span style={st.statKey}>Samples</span>
            <span style={st.statVal}>{hrv.nSamples}</span>
          </div>
          <div style={st.statItem}>
            <span style={st.statKey}>Resonance target</span>
            <span style={st.statVal}>0.1 Hz</span>
          </div>
        </div>
      ) : (
        <p style={st.noHRV}>No HRV data — connect a PPG-capable device (Muse S, Muse Athena).</p>
      )}
    </div>
  )
}
