/**
 * HRVCoherenceTrainer — Tier 2
 * Real-time HRV coherence display + paced breathing guide.
 */
import React from 'react'
import type { HRVCoherenceResult } from '../hooks/useHRVCoherence'

interface Props {
  result: HRVCoherenceResult
  hrBpm: number | null
  hrv:   number | null
}

export default function HRVCoherenceTrainer({ result, hrBpm, hrv }: Props) {
  const {
    coherenceRatio, coherenceLabel, rrBufferLen,
    breathPhase, breathProgress,
    pacerBpm, setPacerBpm, isCoherent,
  } = result

  const coherenceColor =
    isCoherent ? '#3fb950' :
    coherenceRatio >= 0.3 ? '#d2a679' : '#8b949e'

  // Breathing ring
  const R = 36
  const circ = 2 * Math.PI * R
  const phasePct  = breathPhase === 'inhale' ? breathProgress : 1 - breathProgress
  const dashoffset = circ - circ * phasePct
  const ringColor  = breathPhase === 'inhale' ? '#58a6ff' : '#a371f7'

  return (
    <div style={{ color: '#cdd9e5' }}>
      {/* Coherence meter */}
      <div style={{ marginBottom: 14 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 5, fontSize: 12 }}>
          <span style={{ color: '#8b949e' }}>Coherence</span>
          <span style={{ color: coherenceColor, fontWeight: 700 }}>
            {coherenceLabel} ({(coherenceRatio * 100).toFixed(0)}%)
          </span>
        </div>
        <div style={{ height: 8, background: '#21262d', borderRadius: 4, overflow: 'hidden' }}>
          <div style={{
            height: '100%', width: `${coherenceRatio * 100}%`,
            background: coherenceColor,
            transition: 'width 0.8s ease, background 0.8s ease',
            borderRadius: 4,
          }} />
        </div>
        <div style={{ fontSize: 10, color: '#484f58', marginTop: 3 }}>
          {rrBufferLen < 10 ? `Building buffer… ${rrBufferLen}/10 beats` : `${rrBufferLen} beats sampled`}
        </div>
      </div>

      {/* Stats row */}
      <div style={{ display: 'flex', gap: 14, marginBottom: 14, fontSize: 12 }}>
        {hrBpm !== null && (
          <span style={{ color: '#8b949e' }}>HR <b style={{ color: '#cdd9e5' }}>{hrBpm.toFixed(0)}</b> bpm</span>
        )}
        {hrv !== null && (
          <span style={{ color: '#8b949e' }}>RMSSD <b style={{ color: '#cdd9e5' }}>{hrv.toFixed(1)}</b> ms</span>
        )}
      </div>

      {/* Breathing pacer */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 18 }}>
        <div style={{ flexShrink: 0 }}>
          <svg width={90} height={90} viewBox="0 0 90 90">
            <circle cx={45} cy={45} r={R} fill="none" stroke="#30363d" strokeWidth={7} />
            <circle
              cx={45} cy={45} r={R}
              fill="none"
              stroke={ringColor}
              strokeWidth={7}
              strokeDasharray={`${circ} ${circ}`}
              strokeDashoffset={dashoffset}
              strokeLinecap="round"
              style={{ transformOrigin: 'center', transform: 'rotate(-90deg)', transition: 'stroke-dashoffset 0.05s linear' }}
            />
            <text x={45} y={41} textAnchor="middle" fill="#cdd9e5" fontSize={11} fontWeight={600}>
              {breathPhase === 'inhale' ? 'Inhale' : 'Exhale'}
            </text>
            <text x={45} y={56} textAnchor="middle" fill={ringColor} fontSize={14} fontWeight={700}>
              {Math.round(breathProgress * (60 / pacerBpm / 2))}s
            </text>
          </svg>
        </div>
        <div style={{ flex: 1 }}>
          <div style={{ fontSize: 12, color: '#8b949e', marginBottom: 6 }}>
            Pacer rate: <b style={{ color: '#cdd9e5' }}>{pacerBpm.toFixed(1)} breath/min</b>
          </div>
          <input
            type="range" min={4} max={8} step={0.5}
            value={pacerBpm}
            onChange={e => setPacerBpm(Number(e.target.value))}
            style={{ width: '100%', accentColor: '#58a6ff' }}
          />
          <div style={{ fontSize: 10, color: '#484f58', marginTop: 3 }}>
            Target 5.5 breath/min for 0.1 Hz resonance
          </div>
          {isCoherent && (
            <div style={{ color: '#3fb950', fontSize: 12, fontWeight: 600, marginTop: 6 }}>
              ✓ Coherent — maintain this rhythm
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
