/**
 * PersonalBaseline — Tier 2 display component
 * Shows deviation from rolling personal norm for alpha / theta / beta / focus / fatigue.
 */
import React from 'react'
import type { PersonalBaselineResult } from '../hooks/usePersonalBaseline'

interface Props {
  result: PersonalBaselineResult
}

function DevRow({ label, dev, base }: { label: string; dev: number | null; base: number | null }) {
  const pct   = dev !== null ? Math.round(dev) : null
  const color = pct === null ? '#8b949e'
    : pct > 10  ? '#3fb950'
    : pct < -10 ? '#f85149'
    : '#d2a679'
  const arrow = pct === null ? '–'
    : pct > 0 ? `▲ +${pct}%`
    : `▼ ${pct}%`
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center',
      padding: '5px 0', borderBottom: '1px solid #21262d', fontSize: 13 }}>
      <span style={{ color: '#8b949e', textTransform: 'uppercase', letterSpacing: 0.5, fontSize: 11 }}>{label}</span>
      <div style={{ display: 'flex', gap: 14, alignItems: 'center' }}>
        {base !== null && (
          <span style={{ color: '#484f58', fontSize: 11 }}>
            baseline {base > 1 ? base.toFixed(1) : base.toFixed(3)}
          </span>
        )}
        <span style={{ color, fontWeight: 700, minWidth: 70, textAlign: 'right' }}>{arrow}</span>
      </div>
    </div>
  )
}

export default function PersonalBaseline({ result }: Props) {
  const { baseline, deviation, sampleCount, resetBaseline } = result

  return (
    <div style={{ color: '#cdd9e5' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
        <span style={{ fontSize: 11, color: '#8b949e' }}>
          {sampleCount < 30
            ? `Calibrating… ${sampleCount}/30 samples`
            : `Personal baseline · ${sampleCount} samples`}
        </span>
        <button
          onClick={resetBaseline}
          style={{
            padding: '3px 8px', fontSize: 11, background: '#21262d',
            color: '#8b949e', border: '1px solid #30363d', borderRadius: 5, cursor: 'pointer',
          }}
        >↺ Reset</button>
      </div>

      {sampleCount < 10 && (
        <p style={{ fontSize: 12, color: '#484f58', textAlign: 'center', padding: '12px 0' }}>
          Collecting baseline — keep the headset on and stay still
        </p>
      )}

      {sampleCount >= 10 && (
        <>
          <DevRow label="Alpha"   dev={deviation.alpha}        base={baseline.alpha} />
          <DevRow label="Theta"   dev={deviation.theta}        base={baseline.theta} />
          <DevRow label="Beta"    dev={deviation.beta}         base={baseline.beta} />
          <DevRow label="Focus"   dev={deviation.focusScore}   base={baseline.focusScore} />
          <DevRow label="Fatigue" dev={deviation.fatigueScore} base={baseline.fatigueScore} />
        </>
      )}
    </div>
  )
}
