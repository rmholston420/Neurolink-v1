import React from 'react'
import type { BandPowers } from '../types'

interface Props {
  bands: BandPowers | null
}

const BAND_COLORS: Record<string, string> = {
  delta: '#6e7681',
  theta: '#388bfd',
  alpha: '#3fb950',
  beta: '#d29922',
  gamma: '#bc8cff',
}

const BAND_LABELS: Record<string, string> = {
  delta: '\u03b4 Delta',
  theta: '\u03b8 Theta',
  alpha: '\u03b1 Alpha',
  beta: '\u03b2 Beta',
  gamma: '\u03b3 Gamma',
}

const styles: Record<string, React.CSSProperties> = {
  container: { display: 'flex', flexDirection: 'column', gap: 10 },
  row: { display: 'flex', alignItems: 'center', gap: 10 },
  label: { width: 70, fontSize: 13, color: '#8b949e', flexShrink: 0 },
  barBg: {
    flex: 1,
    height: 20,
    background: '#21262d',
    borderRadius: 4,
    overflow: 'hidden',
  },
  value: { width: 44, fontSize: 12, color: '#8b949e', textAlign: 'right' },
}

export default function BandPowerChart({ bands }: Props) {
  if (!bands) {
    return <div style={{ color: '#484f58', fontSize: 14 }}>No data</div>
  }

  const entries = Object.entries(bands) as [string, number][]

  return (
    <div style={styles.container}>
      {entries.map(([band, value]) => (
        <div key={band} style={styles.row}>
          <span style={styles.label}>{BAND_LABELS[band] ?? band}</span>
          <div style={styles.barBg}>
            <div
              style={{
                height: '100%',
                width: `${Math.min(100, value * 100).toFixed(1)}%`,
                background: BAND_COLORS[band] ?? '#58a6ff',
                borderRadius: 4,
                transition: 'width 0.25s ease',
              }}
            />
          </div>
          <span style={styles.value}>{(value * 100).toFixed(1)}%</span>
        </div>
      ))}
    </div>
  )
}
