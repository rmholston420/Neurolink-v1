/**
 * BandTrend — 60-second rolling sparklines for all five EEG bands.
 *
 * Each band gets its own SVG polyline scrolling left over a 60-second
 * window.  The y-axis auto-scales to the session min/max per band.
 * A horizontal dashed baseline reference is drawn at the calibration
 * alpha value (if available) on the alpha panel only.
 */

import React, { useRef, useEffect, useState } from 'react'
import type { BandPowers } from '../types'

const WINDOW_SEC = 60
const UPDATE_HZ  = 4    // SSE frames per second
const MAX_PTS    = WINDOW_SEC * UPDATE_HZ   // 240 points

const BAND_META: { key: keyof BandPowers; label: string; color: string; hz: string }[] = [
  { key: 'alpha', label: 'Alpha',  color: '#388bfd', hz: '8–13 Hz' },
  { key: 'theta', label: 'Theta',  color: '#d29922', hz: '4–8 Hz'  },
  { key: 'beta',  label: 'Beta',   color: '#3fb950', hz: '13–30 Hz'},
  { key: 'delta', label: 'Delta',  color: '#bc8cff', hz: '0.5–4 Hz'},
  { key: 'gamma', label: 'Gamma',  color: '#f85149', hz: '30–100 Hz'},
]

const W = 300, H = 52

function sparkPath(pts: number[], min: number, max: number): string {
  if (pts.length < 2) return ''
  const range = max - min || 1
  return pts.map((v, i) => {
    const x = (i / (MAX_PTS - 1)) * W
    const y = H - 4 - ((v - min) / range) * (H - 8)
    return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`
  }).join(' ')
}

interface Props {
  bands: BandPowers | null
  baselineAlpha?: number | null
}

export default function BandTrend({ bands, baselineAlpha }: Props) {
  const [expanded, setExpanded] = useState<keyof BandPowers | null>('alpha')
  const history = useRef<Record<keyof BandPowers, number[]>>({
    alpha: [], theta: [], beta: [], delta: [], gamma: [],
  })
  const minMax = useRef<Record<keyof BandPowers, [number, number]>>({
    alpha: [1, 0], theta: [1, 0], beta: [1, 0], delta: [1, 0], gamma: [1, 0],
  })
  const [, forceUpdate] = useState(0)

  useEffect(() => {
    if (!bands) return
    BAND_META.forEach(({ key }) => {
      const val = bands[key]
      const arr = history.current[key]
      arr.push(val)
      if (arr.length > MAX_PTS) arr.shift()
      const [mn, mx] = minMax.current[key]
      minMax.current[key] = [
        arr.length === 1 ? val : Math.min(mn, val),
        arr.length === 1 ? val : Math.max(mx, val),
      ]
    })
    forceUpdate(n => n + 1)
  }, [bands])

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      {BAND_META.map(({ key, label, color, hz }) => {
        const pts = history.current[key]
        const [mn, mx] = minMax.current[key]
        const current = pts[pts.length - 1] ?? null
        const isExpanded = expanded === key

        // Baseline delta for alpha
        let deltaEl: React.ReactNode = null
        if (key === 'alpha' && baselineAlpha != null && current != null) {
          const delta = ((current - baselineAlpha) / (baselineAlpha + 1e-9)) * 100
          const sign  = delta >= 0 ? '+' : ''
          deltaEl = (
            <span style={{ fontSize: 10, color: delta >= 0 ? '#3fb950' : '#f85149', marginLeft: 6 }}>
              {sign}{delta.toFixed(0)}% vs baseline
            </span>
          )
        }

        // Baseline y position on SVG for alpha
        let baselineY: number | null = null
        if (key === 'alpha' && baselineAlpha != null) {
          const range = mx - mn || 1
          baselineY = H - 4 - ((baselineAlpha - mn) / range) * (H - 8)
        }

        return (
          <div key={key}>
            <button
              style={{ display: 'flex', alignItems: 'center', gap: 8, width: '100%',
                       background: 'none', border: 'none', cursor: 'pointer', padding: '2px 0' }}
              onClick={() => setExpanded(isExpanded ? null : key)}
            >
              <span style={{ width: 10, height: 10, borderRadius: '50%', background: color, flexShrink: 0 }} />
              <span style={{ fontSize: 12, fontWeight: 600, color: '#cdd9e5' }}>{label}</span>
              <span style={{ fontSize: 10, color: '#484f58' }}>{hz}</span>
              {current != null && (
                <span style={{ fontSize: 11, color, marginLeft: 'auto' }}>{current.toFixed(3)}</span>
              )}
              {deltaEl}
              <span style={{ fontSize: 10, color: '#484f58', marginLeft: 4 }}>{isExpanded ? '▲' : '▼'}</span>
            </button>

            {isExpanded && (
              <svg
                width={W} height={H}
                style={{ width: '100%', height: H, display: 'block', marginTop: 2 }}
                viewBox={`0 0 ${W} ${H}`}
              >
                {/* Background */}
                <rect width={W} height={H} rx={4} fill='#0d1117' />

                {/* Baseline reference line (alpha only) */}
                {baselineY != null && (
                  <>
                    <line x1={0} y1={baselineY} x2={W} y2={baselineY}
                      stroke='#388bfd' strokeWidth={1} strokeDasharray='4 3' opacity={0.5} />
                    <text x={4} y={baselineY - 3} fill='#388bfd' fontSize={8} opacity={0.7}>baseline</text>
                  </>
                )}

                {/* Grid lines */}
                {[0.25, 0.5, 0.75].map(frac => {
                  const y = frac * H
                  return <line key={frac} x1={0} y1={y} x2={W} y2={y}
                    stroke='#21262d' strokeWidth={0.5} />
                })}

                {/* Sparkline */}
                {pts.length > 1 && (
                  <path d={sparkPath(pts, mn, mx)}
                    fill='none' stroke={color} strokeWidth={1.5}
                    strokeLinejoin='round' strokeLinecap='round' />
                )}

                {/* Y-axis labels */}
                <text x={W - 2} y={10} fill='#484f58' fontSize={8} textAnchor='end'>
                  {mx.toFixed(3)}
                </text>
                <text x={W - 2} y={H - 2} fill='#484f58' fontSize={8} textAnchor='end'>
                  {mn.toFixed(3)}
                </text>

                {/* Time labels */}
                <text x={2} y={H - 2} fill='#484f58' fontSize={8}>−60s</text>
                <text x={W / 2} y={H - 2} fill='#484f58' fontSize={8} textAnchor='middle'>−30s</text>
                <text x={W - 2} y={H - 2} fill='#484f58' fontSize={8} textAnchor='end'>now</text>
              </svg>
            )}
          </div>
        )
      })}
    </div>
  )
}
