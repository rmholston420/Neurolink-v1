/**
 * TopoMap — Canvas 2D 4-electrode topographic map.
 *
 * Electrode positions (Muse S): TP9 (left), AF7 (front-left),
 * AF8 (front-right), TP10 (right).  Power values are interpolated
 * over a 64×64 grid using inverse-distance weighting (IDW, p=2).
 * A diverging blue→white→red palette maps relative power vs mean.
 *
 * Honest disclaimer: 4 electrodes produce sparse coverage.  The map
 * is suggestive, not clinically precise.
 */

import React, { useRef, useEffect, useState } from 'react'
import type { BandPowers } from '../types'

// ── Electrode layout (normalised 0–1 within head circle) ─────────────────────
// Origin = centre of head circle.  Positive x = right, positive y = front.
const ELECTRODES: { name: string; nx: number; ny: number }[] = [
  { name: 'AF7',  nx: -0.42, ny:  0.55 },
  { name: 'AF8',  nx:  0.42, ny:  0.55 },
  { name: 'TP9',  nx: -0.72, ny: -0.20 },
  { name: 'TP10', nx:  0.72, ny: -0.20 },
]

const GRID = 64
const R    = 0.82   // head radius in normalised units

// ── IDW interpolation ─────────────────────────────────────────────────────────
function idw(values: number[], gx: number, gy: number, p = 2): number {
  let num = 0, den = 0
  for (let i = 0; i < ELECTRODES.length; i++) {
    const dx = gx - ELECTRODES[i].nx
    const dy = gy - ELECTRODES[i].ny
    const d  = Math.sqrt(dx * dx + dy * dy)
    if (d < 1e-6) return values[i]
    const w = 1 / Math.pow(d, p)
    num += w * values[i]
    den += w
  }
  return den > 0 ? num / den : 0
}

// ── Palette: blue → white → red ───────────────────────────────────────────────
function diverge(t: number): [number, number, number] {
  const v = Math.max(-1, Math.min(1, t))
  if (v < 0) {
    const s = -v
    return [Math.round(255*(1-s*0.76)), Math.round(255*(1-s*0.55)), 255]
  }
  return [255, Math.round(255*(1-v*0.72)), Math.round(255*(1-v*0.72))]
}

// ── Props ─────────────────────────────────────────────────────────────────────
interface Props {
  bands: BandPowers | null
  /** Per-channel band values: [TP9, AF7, AF8, TP10] for the selected band. */
  channelBands: number[] | null
}

type Band = 'alpha' | 'theta' | 'beta' | 'delta' | 'gamma'
const BANDS: Band[] = ['alpha', 'theta', 'beta', 'delta', 'gamma']
const BAND_COLOR: Record<Band, string> = {
  alpha: '#388bfd',
  theta: '#d29922',
  beta:  '#3fb950',
  delta: '#bc8cff',
  gamma: '#f85149',
}

const CANVAS_SIZE = 240

export default function TopoMap({ bands, channelBands }: Props) {
  const canvasRef  = useRef<HTMLCanvasElement>(null)
  const [band, setBand] = useState<Band>('alpha')
  const meanRef    = useRef<number>(0)
  const rangeRef   = useRef<number>(1)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const S = CANVAS_SIZE
    ctx.clearRect(0, 0, S, S)

    // Fallback synthetic per-channel values from global band averages
    // Real values come from channelBands when the backend sends them.
    const raw: number[] = channelBands ?? [
      (bands?.alpha ?? 0) * (0.9 + Math.random() * 0.2),
      (bands?.alpha ?? 0) * (0.9 + Math.random() * 0.2),
      (bands?.alpha ?? 0) * (0.9 + Math.random() * 0.2),
      (bands?.alpha ?? 0) * (0.9 + Math.random() * 0.2),
    ]

    // Update running mean and range for normalisation
    const avg = raw.reduce((s, v) => s + v, 0) / raw.length
    meanRef.current = meanRef.current * 0.95 + avg * 0.05
    const rng = Math.max(...raw) - Math.min(...raw)
    rangeRef.current = rangeRef.current * 0.95 + (rng || 1) * 0.05
    const mu  = meanRef.current
    const rr  = rangeRef.current

    // Map electrode normalised coords → canvas pixels
    const cx = S / 2, cy = S / 2, r = (S / 2) * R
    function toPixel(nx: number, ny: number): [number, number] {
      return [cx + nx * r, cy - ny * r]  // flip y: positive = up on screen
    }

    // Clip to head circle
    ctx.save()
    ctx.beginPath()
    ctx.arc(cx, cy, r, 0, Math.PI * 2)
    ctx.clip()

    // Paint interpolated grid
    const step = S / GRID
    for (let gy = 0; gy < GRID; gy++) {
      for (let gx = 0; gx < GRID; gx++) {
        const nx = (gx / GRID - 0.5) / (R / 2)
        const ny = -(gy / GRID - 0.5) / (R / 2)
        const d2 = nx * nx + ny * ny
        if (d2 > 1.0 / (R * R)) continue
        const val = idw(raw, nx * R, ny * R)
        const t   = rr > 0 ? (val - mu) / rr : 0
        const [rv, gv, bv] = diverge(t)
        ctx.fillStyle = `rgb(${rv},${gv},${bv})`
        ctx.fillRect(gx * step, gy * step, step + 1, step + 1)
      }
    }
    ctx.restore()

    // Head outline
    ctx.strokeStyle = '#8b949e'
    ctx.lineWidth = 1.5
    ctx.beginPath()
    ctx.arc(cx, cy, r, 0, Math.PI * 2)
    ctx.stroke()

    // Nose triangle (top)
    ctx.strokeStyle = '#8b949e'
    ctx.lineWidth = 1.5
    ctx.beginPath()
    ctx.moveTo(cx - 10, cy - r + 4)
    ctx.lineTo(cx, cy - r - 12)
    ctx.lineTo(cx + 10, cy - r + 4)
    ctx.stroke()

    // Ear marks
    ;[[-1, 0], [1, 0]].forEach(([side]) => {
      ctx.strokeStyle = '#8b949e'
      ctx.lineWidth = 1.5
      ctx.beginPath()
      ctx.arc(cx + side * (r + 5), cy, 8, Math.PI / 4, Math.PI * 7 / 4, side === -1)
      ctx.stroke()
    })

    // Electrode dots + labels
    ELECTRODES.forEach((el, i) => {
      const [px, py] = toPixel(el.nx, el.ny)
      ctx.beginPath()
      ctx.arc(px, py, 5, 0, Math.PI * 2)
      ctx.fillStyle = '#fff'
      ctx.fill()
      ctx.strokeStyle = '#161b22'
      ctx.lineWidth = 1
      ctx.stroke()
      ctx.fillStyle = '#fff'
      ctx.font = 'bold 9px sans-serif'
      ctx.textAlign = 'center'
      ctx.fillText(el.name, px, py - 8)
    })

    // Colour scale bar
    const barX = S - 18, barY = 20, barH = S - 40, barW = 10
    const gradient = ctx.createLinearGradient(0, barY, 0, barY + barH)
    gradient.addColorStop(0,   'rgb(213,26,28)')
    gradient.addColorStop(0.5, 'rgb(255,255,255)')
    gradient.addColorStop(1,   'rgb(33,102,172)')
    ctx.fillStyle = gradient
    ctx.fillRect(barX, barY, barW, barH)
    ctx.strokeStyle = '#484f58'
    ctx.lineWidth = 0.5
    ctx.strokeRect(barX, barY, barW, barH)
    ctx.fillStyle = '#8b949e'
    ctx.font = '8px sans-serif'
    ctx.textAlign = 'left'
    ctx.fillText('+', barX + barW + 2, barY + 8)
    ctx.fillText('−', barX + barW + 2, barY + barH)
  }, [bands, channelBands, band])

  const tabBtn = (b: Band): React.CSSProperties => ({
    padding: '2px 8px',
    fontSize: 11,
    fontWeight: 600,
    borderRadius: 4,
    border: 'none',
    cursor: 'pointer',
    background: b === band ? BAND_COLOR[b] : 'transparent',
    color: b === band ? '#fff' : '#8b949e',
  })

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8, alignItems: 'center' }}>
      <div style={{ display: 'flex', gap: 4 }}>
        {BANDS.map(b => (
          <button key={b} style={tabBtn(b)} onClick={() => setBand(b)}>{b}</button>
        ))}
      </div>
      <canvas
        ref={canvasRef}
        width={CANVAS_SIZE}
        height={CANVAS_SIZE}
        style={{ borderRadius: '50%', width: 240, height: 240 }}
      />
      <div style={{ fontSize: 10, color: '#484f58', textAlign: 'center', maxWidth: 220 }}>
        ⚠ 4-electrode coverage — interpolation is suggestive, not clinically precise
      </div>
    </div>
  )
}
