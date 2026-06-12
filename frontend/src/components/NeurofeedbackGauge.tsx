/**
 * NeurofeedbackGauge — radial SVG arc gauge.
 *
 * Displays Focus and Fatigue scores (0–100) as arc gauges with:
 *  - 4-second exponential moving average to smooth noise
 *  - Baseline reference tick
 *  - Colour transition: green → amber → red (Focus inverts: red=low)
 *  - Session min/max indicators
 */

import React, { useRef, useEffect, useState } from 'react'

interface GaugeProps {
  value: number        // 0–100
  label: string
  color: string
  baselineRef?: number | null
  unit?: string
}

const SVG_SIZE = 120
const CX = SVG_SIZE / 2, CY = SVG_SIZE / 2 + 10
const RADIUS = 44
const START_ANGLE = -220  // degrees
const SWEEP      = 260    // degrees

function polar(cx: number, cy: number, r: number, deg: number): [number, number] {
  const rad = (deg * Math.PI) / 180
  return [cx + r * Math.cos(rad), cy + r * Math.sin(rad)]
}

function arcPath(cx: number, cy: number, r: number, startDeg: number, endDeg: number): string {
  const [x1, y1] = polar(cx, cy, r, startDeg)
  const [x2, y2] = polar(cx, cy, r, endDeg)
  const large = endDeg - startDeg > 180 ? 1 : 0
  return `M${x1.toFixed(2)},${y1.toFixed(2)} A${r},${r} 0 ${large},1 ${x2.toFixed(2)},${y2.toFixed(2)}`
}

function valueColor(value: number, invert = false): string {
  const v = invert ? 100 - value : value
  if (v < 35) return '#3fb950'
  if (v < 65) return '#d29922'
  return '#f85149'
}

function SingleGauge({ value, label, color, baselineRef }: GaugeProps) {
  const clamp = Math.max(0, Math.min(100, value))
  const endAngle = START_ANGLE + (clamp / 100) * SWEEP

  // Baseline tick
  let baselineTick: React.ReactNode = null
  if (baselineRef != null) {
    const bAngle = START_ANGLE + (Math.max(0, Math.min(100, baselineRef)) / 100) * SWEEP
    const [bx1, by1] = polar(CX, CY, RADIUS - 8, bAngle)
    const [bx2, by2] = polar(CX, CY, RADIUS + 4, bAngle)
    baselineTick = (
      <line x1={bx1} y1={by1} x2={bx2} y2={by2}
        stroke='#484f58' strokeWidth={2} strokeLinecap='round' />
    )
  }

  return (
    <svg width={SVG_SIZE} height={SVG_SIZE}
      viewBox={`0 0 ${SVG_SIZE} ${SVG_SIZE}`}
      style={{ width: SVG_SIZE, height: SVG_SIZE }}>
      {/* Track */}
      <path d={arcPath(CX, CY, RADIUS, START_ANGLE, START_ANGLE + SWEEP)}
        fill='none' stroke='#21262d' strokeWidth={7} strokeLinecap='round' />
      {/* Fill */}
      {clamp > 0 && (
        <path d={arcPath(CX, CY, RADIUS, START_ANGLE, endAngle)}
          fill='none' stroke={color} strokeWidth={7} strokeLinecap='round'
          style={{ transition: 'stroke-dashoffset 0.4s ease' }} />
      )}
      {/* Baseline tick */}
      {baselineTick}
      {/* Value text */}
      <text x={CX} y={CY + 5} textAnchor='middle'
        fill={color} fontSize={20} fontWeight='bold'>
        {Math.round(clamp)}
      </text>
      {/* Label */}
      <text x={CX} y={CY + 20} textAnchor='middle'
        fill='#8b949e' fontSize={10}>
        {label}
      </text>
      {/* Min/max ticks */}
      {[0, 25, 50, 75, 100].map(pct => {
        const deg = START_ANGLE + (pct / 100) * SWEEP
        const [tx1, ty1] = polar(CX, CY, RADIUS - 5, deg)
        const [tx2, ty2] = polar(CX, CY, RADIUS - 2, deg)
        return <line key={pct} x1={tx1} y1={ty1} x2={tx2} y2={ty2}
          stroke='#30363d' strokeWidth={1} />
      })}
    </svg>
  )
}

interface Props {
  focusScore: number
  fatigueScore: number
  focusState: string
  baselineAlpha?: number | null
}

const SMOOTH_ALPHA = 0.18   // ~4-second smoothing at 4 Hz

export default function NeurofeedbackGauge({ focusScore, fatigueScore, focusState, baselineAlpha }: Props) {
  const smoothFocus   = useRef(focusScore)
  const smoothFatigue = useRef(fatigueScore)
  const [display, setDisplay] = useState({ focus: focusScore, fatigue: fatigueScore })
  const minMaxFocus   = useRef<[number, number]>([focusScore, focusScore])
  const minMaxFatigue = useRef<[number, number]>([fatigueScore, fatigueScore])

  useEffect(() => {
    smoothFocus.current   = smoothFocus.current   * (1 - SMOOTH_ALPHA) + focusScore   * SMOOTH_ALPHA
    smoothFatigue.current = smoothFatigue.current * (1 - SMOOTH_ALPHA) + fatigueScore * SMOOTH_ALPHA
    const sf = smoothFocus.current, sg = smoothFatigue.current
    minMaxFocus.current   = [Math.min(minMaxFocus.current[0],   sf), Math.max(minMaxFocus.current[1],   sf)]
    minMaxFatigue.current = [Math.min(minMaxFatigue.current[0], sg), Math.max(minMaxFatigue.current[1], sg)]
    setDisplay({ focus: sf, fatigue: sg })
  }, [focusScore, fatigueScore])

  const focusColor   = valueColor(display.focus, true)   // inverted: high focus = green
  const fatigueColor = valueColor(display.fatigue, false) // high fatigue = red

  const stateColor: Record<string, string> = {
    focused: '#3fb950', alert: '#3fb950',
    neutral: '#d29922',
    drowsy: '#f85149', fatigued: '#f85149', unknown: '#484f58',
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8, alignItems: 'center' }}>
      <div style={{ display: 'flex', gap: 16 }}>
        <SingleGauge
          value={display.focus}
          label='Focus'
          color={focusColor}
          baselineRef={baselineAlpha ? 50 : null}
        />
        <SingleGauge
          value={display.fatigue}
          label='Fatigue'
          color={fatigueColor}
        />
      </div>

      {/* State badge */}
      <div style={{
        padding: '3px 12px', borderRadius: 20, fontSize: 12, fontWeight: 600,
        background: `${stateColor[focusState] ?? '#484f58'}22`,
        color: stateColor[focusState] ?? '#484f58',
        border: `1px solid ${stateColor[focusState] ?? '#484f58'}44`,
      }}>
        {focusState}
      </div>

      {/* Session range */}
      <div style={{ display: 'flex', gap: 16, fontSize: 10, color: '#484f58' }}>
        <span>Focus range: {minMaxFocus.current[0].toFixed(0)}–{minMaxFocus.current[1].toFixed(0)}</span>
        <span>Fatigue range: {minMaxFatigue.current[0].toFixed(0)}–{minMaxFatigue.current[1].toFixed(0)}</span>
      </div>

      <div style={{ fontSize: 10, color: '#484f58', textAlign: 'center' }}>
        4 s EMA smoothing · baseline ref tick shown when calibrated
      </div>
    </div>
  )
}
