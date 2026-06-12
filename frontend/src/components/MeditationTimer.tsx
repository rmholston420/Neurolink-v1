/**
 * MeditationTimer — Tier 2 Feature
 * Countdown / count-up timer with configurable stages, bell cues, and SVG progress ring.
 */
import React, { useState, useEffect, useRef, useCallback } from 'react'

export interface MeditationStage {
  label: string
  durationSec: number
  color?: string
}

const DEFAULT_STAGES: MeditationStage[] = [
  { label: 'Settling',    durationSec: 120, color: '#58a6ff' },
  { label: 'Deepening',   durationSec: 300, color: '#a371f7' },
  { label: 'Open Awareness', durationSec: 600, color: '#3fb950' },
  { label: 'Integration', durationSec: 180, color: '#d2a679' },
]

function ringPath(cx: number, cy: number, r: number, pct: number): string {
  const circ = 2 * Math.PI * r
  const offset = circ * (1 - pct)
  return `${circ} ${offset}`
}

function fmt(sec: number): string {
  const m = Math.floor(sec / 60).toString().padStart(2, '0')
  const s = (sec % 60).toString().padStart(2, '0')
  return `${m}:${s}`
}

function playBell(ctx: AudioContext | null): void {
  if (!ctx) return
  const osc = ctx.createOscillator()
  const gain = ctx.createGain()
  osc.connect(gain)
  gain.connect(ctx.destination)
  osc.type = 'sine'
  osc.frequency.setValueAtTime(528, ctx.currentTime)
  osc.frequency.exponentialRampToValueAtTime(440, ctx.currentTime + 1.2)
  gain.gain.setValueAtTime(0.6, ctx.currentTime)
  gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 2.5)
  osc.start(ctx.currentTime)
  osc.stop(ctx.currentTime + 2.5)
}

interface Props {
  stages?: MeditationStage[]
}

export default function MeditationTimer({ stages = DEFAULT_STAGES }: Props) {
  const [running,    setRunning]    = useState(false)
  const [stageIdx,   setStageIdx]   = useState(0)
  const [elapsed,    setElapsed]    = useState(0)   // secs elapsed in current stage
  const [totalElapsed, setTotalElapsed] = useState(0)
  const [countUp,    setCountUp]    = useState(false)
  const [done,       setDone]       = useState(false)
  const [customStages, setCustomStages] = useState<MeditationStage[]>(stages)
  const [editMode,   setEditMode]   = useState(false)

  const audioCtxRef = useRef<AudioContext | null>(null)
  const tickRef     = useRef<ReturnType<typeof setInterval> | null>(null)

  const totalSec = customStages.reduce((s, st) => s + st.durationSec, 0)
  const currentStage = customStages[stageIdx] ?? customStages[0]
  const stageSec = currentStage?.durationSec ?? 60
  const pct = countUp
    ? totalElapsed / totalSec
    : (stageSec - elapsed) / stageSec

  const stop = useCallback(() => {
    if (tickRef.current) clearInterval(tickRef.current)
    setRunning(false)
  }, [])

  const reset = useCallback(() => {
    stop()
    setStageIdx(0)
    setElapsed(0)
    setTotalElapsed(0)
    setDone(false)
  }, [stop])

  useEffect(() => {
    if (!running) return
    tickRef.current = setInterval(() => {
      setElapsed(e => {
        const next = e + 1
        setTotalElapsed(t => t + 1)
        if (next >= stageSec) {
          // advance stage
          setStageIdx(si => {
            const nextSi = si + 1
            if (nextSi >= customStages.length) {
              stop()
              setDone(true)
              playBell(audioCtxRef.current)
              return si
            }
            playBell(audioCtxRef.current)
            return nextSi
          })
          return 0
        }
        return next
      })
    }, 1000)
    return () => { if (tickRef.current) clearInterval(tickRef.current) }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [running, stageSec, customStages.length])

  function handleStart() {
    if (!audioCtxRef.current) {
      audioCtxRef.current = new AudioContext()
    }
    if (done) reset()
    setRunning(true)
  }

  const R = 56, CX = 70, CY = 70
  const circ = 2 * Math.PI * R
  const dasharray = ringPath(CX, CY, R, Math.max(0, Math.min(1, pct)))

  const remainSec = countUp
    ? totalSec - totalElapsed
    : stageSec - elapsed

  return (
    <div style={{ color: '#cdd9e5' }}>
      {/* SVG progress ring */}
      <div style={{ display: 'flex', justifyContent: 'center', marginBottom: 12 }}>
        <svg width={140} height={140} viewBox="0 0 140 140">
          <circle cx={CX} cy={CY} r={R} fill="none" stroke="#30363d" strokeWidth={10} />
          <circle
            cx={CX} cy={CY} r={R}
            fill="none"
            stroke={currentStage?.color ?? '#58a6ff'}
            strokeWidth={10}
            strokeDasharray={`${circ} ${circ}`}
            strokeDashoffset={circ - circ * Math.max(0, Math.min(1, pct))}
            strokeLinecap="round"
            style={{ transition: 'stroke-dashoffset 1s linear', transformOrigin: 'center', transform: 'rotate(-90deg)' }}
          />
          <text x={CX} y={CY - 8} textAnchor="middle" fill="#cdd9e5" fontSize={20} fontWeight={700}>
            {fmt(Math.max(0, remainSec))}
          </text>
          <text x={CX} y={CY + 14} textAnchor="middle" fill="#8b949e" fontSize={10}>
            {done ? 'Complete' : currentStage?.label ?? ''}
          </text>
        </svg>
      </div>

      {/* Stage progress dots */}
      <div style={{ display: 'flex', justifyContent: 'center', gap: 6, marginBottom: 12 }}>
        {customStages.map((st, i) => (
          <div key={i} style={{
            width: 8, height: 8, borderRadius: '50%',
            background: i < stageIdx ? '#3fb950' : i === stageIdx ? (st.color ?? '#58a6ff') : '#30363d',
            transition: 'background 0.3s',
          }} title={st.label} />
        ))}
      </div>

      {/* Controls */}
      <div style={{ display: 'flex', gap: 8, justifyContent: 'center', flexWrap: 'wrap' }}>
        {!running && !done && (
          <button onClick={handleStart} style={btnStyle('#238636')}>
            ▶ Start
          </button>
        )}
        {running && (
          <button onClick={stop} style={btnStyle('#b08800')}>
            ⏸ Pause
          </button>
        )}
        <button onClick={reset} style={btnStyle('#30363d')}>
          ↺ Reset
        </button>
        <button
          onClick={() => setCountUp(v => !v)}
          style={btnStyle('#21262d', countUp ? '#58a6ff' : '#8b949e')}
        >
          {countUp ? '⏱ Total' : '⏱ Stage'}
        </button>
        <button onClick={() => setEditMode(v => !v)} style={btnStyle('#21262d')}>⚙ Stages</button>
      </div>

      {/* Stage editor */}
      {editMode && (
        <div style={{ marginTop: 12, fontSize: 12 }}>
          {customStages.map((st, i) => (
            <div key={i} style={{ display: 'flex', gap: 6, alignItems: 'center', marginBottom: 6 }}>
              <input
                value={st.label}
                onChange={e => setCustomStages(prev => prev.map((s, j) => j === i ? { ...s, label: e.target.value } : s))}
                style={inputStyle}
                placeholder="Stage name"
              />
              <input
                type="number" min={10} max={3600}
                value={st.durationSec}
                onChange={e => setCustomStages(prev => prev.map((s, j) => j === i ? { ...s, durationSec: Number(e.target.value) } : s))}
                style={{ ...inputStyle, width: 60 }}
              />
              <span style={{ color: '#8b949e' }}>s</span>
            </div>
          ))}
          <button
            onClick={() => setCustomStages(prev => [...prev, { label: 'Stage', durationSec: 120 }])}
            style={btnStyle('#21262d')}
          >+ Add Stage</button>
        </div>
      )}

      {done && (
        <p style={{ textAlign: 'center', color: '#3fb950', marginTop: 10, fontSize: 13, fontWeight: 600 }}>
          🔔 Session complete — {fmt(totalSec)} total
        </p>
      )}
    </div>
  )
}

function btnStyle(bg: string, color = '#cdd9e5'): React.CSSProperties {
  return {
    padding: '5px 12px', fontSize: 12, fontWeight: 600,
    background: bg, color,
    border: '1px solid #30363d', borderRadius: 6,
    cursor: 'pointer',
  }
}
const inputStyle: React.CSSProperties = {
  background: '#0d1117', border: '1px solid #30363d', borderRadius: 4,
  color: '#cdd9e5', padding: '3px 7px', fontSize: 12, flex: 1,
}
