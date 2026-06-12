/**
 * MeditationTimer
 *
 * Visual interface for useMeditationTimer.
 *
 * Layout:
 *   - SVG progress ring (stage progress inner, total progress outer)
 *   - Stage name + countdown in centre
 *   - Stage pills row (completed / active / upcoming)
 *   - Start / Pause / Resume / Reset controls
 *   - Stage editor: click ✎ to rename or change duration
 */
import React, { useState } from 'react'
import type { MeditationTimerReturn, TimerStage } from '../hooks/useMeditationTimer'

interface Props {
  timer: MeditationTimerReturn
}

function fmt(s: number): string {
  const m = Math.floor(s / 60)
  const sec = Math.floor(s % 60)
  return `${m}:${sec < 10 ? '0' : ''}${sec}`
}

const STAGE_COLOURS = ['#58a6ff', '#3fb950', '#e3b341', '#f0883e', '#bc8cff']

const C = 220  // SVG viewBox size
const cx = C / 2, cy = C / 2
const R_OUTER = 90, R_INNER = 74
const CIRC_OUTER = 2 * Math.PI * R_OUTER
const CIRC_INNER = 2 * Math.PI * R_INNER

function arc(r: number, circ: number, progress: number) {
  const dash   = Math.max(0, progress * circ)
  const gap    = circ - dash
  return `${dash} ${gap}`
}

const st: Record<string, React.CSSProperties> = {
  root:    { display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 16 },
  ringWrap:{ position: 'relative', width: C, height: C, maxWidth: '100%' },
  centreText: {
    position: 'absolute', top: '50%', left: '50%',
    transform: 'translate(-50%,-50%)',
    textAlign: 'center', pointerEvents: 'none',
  },
  stageName: { fontSize: 14, fontWeight: 700, color: '#8b949e', textTransform: 'uppercase', letterSpacing: 1 },
  countdown: { fontSize: 34, fontWeight: 700, color: '#e6edf3', fontVariantNumeric: 'tabular-nums', lineHeight: 1.1 },
  totalTime: { fontSize: 11, color: '#484f58' },
  stagePills: { display: 'flex', gap: 6, flexWrap: 'wrap', justifyContent: 'center' },
  pill: (active: boolean, done: boolean, colour: string): React.CSSProperties => ({
    padding: '3px 10px', borderRadius: 20, fontSize: 11, fontWeight: 700,
    border: `1px solid ${active ? colour : done ? 'transparent' : '#30363d'}`,
    background: active ? `${colour}22` : done ? '#21262d' : 'transparent',
    color: active ? colour : done ? '#484f58' : '#8b949e',
    textDecoration: done ? 'line-through' : 'none',
  }),
  controls: { display: 'flex', gap: 8 },
  btn: (variant: 'primary'|'ghost'): React.CSSProperties => ({
    padding: '7px 18px', borderRadius: 8, fontSize: 13, fontWeight: 700,
    cursor: 'pointer',
    border: variant === 'primary' ? 'none' : '1px solid #30363d',
    background: variant === 'primary' ? '#388bfd' : 'rgba(139,148,158,0.1)',
    color: variant === 'primary' ? '#fff' : '#8b949e',
    transition: 'all 150ms ease',
  }),
  editorToggle: {
    fontSize: 11, color: '#484f58', cursor: 'pointer',
    textDecoration: 'underline', textUnderlineOffset: 3,
    background: 'none', border: 'none', padding: 0,
  },
  editor: {
    display: 'flex', flexDirection: 'column', gap: 6, width: '100%',
    padding: '10px 12px',
    background: 'rgba(22,27,34,0.8)',
    border: '1px solid #30363d', borderRadius: 8,
  },
  editorRow: { display: 'flex', gap: 6, alignItems: 'center' },
  input: {
    background: '#0d1117', border: '1px solid #30363d',
    borderRadius: 5, padding: '4px 8px', color: '#e6edf3',
    fontSize: 12, outline: 'none',
  },
  completeBadge: {
    display: 'flex', alignItems: 'center', gap: 8,
    padding: '10px 18px', borderRadius: 10,
    background: 'rgba(63,185,80,0.12)', border: '1px solid #238636',
    color: '#3fb950', fontSize: 14, fontWeight: 700,
  },
}

export default function MeditationTimer({ timer }: Props) {
  const [showEditor, setShowEditor] = useState(false)
  const [draftStages, setDraftStages] = useState<TimerStage[]>(timer.stages)

  const colour = STAGE_COLOURS[timer.stageIndex % STAGE_COLOURS.length]

  const applyStages = () => {
    timer.setStages(draftStages)
    timer.reset()
    setShowEditor(false)
  }

  const updateDraft = (i: number, field: 'name' | 'durationS', val: string) => {
    setDraftStages(prev => prev.map((s, idx) =>
      idx === i
        ? { ...s, [field]: field === 'durationS' ? Math.max(30, parseInt(val) || 60) * 60 : val }
        : s
    ))
  }

  return (
    <div style={st.root}>
      {/* Progress rings */}
      <div style={st.ringWrap}>
        <svg width={C} height={C} viewBox={`0 0 ${C} ${C}`} style={{ width: '100%', height: 'auto' }}>
          {/* Track rings */}
          <circle cx={cx} cy={cy} r={R_OUTER} fill="none" stroke="#21262d" strokeWidth={8} />
          <circle cx={cx} cy={cy} r={R_INNER} fill="none" stroke="#161b22" strokeWidth={14} />

          {/* Outer ring = total progress */}
          <circle
            cx={cx} cy={cy} r={R_OUTER} fill="none"
            stroke="#388bfd" strokeWidth={8}
            strokeDasharray={arc(R_OUTER, CIRC_OUTER, timer.totalProgress)}
            strokeLinecap="round"
            transform={`rotate(-90 ${cx} ${cy})`}
            style={{ transition: 'stroke-dasharray 0.5s ease' }}
          />

          {/* Inner ring = stage progress */}
          <circle
            cx={cx} cy={cy} r={R_INNER} fill="none"
            stroke={colour} strokeWidth={14}
            strokeDasharray={arc(R_INNER, CIRC_INNER, timer.stageProgress)}
            strokeLinecap="round"
            transform={`rotate(-90 ${cx} ${cy})`}
            style={{ transition: 'stroke-dasharray 0.5s ease' }}
          />
        </svg>

        <div style={st.centreText}>
          {timer.status === 'complete' ? (
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: 28 }}>🔔</div>
              <div style={{ fontSize: 13, color: '#3fb950', fontWeight: 700 }}>Complete</div>
            </div>
          ) : (
            <>
              <div style={st.stageName}>{timer.currentStage.name}</div>
              <div style={st.countdown}>{fmt(timer.stageRemaining)}</div>
              <div style={st.totalTime}>{fmt(timer.totalElapsed)} / {fmt(timer.totalDuration)}</div>
            </>
          )}
        </div>
      </div>

      {/* Stage pills */}
      <div style={st.stagePills}>
        {timer.stages.map((s, i) => (
          <span key={i} style={st.pill(
            i === timer.stageIndex && timer.status !== 'complete',
            i < timer.stageIndex || timer.status === 'complete',
            STAGE_COLOURS[i % STAGE_COLOURS.length]
          )}>
            {s.name} · {fmt(s.durationS)}
          </span>
        ))}
      </div>

      {/* Controls */}
      {timer.status === 'complete' ? (
        <div style={st.completeBadge}>
          🏆 Session complete — {fmt(timer.totalDuration)}
          <button style={st.btn('ghost')} onClick={timer.reset}>New Session</button>
        </div>
      ) : (
        <div style={st.controls}>
          {timer.status === 'idle'    && <button style={st.btn('primary')} onClick={timer.start}>▶ Start</button>}
          {timer.status === 'running' && <button style={st.btn('ghost')}   onClick={timer.pause}>⏸ Pause</button>}
          {timer.status === 'paused'  && <button style={st.btn('primary')} onClick={timer.resume}>▶ Resume</button>}
          {timer.status !== 'idle'    && <button style={st.btn('ghost')}   onClick={timer.reset}>↺ Reset</button>}
        </div>
      )}

      {/* Stage editor */}
      {timer.status === 'idle' && (
        <button style={st.editorToggle} onClick={() => setShowEditor(v => !v)}>
          ✎ {showEditor ? 'Hide stage editor' : 'Edit stages'}
        </button>
      )}

      {showEditor && timer.status === 'idle' && (
        <div style={st.editor}>
          {draftStages.map((s, i) => (
            <div key={i} style={st.editorRow}>
              <span style={{ fontSize: 11, color: '#484f58', minWidth: 16 }}>{i + 1}.</span>
              <input
                style={{ ...st.input, flex: 1 }}
                value={s.name}
                onChange={e => updateDraft(i, 'name', e.target.value)}
                placeholder="Stage name"
              />
              <input
                style={{ ...st.input, width: 52 }}
                type="number" min={1} max={60}
                value={Math.round(s.durationS / 60)}
                onChange={e => updateDraft(i, 'durationS', e.target.value)}
              />
              <span style={{ fontSize: 11, color: '#484f58' }}>min</span>
            </div>
          ))}
          <button style={{ ...st.btn('primary'), alignSelf: 'flex-start', marginTop: 4 }} onClick={applyStages}>
            Apply
          </button>
        </div>
      )}
    </div>
  )
}
