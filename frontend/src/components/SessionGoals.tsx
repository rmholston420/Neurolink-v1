/**
 * SessionGoals — Tier 2
 * Pre-session goal setter, live dual progress bars, achievement badge shelf.
 */
import React, { useState } from 'react'
import type { SessionGoalsResult } from '../hooks/useSessionGoals'

interface Props {
  goals: SessionGoalsResult
}

function fmt(sec: number): string {
  const m = Math.floor(sec / 60).toString().padStart(2, '0')
  const s = (sec % 60).toString().padStart(2, '0')
  return `${m}:${s}`
}

function ProgressBar({ label, pct, color }: { label: string; pct: number; color: string }) {
  return (
    <div style={{ marginBottom: 10 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 4 }}>
        <span style={{ color: '#8b949e' }}>{label}</span>
        <span style={{ color, fontWeight: 700 }}>{(pct * 100).toFixed(0)}%</span>
      </div>
      <div style={{ height: 8, background: '#21262d', borderRadius: 4, overflow: 'hidden' }}>
        <div style={{
          height: '100%', width: `${Math.min(100, pct * 100)}%`,
          background: pct >= 1 ? '#3fb950' : color,
          transition: 'width 0.6s ease', borderRadius: 4,
        }} />
      </div>
    </div>
  )
}

export default function SessionGoals({ goals }: Props) {
  const {
    goal, setGoal, clearGoal,
    elapsedSec, ea1Pct,
    durationProgress, ea1Progress,
    badges, isRunning, startSession, endSession,
  } = goals

  const [draftMin, setDraftMin] = useState(20)
  const [draftEA1, setDraftEA1] = useState(30)

  return (
    <div style={{ color: '#cdd9e5' }}>
      {!goal && (
        <div>
          <div style={{ display: 'flex', gap: 10, marginBottom: 10, flexWrap: 'wrap' }}>
            <div style={{ flex: 1 }}>
              <label style={{ fontSize: 11, color: '#8b949e', display: 'block', marginBottom: 3 }}>
                Target Duration (min)
              </label>
              <input
                type="number" min={1} max={120} value={draftMin}
                onChange={e => setDraftMin(Number(e.target.value))}
                style={inputStyle}
              />
            </div>
            <div style={{ flex: 1 }}>
              <label style={{ fontSize: 11, color: '#8b949e', display: 'block', marginBottom: 3 }}>
                EA-1 Target (%)
              </label>
              <input
                type="number" min={0} max={100} value={draftEA1}
                onChange={e => setDraftEA1(Number(e.target.value))}
                style={inputStyle}
              />
            </div>
          </div>
          <button
            onClick={() => { setGoal({ targetMinutes: draftMin, targetEA1Pct: draftEA1 }); startSession() }}
            style={btnStyle('#238636')}
          >▶ Start Session</button>
        </div>
      )}

      {goal && (
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 10, fontSize: 12 }}>
            <span style={{ color: '#8b949e' }}>
              {isRunning ? '🔴 Live' : '⏹ Ended'} · {fmt(elapsedSec)}
            </span>
            <span style={{ color: '#8b949e' }}>
              Goal: {goal.targetMinutes} min · {goal.targetEA1Pct}% EA-1
            </span>
          </div>

          <ProgressBar label="Duration" pct={durationProgress} color="#58a6ff" />
          <ProgressBar label={`EA-1 Time (${ea1Pct.toFixed(0)}% live)`} pct={ea1Progress} color="#a371f7" />

          <div style={{ display: 'flex', gap: 8, marginTop: 10 }}>
            {isRunning && (
              <button onClick={endSession} style={btnStyle('#b08800')}>⏸ End Session</button>
            )}
            <button onClick={clearGoal} style={btnStyle('#30363d')}>✕ Clear Goal</button>
          </div>
        </div>
      )}

      {/* Badge shelf */}
      {badges.length > 0 && (
        <div style={{ marginTop: 14 }}>
          <div style={{ fontSize: 11, color: '#8b949e', textTransform: 'uppercase',
            letterSpacing: 1, marginBottom: 8 }}>Achievements</div>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            {badges.map(b => (
              <div key={b.id} title={b.detail} style={{
                display: 'flex', flexDirection: 'column', alignItems: 'center',
                background: '#21262d', border: '1px solid #30363d',
                borderRadius: 8, padding: '6px 10px', fontSize: 11, gap: 3,
                cursor: 'default',
              }}>
                <span style={{ fontSize: 20 }}>{b.emoji}</span>
                <span style={{ color: '#cdd9e5', fontWeight: 600 }}>{b.label}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

function btnStyle(bg: string): React.CSSProperties {
  return {
    padding: '5px 12px', fontSize: 12, fontWeight: 600,
    background: bg, color: '#fff',
    border: '1px solid #30363d', borderRadius: 6, cursor: 'pointer',
  }
}
const inputStyle: React.CSSProperties = {
  width: '100%', background: '#0d1117', border: '1px solid #30363d',
  borderRadius: 5, color: '#cdd9e5', padding: '5px 8px',
  fontSize: 13, boxSizing: 'border-box' as const,
}
