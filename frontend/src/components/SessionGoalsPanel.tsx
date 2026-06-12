/**
 * SessionGoalsPanel
 *
 * Displays session goals with live progress bars and an achievement shelf.
 *
 * Layout:
 *   - Three goal rows: Duration, EA-1%, Focus% — each with an editable target
 *     and an animated progress bar
 *   - Achievement shelf: 6 badges, greyed out until earned, with tooltip
 *   - Achievement toast: slides in at bottom-right when a badge is earned
 */
import React, { useEffect, useState } from 'react'
import type { SessionGoalsReturn, Achievement } from '../hooks/useSessionGoals'

interface Props {
  goals: SessionGoalsReturn
}

function pctBar(pct: number, colour: string): React.CSSProperties {
  return {
    height: '100%', borderRadius: 4,
    width: `${Math.min(100, pct * 100)}%`,
    background: colour,
    transition: 'width 600ms ease',
    minWidth: pct > 0 ? 4 : 0,
  }
}

function goalColour(pct: number): string {
  if (pct >= 1)   return '#3fb950'
  if (pct >= 0.5) return '#e3b341'
  return '#388bfd'
}

const st: Record<string, React.CSSProperties> = {
  root:  { display: 'flex', flexDirection: 'column', gap: 16 },
  goalsSection: { display: 'flex', flexDirection: 'column', gap: 10 },
  sectionTitle: {
    fontSize: 11, fontWeight: 700, color: '#8b949e',
    textTransform: 'uppercase', letterSpacing: 0.8, marginBottom: 4,
  },
  goalRow: {
    display: 'flex', alignItems: 'center', gap: 10,
  },
  goalLabel: { fontSize: 12, color: '#8b949e', minWidth: 70 },
  barTrack: {
    flex: 1, height: 8, borderRadius: 4,
    background: '#21262d', overflow: 'hidden',
  },
  goalVal: { fontSize: 12, color: '#e6edf3', minWidth: 44, textAlign: 'right', fontVariantNumeric: 'tabular-nums' },
  targetInput: {
    width: 44, background: '#0d1117', border: '1px solid #30363d',
    borderRadius: 4, padding: '2px 6px',
    color: '#8b949e', fontSize: 11, textAlign: 'center', outline: 'none',
  },
  targetLabel: { fontSize: 10, color: '#484f58', minWidth: 20 },
  allMetBadge: {
    display: 'flex', alignItems: 'center', gap: 8,
    padding: '8px 14px', borderRadius: 8,
    background: 'rgba(63,185,80,0.1)', border: '1px solid #238636',
    color: '#3fb950', fontSize: 13, fontWeight: 700,
  },
  achieveGrid: {
    display: 'flex', gap: 8, flexWrap: 'wrap',
  },
  badge: (earned: boolean): React.CSSProperties => ({
    display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 3,
    padding: '8px 10px', borderRadius: 8, minWidth: 72, cursor: 'default',
    border: `1px solid ${earned ? '#30363d' : '#21262d'}`,
    background: earned ? 'rgba(22,27,34,0.8)' : 'rgba(13,17,23,0.4)',
    opacity: earned ? 1 : 0.35,
    transition: 'opacity 400ms, background 400ms',
    position: 'relative',
  }),
  badgeIcon: (earned: boolean): React.CSSProperties => ({
    fontSize: 22,
    filter: earned ? 'none' : 'grayscale(1)',
    transition: 'filter 400ms',
  }),
  badgeLabel: { fontSize: 9, fontWeight: 700, color: '#8b949e', textAlign: 'center', textTransform: 'uppercase', letterSpacing: 0.5 },
  badgeDate:  { fontSize: 9, color: '#484f58' },
  toast: {
    position: 'fixed', bottom: 24, right: 24, zIndex: 9999,
    display: 'flex', alignItems: 'center', gap: 12,
    padding: '12px 18px', borderRadius: 12,
    background: '#161b22', border: '1px solid #238636',
    boxShadow: '0 8px 32px rgba(0,0,0,0.4)',
    color: '#e6edf3', fontSize: 13, fontWeight: 600,
    animation: 'nlSlideIn 0.4s cubic-bezier(0.16,1,0.3,1)',
    maxWidth: 320,
  },
  toastIcon: { fontSize: 28, lineHeight: 1 },
  toastBody: { display: 'flex', flexDirection: 'column', gap: 2 },
  toastTitle: { color: '#3fb950', fontSize: 11, fontWeight: 700, textTransform: 'uppercase', letterSpacing: 0.8 },
  toastClose: {
    marginLeft: 'auto', cursor: 'pointer',
    background: 'none', border: 'none', color: '#484f58', fontSize: 16, padding: 0,
  },
}

function BadgeTile({ a }: { a: Achievement }) {
  const earned = a.earnedAt !== null
  const dateStr = earned
    ? new Date(a.earnedAt!).toLocaleDateString([], { month: 'short', day: 'numeric' })
    : ''
  return (
    <div style={st.badge(earned)} title={a.description}>
      <span style={st.badgeIcon(earned)}>{a.icon}</span>
      <span style={st.badgeLabel}>{a.label}</span>
      {earned && <span style={st.badgeDate}>{dateStr}</span>}
    </div>
  )
}

function AchievementToast({ a, onClose }: { a: Achievement; onClose: () => void }) {
  useEffect(() => {
    const t = setTimeout(onClose, 5000)
    return () => clearTimeout(t)
  }, [onClose])

  return (
    <div style={st.toast}>
      <span style={st.toastIcon}>{a.icon}</span>
      <div style={st.toastBody}>
        <span style={st.toastTitle}>Achievement unlocked</span>
        <span>{a.label}</span>
        <span style={{ fontSize: 11, color: '#8b949e' }}>{a.description}</span>
      </div>
      <button style={st.toastClose} onClick={onClose}>✕</button>
    </div>
  )
}

export default function SessionGoalsPanel({ goals: g }: Props) {
  const { goals, progress, achievements, newAchievement, clearToast } = g

  const goalRows = [
    {
      label: 'Duration',
      current: `${progress.durationMin.toFixed(1)} min`,
      pct: progress.durationPct,
      target: goals.durationMinTarget,
      unit: 'min',
      setTarget: (v: number) => g.setGoals({ durationMinTarget: v || null }),
    },
    {
      label: 'EA-1 %',
      current: `${progress.ea1Pct.toFixed(0)}%`,
      pct: progress.ea1ProgressPct,
      target: goals.ea1PctTarget,
      unit: '%',
      setTarget: (v: number) => g.setGoals({ ea1PctTarget: v || null }),
    },
    {
      label: 'Focus %',
      current: `${progress.focusPct.toFixed(0)}%`,
      pct: progress.focusProgressPct,
      target: goals.focusPctTarget,
      unit: '%',
      setTarget: (v: number) => g.setGoals({ focusPctTarget: v || null }),
    },
  ]

  return (
    <div style={st.root}>
      {/* Goals */}
      <div style={st.goalsSection}>
        <div style={st.sectionTitle}>Session Goals</div>
        {goalRows.map(row => (
          <div key={row.label} style={st.goalRow}>
            <span style={st.goalLabel}>{row.label}</span>
            <div style={st.barTrack}>
              <div style={pctBar(row.pct, goalColour(row.pct))} />
            </div>
            <span style={st.goalVal}>{row.current}</span>
            <input
              type="number" min={0}
              style={st.targetInput}
              value={row.target ?? ''}
              placeholder="—"
              onChange={e => row.setTarget(parseFloat(e.target.value))}
              title={`Target ${row.unit}`}
            />
            <span style={st.targetLabel}>{row.unit}</span>
          </div>
        ))}
      </div>

      {progress.allGoalsMet && (
        <div style={st.allMetBadge}>🏆 All goals met for this session!</div>
      )}

      {/* Achievements shelf */}
      <div>
        <div style={st.sectionTitle}>Achievements</div>
        <div style={st.achieveGrid}>
          {achievements.map(a => <BadgeTile key={a.id} a={a} />)}
        </div>
      </div>

      {/* Toast */}
      {newAchievement && (
        <AchievementToast a={newAchievement} onClose={clearToast} />
      )}
    </div>
  )
}
