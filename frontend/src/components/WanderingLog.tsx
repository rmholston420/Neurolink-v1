/**
 * WanderingLog
 *
 * Displays mind-wandering detection results for the current session:
 *   - Live wandering badge (always-mounted, fades in/out — no layout shift)
 *   - Session stats row (events, mean recovery, longest focus run, EA-1%)
 *   - Horizontal timeline bar (50 buckets coloured by focus level)
 *   - Scrollable event list with timestamps and recovery times
 */
import React from 'react'
import type { WanderingDetectorReturn } from '../hooks/useWanderingDetector'

interface Props {
  detector: WanderingDetectorReturn
}

function focusColour(v: number): string {
  if (v >= 0.7) return '#3fb950'
  if (v >= 0.4) return '#e3b341'
  return '#f85149'
}

function fmtMs(ms: number | null): string {
  if (ms === null) return '—'
  if (ms < 1000) return `${Math.round(ms)} ms`
  return `${(ms / 1000).toFixed(1)} s`
}

function fmtTime(ts: number): string {
  return new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

const s: Record<string, React.CSSProperties> = {
  root:  { display: 'flex', flexDirection: 'column', gap: 14 },
  stats: { display: 'flex', gap: 16, flexWrap: 'wrap' as const },
  statBox: {
    display: 'flex', flexDirection: 'column', gap: 2,
    padding: '8px 12px',
    background: 'rgba(22,27,34,0.6)',
    border: '1px solid #21262d',
    borderRadius: 8, minWidth: 90,
  },
  statLabel: { fontSize: 10, fontWeight: 700, color: '#484f58', textTransform: 'uppercase' as const, letterSpacing: 0.8 },
  statValue: { fontSize: 18, fontWeight: 700, color: '#e6edf3', lineHeight: 1.2 },
  statSub:   { fontSize: 10, color: '#8b949e' },
  timeline: {
    display: 'flex', height: 18, gap: 1, borderRadius: 4, overflow: 'hidden',
  },
  timelineBucket: (v: number): React.CSSProperties => ({
    flex: 1,
    background: v === 0 ? '#21262d' : focusColour(v),
    opacity: 0.3 + v * 0.7,
    transition: 'background 300ms ease, opacity 300ms ease',
  }),
  timelineLabel: { display: 'flex', justifyContent: 'space-between', fontSize: 10, color: '#484f58' },
  eventList: {
    display: 'flex', flexDirection: 'column', gap: 4,
    maxHeight: 160, overflowY: 'auto' as const,
  },
  eventRow: (recent: boolean): React.CSSProperties => ({
    display: 'flex', alignItems: 'center', gap: 10,
    padding: '5px 10px', borderRadius: 6, fontSize: 12,
    background: recent ? 'rgba(248,81,73,0.08)' : 'transparent',
    border: `1px solid ${recent ? 'rgba(248,81,73,0.2)' : '#21262d'}`,
    color: '#8b949e',
  }),
  eventTime:  { color: '#58a6ff', fontVariantNumeric: 'tabular-nums', minWidth: 80 },
  eventRec:   { marginLeft: 'auto', color: '#3fb950', fontVariantNumeric: 'tabular-nums' },
  noEvents:   { fontSize: 12, color: '#484f58', fontStyle: 'italic', padding: '8px 0' },
  // Always-mounted row — height is reserved at all times, badge fades in/out.
  // height: 26px matches badge line-height; no layout shift ever occurs.
  wanderRow: (visible: boolean): React.CSSProperties => ({
    height: 26,
    display: 'flex', alignItems: 'center',
    opacity: visible ? 1 : 0,
    visibility: visible ? 'visible' : 'hidden',
    transition: 'opacity 300ms ease, visibility 300ms ease',
    pointerEvents: visible ? 'auto' : 'none',
  }),
  wanderBadge: {
    display: 'inline-flex', alignItems: 'center', gap: 5,
    padding: '2px 8px', borderRadius: 20, fontSize: 11, fontWeight: 700,
    background: 'rgba(248,81,73,0.15)', border: '1px solid rgba(248,81,73,0.35)',
    color: '#f85149', animation: 'nlPulse 1s ease-in-out infinite',
  },
}

export default function WanderingLog({ detector }: Props) {
  const { events, sessionStats, timeline, isWandering } = detector

  return (
    <div style={s.root}>
      {/* Live wandering indicator — always mounted, fades in/out to avoid layout shift */}
      <div style={s.wanderRow(isWandering)} aria-live="polite" aria-atomic="true">
        <span style={s.wanderBadge}>🧠 Mind wandering detected</span>
      </div>

      {/* Session stats */}
      <div style={s.stats}>
        <div style={s.statBox}>
          <span style={s.statLabel}>Events</span>
          <span style={s.statValue}>{sessionStats.eventCount}</span>
          <span style={s.statSub}>this session</span>
        </div>
        <div style={s.statBox}>
          <span style={s.statLabel}>Avg Recovery</span>
          <span style={s.statValue}>{fmtMs(sessionStats.meanRecoveryMs)}</span>
          <span style={s.statSub}>time to refocus</span>
        </div>
        <div style={s.statBox}>
          <span style={s.statLabel}>Longest Focus</span>
          <span style={s.statValue}>{sessionStats.longestFocusRunS.toFixed(0)} s</span>
          <span style={s.statSub}>uninterrupted</span>
        </div>
        <div style={s.statBox}>
          <span style={s.statLabel}>EA-1 %</span>
          <span style={s.statValue}>{sessionStats.ea1EligiblePct.toFixed(0)}%</span>
          <span style={s.statSub}>of session</span>
        </div>
      </div>

      {/* Focus timeline bar */}
      <div>
        <div style={s.timeline}>
          {timeline.map((v, i) => (
            <div key={i} style={s.timelineBucket(v)} title={`Segment ${i + 1}: focus ${(v * 100).toFixed(0)}%`} />
          ))}
        </div>
        <div style={s.timelineLabel}>
          <span>Session start</span>
          <span style={{ color: '#3fb950' }}>■ focus</span>
          <span style={{ color: '#e3b341' }}>■ fair</span>
          <span style={{ color: '#f85149' }}>■ wander</span>
          <span>Now</span>
        </div>
      </div>

      {/* Event list */}
      <div style={s.eventList}>
        {events.length === 0 ? (
          <p style={s.noEvents}>No wandering events detected yet.</p>
        ) : (
          [...events].reverse().map((ev, i) => (
            <div key={ev.timestamp} style={s.eventRow(i === 0)}>
              <span style={s.eventTime}>{fmtTime(ev.timestamp)}</span>
              <span>engagement {ev.engageValue.toFixed(3)}</span>
              <span style={{ color: '#8b949e' }}>focus {(ev.focusAtEvent * 100).toFixed(0)}%</span>
              <span style={s.eventRec}>↺ {fmtMs(ev.recoveryMs)}</span>
            </div>
          ))
        )}
      </div>
    </div>
  )
}
