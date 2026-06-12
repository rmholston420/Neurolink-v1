/**
 * SessionHistoryPanel
 *
 * Displays session history accumulated by useSessionHistory:
 *   - 7-day focus sparkline
 *   - GitHub-style practice heatmap (52 weeks × 7 days)
 *   - Scrollable session table (last 30 sessions)
 *   - CSV export button
 */
import React, { useState } from 'react'
import type { SessionHistoryReturn } from '../hooks/useSessionHistory'

interface Props {
  history: SessionHistoryReturn
}

function heatColour(count: number): string {
  if (count === 0) return '#161b22'
  if (count === 1) return 'rgba(46,160,67,0.3)'
  if (count === 2) return 'rgba(46,160,67,0.55)'
  if (count === 3) return 'rgba(46,160,67,0.75)'
  return '#3fb950'
}

function sparkColour(v: number): string {
  if (v >= 0.7) return '#3fb950'
  if (v >= 0.4) return '#e3b341'
  if (v > 0)   return '#f85149'
  return '#30363d'
}

function fmtDuration(s: number): string {
  if (s < 60) return `${Math.round(s)}s`
  const m = Math.floor(s / 60)
  const sec = Math.round(s % 60)
  return `${m}m ${sec < 10 ? '0' : ''}${sec}s`
}

const DAY_LABELS = ['S', 'M', 'T', 'W', 'T', 'F', 'S']

const st: Record<string, React.CSSProperties> = {
  root: { display: 'flex', flexDirection: 'column', gap: 20 },
  sectionTitle: {
    fontSize: 11, fontWeight: 700, color: '#8b949e',
    textTransform: 'uppercase' as const, letterSpacing: 0.8, marginBottom: 8,
  },
  sparkRow: { display: 'flex', alignItems: 'flex-end', gap: 3, height: 40 },
  sparkBar: (v: number, maxV: number): React.CSSProperties => ({
    flex: 1,
    height: `${maxV > 0 ? Math.max(4, (v / maxV) * 100) : 4}%`,
    background: sparkColour(v),
    borderRadius: '2px 2px 0 0',
    transition: 'height 400ms ease',
    minHeight: 4,
  }),
  sparkLabels: {
    display: 'flex', justifyContent: 'space-between',
    fontSize: 10, color: '#484f58', marginTop: 4,
  },
  heatmap: {
    display: 'flex', gap: 2, overflowX: 'auto' as const,
  },
  heatWeek: { display: 'flex', flexDirection: 'column', gap: 2 },
  heatCell: (count: number): React.CSSProperties => ({
    width: 11, height: 11, borderRadius: 2,
    background: heatColour(count),
    transition: 'background 300ms',
    cursor: count > 0 ? 'pointer' : 'default',
  }),
  dayLabels: {
    display: 'flex', flexDirection: 'column', gap: 2, marginRight: 4,
  },
  dayLabel: { fontSize: 8, color: '#484f58', height: 11, lineHeight: '11px' },
  table: { width: '100%', borderCollapse: 'collapse' as const, fontSize: 12 },
  th: {
    padding: '5px 8px', textAlign: 'left' as const,
    fontSize: 10, fontWeight: 700, color: '#484f58',
    textTransform: 'uppercase' as const, letterSpacing: 0.6,
    borderBottom: '1px solid #21262d',
  },
  td: {
    padding: '6px 8px', color: '#8b949e',
    borderBottom: '1px solid #161b22',
    fontVariantNumeric: 'tabular-nums',
    whiteSpace: 'nowrap' as const,
  },
  tdHighlight: { color: '#e6edf3', fontWeight: 600 },
  tableWrap: { maxHeight: 260, overflowY: 'auto' as const },
  btnRow: { display: 'flex', gap: 8 },
  btn: {
    padding: '6px 14px', borderRadius: 6, fontSize: 12, fontWeight: 600,
    cursor: 'pointer', border: '1px solid #30363d',
    background: 'rgba(139,148,158,0.1)', color: '#8b949e',
    transition: 'all 150ms ease',
  },
  noData: { fontSize: 12, color: '#484f58', fontStyle: 'italic', padding: '8px 0' },
}

export default function SessionHistoryPanel({ history }: Props) {
  const { sessions, sparkline7d, heatmap, flushSession, exportCSV } = history
  const [showHeatmap, setShowHeatmap] = useState(false)

  const maxSparkVal = Math.max(...sparkline7d, 0.01)
  const dayLabelsForSpark = Array.from({ length: 7 }, (_, i) => {
    const d = new Date(); d.setDate(d.getDate() - (6 - i))
    return d.toLocaleDateString([], { weekday: 'short' }).slice(0, 2)
  })

  return (
    <div style={st.root}>
      {/* 7-day sparkline */}
      <div>
        <div style={st.sectionTitle}>7-Day Focus Trend</div>
        <div style={st.sparkRow}>
          {sparkline7d.map((v, i) => (
            <div
              key={i}
              style={st.sparkBar(v, maxSparkVal)}
              title={`${dayLabelsForSpark[i]}: focus ${(v * 100).toFixed(0)}%`}
            />
          ))}
        </div>
        <div style={st.sparkLabels}>
          {dayLabelsForSpark.map((d, i) => <span key={i}>{d}</span>)}
        </div>
      </div>

      {/* Practice heatmap (toggle) */}
      <div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
          <span style={st.sectionTitle}>Practice Calendar</span>
          <button style={{ ...st.btn, padding: '2px 8px', fontSize: 10 }} onClick={() => setShowHeatmap(h => !h)}>
            {showHeatmap ? 'Hide' : 'Show'}
          </button>
        </div>
        {showHeatmap && (
          <div style={{ display: 'flex', alignItems: 'flex-start' }}>
            <div style={st.dayLabels}>
              {DAY_LABELS.map((d, i) => <span key={i} style={st.dayLabel}>{d}</span>)}
            </div>
            <div style={st.heatmap}>
              {heatmap.map((week, w) => (
                <div key={w} style={st.heatWeek}>
                  {week.map((count, d) => (
                    <div
                      key={d}
                      style={st.heatCell(count)}
                      title={count > 0 ? `${count} session${count > 1 ? 's' : ''}` : 'No session'}
                    />
                  ))}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Session table */}
      <div>
        <div style={st.sectionTitle}>Recent Sessions ({sessions.length})</div>
        {sessions.length === 0 ? (
          <p style={st.noData}>No sessions recorded yet. Connect your headband to start tracking.</p>
        ) : (
          <div style={st.tableWrap}>
            <table style={st.table}>
              <thead>
                <tr>
                  <th style={st.th}>Date</th>
                  <th style={st.th}>Duration</th>
                  <th style={st.th}>Focus</th>
                  <th style={st.th}>α Mean</th>
                  <th style={st.th}>EA-1%</th>
                  <th style={st.th}>Stage</th>
                </tr>
              </thead>
              <tbody>
                {sessions.slice(0, 30).map(s => (
                  <tr key={s.id}>
                    <td style={{ ...st.td, ...st.tdHighlight }}>{s.date}</td>
                    <td style={st.td}>{fmtDuration(s.durationS)}</td>
                    <td style={{ ...st.td, color: s.meanFocus >= 0.7 ? '#3fb950' : s.meanFocus >= 0.4 ? '#e3b341' : '#f85149' }}>
                      {(s.meanFocus * 100).toFixed(0)}%
                    </td>
                    <td style={st.td}>{s.meanAlpha.toFixed(4)}</td>
                    <td style={{ ...st.td, color: s.ea1EligiblePct >= 30 ? '#3fb950' : '#8b949e' }}>
                      {s.ea1EligiblePct.toFixed(0)}%
                    </td>
                    <td style={{ ...st.td, color: '#58a6ff' }}>{s.alchemicalStage}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Actions */}
      <div style={st.btnRow}>
        <button style={st.btn} onClick={flushSession} title="Save current session now">
          💾 Save Session
        </button>
        <button
          style={{ ...st.btn, opacity: sessions.length === 0 ? 0.4 : 1 }}
          onClick={exportCSV}
          disabled={sessions.length === 0}
          title="Export all sessions as CSV"
        >
          ↓ Export CSV
        </button>
      </div>
    </div>
  )
}
