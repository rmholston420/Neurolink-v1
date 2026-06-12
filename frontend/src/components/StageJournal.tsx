/**
 * StageJournal
 *
 * Displays the alchemical stage journal.
 *
 * Layout:
 *   - Search/filter input + Export CSV + Add Note button
 *   - Scrollable entry list, each with:
 *     • Stage badge (colour-coded by stage)
 *     • Timestamp, EA-1 indicator, focus %
 *     • Auto-title or manual note text
 *     • Expandable notes thread
 *     • Add note input (inline)
 */
import React, { useState } from 'react'
import type { StageJournalReturn, JournalEntry } from '../hooks/useStageJournal'

interface Props {
  journal: StageJournalReturn
}

const STAGE_COLOURS: Record<string, string> = {
  Nigredo:    '#8b949e',
  Albedo:     '#c9d1d9',
  Citrinitas: '#e3b341',
  Rubedo:     '#f0883e',
  Unknown:    '#484f58',
}

function stageColour(stage: string): string {
  return STAGE_COLOURS[stage] ?? '#bc8cff'
}

function fmtTs(ts: number): string {
  return new Date(ts).toLocaleString([], {
    month: 'short', day: 'numeric',
    hour: '2-digit', minute: '2-digit',
  })
}

const st: Record<string, React.CSSProperties> = {
  root:    { display: 'flex', flexDirection: 'column', gap: 12 },
  toolbar: { display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' },
  searchInput: {
    flex: 1, minWidth: 120,
    background: '#0d1117', border: '1px solid #30363d',
    borderRadius: 6, padding: '5px 10px',
    color: '#e6edf3', fontSize: 12, outline: 'none',
  },
  toolBtn: {
    padding: '5px 12px', borderRadius: 6, fontSize: 12, fontWeight: 600,
    cursor: 'pointer', border: '1px solid #30363d',
    background: 'rgba(139,148,158,0.1)', color: '#8b949e',
    transition: 'all 150ms',
  },
  entries: { display: 'flex', flexDirection: 'column', gap: 8, maxHeight: 420, overflowY: 'auto' },
  entry:   {
    padding: '10px 12px', borderRadius: 8,
    background: 'rgba(22,27,34,0.7)',
    border: '1px solid #21262d',
  },
  entryHeader: { display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap', marginBottom: 4 },
  stageBadge: (stage: string): React.CSSProperties => ({
    padding: '1px 8px', borderRadius: 20, fontSize: 10, fontWeight: 700,
    border: `1px solid ${stageColour(stage)}44`,
    background: `${stageColour(stage)}18`,
    color: stageColour(stage),
  }),
  ts:   { fontSize: 11, color: '#484f58' },
  ea1ok: { fontSize: 10, color: '#3fb950', fontWeight: 700 },
  focusVal: { fontSize: 11, color: '#8b949e', marginLeft: 'auto' },
  title: { fontSize: 13, color: '#e6edf3', fontWeight: 600, marginBottom: 4 },
  note:  { fontSize: 12, color: '#8b949e', paddingLeft: 10, borderLeft: '2px solid #30363d', margin: '3px 0' },
  noteInput: {
    width: '100%', background: '#0d1117', border: '1px solid #21262d',
    borderRadius: 5, padding: '4px 8px', color: '#e6edf3',
    fontSize: 12, outline: 'none', marginTop: 6,
  },
  addBtn: {
    marginTop: 4, padding: '3px 10px', fontSize: 11, fontWeight: 600,
    borderRadius: 5, cursor: 'pointer',
    border: '1px solid #238636', background: 'rgba(46,160,67,0.1)',
    color: '#3fb950', transition: 'all 150ms',
  },
  deleteBtn: {
    marginLeft: 6, padding: '1px 6px', fontSize: 10,
    borderRadius: 4, cursor: 'pointer',
    border: '1px solid rgba(248,81,73,0.3)',
    background: 'transparent', color: '#f85149', transition: 'all 150ms',
  },
  empty: { fontSize: 12, color: '#484f58', fontStyle: 'italic', padding: '8px 0' },
  manualCompose: {
    display: 'flex', gap: 6, marginTop: 4,
  },
  composeInput: {
    flex: 1, background: '#0d1117', border: '1px solid #30363d',
    borderRadius: 6, padding: '5px 10px', color: '#e6edf3',
    fontSize: 12, outline: 'none',
  },
}

function EntryCard({ entry, journal }: { entry: JournalEntry; journal: StageJournalReturn }) {
  const [noteText, setNoteText] = useState('')
  const [showInput, setShowInput] = useState(false)

  const submit = () => {
    if (noteText.trim()) {
      journal.addNote(entry.id, noteText)
      setNoteText('')
      setShowInput(false)
    }
  }

  return (
    <div style={st.entry}>
      <div style={st.entryHeader}>
        <span style={st.stageBadge(entry.stage)}>{entry.stage}</span>
        <span style={st.ts}>{fmtTs(entry.timestamp)}</span>
        {entry.ea1Eligible && <span style={st.ea1ok}>✦ EA-1</span>}
        <span style={st.focusVal}>focus {(entry.focusScore * 100).toFixed(0)}%</span>
        <button style={st.deleteBtn} onClick={() => journal.deleteEntry(entry.id)} title="Delete entry">✕</button>
      </div>

      <div style={st.title}>{entry.autoTitle}</div>

      {entry.notes.map((n, i) => (
        <div key={i} style={st.note}>{n}</div>
      ))}

      {showInput ? (
        <>
          <input
            style={st.noteInput}
            value={noteText}
            onChange={e => setNoteText(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && submit()}
            placeholder="Add note…"
            autoFocus
          />
          <button style={st.addBtn} onClick={submit}>Save note</button>
          <button style={{ ...st.addBtn, marginLeft: 6, borderColor: '#484f58', color: '#484f58', background: 'transparent' }}
            onClick={() => setShowInput(false)}>Cancel</button>
        </>
      ) : (
        <button style={{ ...st.addBtn, marginTop: 6 }} onClick={() => setShowInput(true)}>+ Add note</button>
      )}
    </div>
  )
}

export default function StageJournal({ journal }: Props) {
  const [composeText, setComposeText] = useState('')

  const submitManual = () => {
    if (composeText.trim()) {
      journal.addManualEntry(composeText)
      setComposeText('')
    }
  }

  return (
    <div style={st.root}>
      {/* Toolbar */}
      <div style={st.toolbar}>
        <input
          style={st.searchInput}
          value={journal.filter}
          onChange={e => journal.setFilter(e.target.value)}
          placeholder="Filter by stage or text…"
        />
        <button style={st.toolBtn} onClick={journal.exportCSV}
          title="Export journal as CSV">↓ CSV</button>
      </div>

      {/* Manual note composer */}
      <div style={st.manualCompose}>
        <input
          style={st.composeInput}
          value={composeText}
          onChange={e => setComposeText(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && submitManual()}
          placeholder="Write a note about this moment…"
        />
        <button style={{ ...st.addBtn, marginTop: 0, flexShrink: 0 }} onClick={submitManual}>
          + Add
        </button>
      </div>

      {/* Entry list */}
      <div style={st.entries}>
        {journal.filteredEntries.length === 0 ? (
          <p style={st.empty}>
            {journal.entries.length === 0
              ? 'No journal entries yet. Stage transitions will appear here automatically.'
              : 'No entries match the current filter.'}
          </p>
        ) : (
          journal.filteredEntries.map(entry => (
            <EntryCard key={entry.id} entry={entry} journal={journal} />
          ))
        )}
      </div>
    </div>
  )
}
