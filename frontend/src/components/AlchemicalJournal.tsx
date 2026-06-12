/**
 * AlchemicalJournal — Tier 2
 * Rich per-stage note-taking panel with auto-tags, search, and scrollable log.
 */
import React, { useState } from 'react'
import type { AlchemicalJournalResult } from '../hooks/useAlchemicalJournal'

interface Props {
  journal: AlchemicalJournalResult
}

const stageColors: Record<string, string> = {
  Nigredo: '#8b949e', Albedo: '#cdd9e5', Citrinitas: '#d2a679', Rubedo: '#f85149',
}

export default function AlchemicalJournal({ journal }: Props) {
  const { addEntry, deleteEntry, query, setQuery, filtered } = journal
  const [draft, setDraft]       = useState('')
  const [tagInput, setTagInput]  = useState('')

  function submit() {
    const trimmed = draft.trim()
    if (!trimmed) return
    const extra = tagInput.split(',').map(t => t.trim()).filter(Boolean)
    addEntry(trimmed, extra)
    setDraft('')
    setTagInput('')
  }

  return (
    <div style={{ color: '#cdd9e5', display: 'flex', flexDirection: 'column', gap: 12 }}>
      {/* Entry composer */}
      <div>
        <textarea
          value={draft}
          onChange={e => setDraft(e.target.value)}
          placeholder="Note your inner experience — tags are added automatically from current stage, focus, and EA-1 status…"
          rows={3}
          style={{
            width: '100%', background: '#0d1117', border: '1px solid #30363d',
            borderRadius: 6, color: '#cdd9e5', padding: '8px 10px',
            fontSize: 13, resize: 'vertical', boxSizing: 'border-box',
          }}
          onKeyDown={e => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) submit() }}
        />
        <div style={{ display: 'flex', gap: 6, marginTop: 6 }}>
          <input
            value={tagInput}
            onChange={e => setTagInput(e.target.value)}
            placeholder="Extra tags (comma-separated)"
            style={{
              flex: 1, background: '#0d1117', border: '1px solid #30363d',
              borderRadius: 5, color: '#cdd9e5', padding: '4px 8px', fontSize: 12,
            }}
          />
          <button
            onClick={submit}
            disabled={!draft.trim()}
            style={{
              padding: '4px 14px', fontSize: 12, fontWeight: 600,
              background: draft.trim() ? '#238636' : '#21262d',
              color: draft.trim() ? '#fff' : '#8b949e',
              border: '1px solid #30363d', borderRadius: 5, cursor: 'pointer',
            }}
          >Save ⌘↵</button>
        </div>
      </div>

      {/* Search */}
      <input
        value={query}
        onChange={e => setQuery(e.target.value)}
        placeholder="🔍 Search entries…"
        style={{
          background: '#0d1117', border: '1px solid #30363d',
          borderRadius: 5, color: '#cdd9e5', padding: '5px 10px',
          fontSize: 12, width: '100%', boxSizing: 'border-box',
        }}
      />

      {/* Entry list */}
      <div style={{ maxHeight: 320, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 8 }}>
        {filtered.length === 0 && (
          <p style={{ color: '#484f58', fontSize: 12, textAlign: 'center', padding: '16px 0' }}>
            No entries yet — write your first note above
          </p>
        )}
        {filtered.map(entry => (
          <div key={entry.id} style={{
            background: '#0d1117', border: '1px solid #21262d',
            borderRadius: 7, padding: '9px 12px',
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 5 }}>
              <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                {entry.tags.map(t => (
                  <span key={t} style={{
                    padding: '1px 7px', borderRadius: 10, fontSize: 10, fontWeight: 700,
                    background: '#21262d',
                    color: stageColors[t] ?? '#58a6ff',
                    border: `1px solid ${stageColors[t] ? stageColors[t] + '44' : '#30363d'}`,
                  }}>{t}</span>
                ))}
              </div>
              <button
                onClick={() => deleteEntry(entry.id)}
                style={{ background: 'none', border: 'none', color: '#484f58', cursor: 'pointer', fontSize: 13 }}
                title="Delete entry"
              >✕</button>
            </div>
            <p style={{ fontSize: 13, margin: 0, lineHeight: 1.5 }}>{entry.text}</p>
            <div style={{ fontSize: 10, color: '#484f58', marginTop: 5 }}>
              {new Date(entry.timestamp).toLocaleString()} ·
              Focus {entry.focusScore.toFixed(0)}%
              {entry.ea1Score !== null ? ` · EA-1 ${(entry.ea1Score * 100).toFixed(0)}%` : ''}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
