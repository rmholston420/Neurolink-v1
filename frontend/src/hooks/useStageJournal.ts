/**
 * useStageJournal
 *
 * Maintains a journal of alchemical stage transitions and user notes.
 *
 * Auto-behaviour:
 *   - When alchemical_stage changes, a new journal entry is auto-created
 *     tagged with stage, EA-1 eligible, focus_score, and timestamp
 *   - User can attach free-text notes to any entry
 *
 * Manual behaviour:
 *   - addNote(entryId, text)   — append a note to an existing entry
 *   - addManualEntry(text)     — add a freeform entry not tied to a stage change
 *   - deleteEntry(id)          — remove an entry
 *   - exportCSV()              — Blob CSV download
 *
 * Search:
 *   - setFilter(text)  — filters entries by stage name or note text
 */
import { useState, useRef, useEffect, useCallback } from 'react'
import type { NeurolinkState } from '../types'

export interface JournalEntry {
  id:           string
  timestamp:    number
  type:         'auto' | 'manual'
  stage:        string
  ea1Eligible:  boolean
  focusScore:   number
  notes:        string[]   // user-appended notes
  autoTitle:    string     // e.g. "Entered Citrinitas"
}

export interface StageJournalReturn {
  entries:        JournalEntry[]
  filteredEntries: JournalEntry[]
  filter:         string
  setFilter:      (f: string) => void
  addNote:        (id: string, text: string) => void
  addManualEntry: (text: string, stage?: string) => void
  deleteEntry:    (id: string) => void
  exportCSV:      () => void
}

function todayISO(): string {
  return new Date().toISOString()
}

export function useStageJournal(
  state: Partial<NeurolinkState> | null,
): StageJournalReturn {
  const [entries, setEntries] = useState<JournalEntry[]>([])
  const [filter,  setFilter]  = useState('')
  const prevStageRef = useRef<string>('')

  // Auto-create entry on stage transition
  useEffect(() => {
    const stage = state?.alchemical_stage ?? ''
    if (!stage || stage === prevStageRef.current) return
    const isFirst = prevStageRef.current === ''
    prevStageRef.current = stage
    if (isFirst) return  // don't log the very first stage on mount

    const entry: JournalEntry = {
      id:          `${Date.now()}-auto`,
      timestamp:   Date.now(),
      type:        'auto',
      stage,
      ea1Eligible: state?.ea1?.eligible ?? false,
      focusScore:  state?.focus_score   ?? 0,
      notes:       [],
      autoTitle:   `Entered ${stage}`,
    }
    setEntries(prev => [entry, ...prev])
  }, [state?.alchemical_stage])

  const addNote = useCallback((id: string, text: string) => {
    if (!text.trim()) return
    setEntries(prev => prev.map(e =>
      e.id === id ? { ...e, notes: [...e.notes, text.trim()] } : e
    ))
  }, [])

  const addManualEntry = useCallback((text: string, stage?: string) => {
    if (!text.trim()) return
    const entry: JournalEntry = {
      id:          `${Date.now()}-manual`,
      timestamp:   Date.now(),
      type:        'manual',
      stage:       stage ?? prevStageRef.current ?? 'Unknown',
      ea1Eligible: state?.ea1?.eligible ?? false,
      focusScore:  state?.focus_score   ?? 0,
      notes:       [text.trim()],
      autoTitle:   'Note',
    }
    setEntries(prev => [entry, ...prev])
  }, [state?.ea1?.eligible, state?.focus_score])

  const deleteEntry = useCallback((id: string) => {
    setEntries(prev => prev.filter(e => e.id !== id))
  }, [])

  const exportCSV = useCallback(() => {
    if (entries.length === 0) return
    const header = 'Timestamp,Type,Stage,EA1,Focus,Title,Notes\n'
    const rows = entries.map(e =>
      `"${new Date(e.timestamp).toISOString()}",${e.type},${e.stage},${e.ea1Eligible},${e.focusScore.toFixed(2)},"${e.autoTitle}","${e.notes.join(' | ')}"`
    ).join('\n')
    const blob = new Blob([header + rows], { type: 'text/csv' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = `neurolink-journal-${new Date().toISOString().slice(0,10)}.csv`
    a.click()
    URL.revokeObjectURL(a.href)
  }, [entries])

  const filteredEntries = filter.trim()
    ? entries.filter(e =>
        e.stage.toLowerCase().includes(filter.toLowerCase()) ||
        e.autoTitle.toLowerCase().includes(filter.toLowerCase()) ||
        e.notes.some(n => n.toLowerCase().includes(filter.toLowerCase()))
      )
    : entries

  return { entries, filteredEntries, filter, setFilter, addNote, addManualEntry, deleteEntry, exportCSV }
}
