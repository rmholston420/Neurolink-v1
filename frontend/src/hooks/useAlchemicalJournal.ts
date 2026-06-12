/**
 * useAlchemicalJournal — Tier 2
 * In-memory per-stage journal with auto-tagging from live NeurolinkState.
 * Entries are persisted to a module-level array so they survive tab switches.
 */
import { useState, useRef } from 'react'
import type { NeurolinkState } from '../types'

export interface JournalEntry {
  id:             string
  timestamp:      number    // unix ms
  stage:          string
  ea1Score:       number | null
  focusScore:     number
  focusState:     string
  alchemicalStage: string
  text:           string
  tags:           string[]
}

export interface AlchemicalJournalResult {
  entries:     JournalEntry[]
  addEntry:    (text: string, extraTags?: string[]) => void
  deleteEntry: (id: string) => void
  query:       string
  setQuery:    (q: string) => void
  filtered:    JournalEntry[]
}

const STORE: JournalEntry[] = []   // module-level persistence across mounts

export function useAlchemicalJournal(state: NeurolinkState | null): AlchemicalJournalResult {
  const [entries, setEntries] = useState<JournalEntry[]>(STORE)
  const [query, setQuery]    = useState('')
  const stateRef = useRef(state)
  stateRef.current = state

  function addEntry(text: string, extraTags: string[] = []) {
    const s = stateRef.current
    const stage  = s?.alchemical_stage ?? 'Unknown'
    const autoTags: string[] = [
      stage,
      s?.focus_state ?? '',
      s?.ea1?.eligible ? 'EA-1' : '',
      ...extraTags,
    ].filter(Boolean)

    const entry: JournalEntry = {
      id:             crypto.randomUUID(),
      timestamp:      Date.now(),
      stage:          s?.alchemical_stage_v01 ?? stage,
      ea1Score:       s?.ea1?.score ?? null,
      focusScore:     s?.focus_score ?? 0,
      focusState:     s?.focus_state ?? 'unknown',
      alchemicalStage: stage,
      text,
      tags:           autoTags,
    }
    STORE.unshift(entry)
    setEntries([...STORE])
  }

  function deleteEntry(id: string) {
    const idx = STORE.findIndex(e => e.id === id)
    if (idx !== -1) STORE.splice(idx, 1)
    setEntries([...STORE])
  }

  const q = query.toLowerCase()
  const filtered = q
    ? entries.filter(e =>
        e.text.toLowerCase().includes(q) ||
        e.alchemicalStage.toLowerCase().includes(q) ||
        e.tags.some(t => t.toLowerCase().includes(q))
      )
    : entries

  return { entries, addEntry, deleteEntry, query, setQuery, filtered }
}
