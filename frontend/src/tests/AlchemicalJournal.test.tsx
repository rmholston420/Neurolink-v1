import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import AlchemicalJournal from '../components/AlchemicalJournal'
import type { AlchemicalJournalResult, JournalEntry } from '../hooks/useAlchemicalJournal'

function makeJournal(overrides: Partial<AlchemicalJournalResult> = {}): AlchemicalJournalResult {
  return {
    entries: [],
    addEntry: vi.fn(),
    deleteEntry: vi.fn(),
    query: '',
    setQuery: vi.fn(),
    filtered: [],
    ...overrides,
  }
}

function makeEntry(overrides: Partial<JournalEntry> = {}): JournalEntry {
  return {
    id: 'entry-1',
    timestamp: new Date('2026-01-01T12:00:00Z').getTime(),
    stage: 'Albedo',
    ea1Score: 0.85,
    focusScore: 72,
    focusState: 'high_focus',
    alchemicalStage: 'Albedo',
    text: 'Deep stillness observed.',
    tags: ['Albedo', 'EA-1'],
    ...overrides,
  }
}

describe('AlchemicalJournal', () => {
  beforeEach(() => {
    // No STORE cleanup needed; component is driven entirely by props
  })

  it('renders the note textarea', () => {
    render(<AlchemicalJournal journal={makeJournal()} />)
    expect(screen.getByPlaceholderText(/Note your inner experience/i)).toBeTruthy()
  })

  it('renders the Save button', () => {
    render(<AlchemicalJournal journal={makeJournal()} />)
    expect(screen.getByRole('button', { name: /Save/i })).toBeTruthy()
  })

  it('Save button is disabled when textarea is empty', () => {
    render(<AlchemicalJournal journal={makeJournal()} />)
    const btn = screen.getByRole('button', { name: /Save/i })
    expect(btn.hasAttribute('disabled')).toBe(true)
  })

  it('calls addEntry when Save button clicked with text', () => {
    const addEntry = vi.fn()
    render(<AlchemicalJournal journal={makeJournal({ addEntry })} />)
    const textarea = screen.getByPlaceholderText(/Note your inner experience/i)
    fireEvent.change(textarea, { target: { value: 'My observation' } })
    fireEvent.click(screen.getByRole('button', { name: /Save/i }))
    expect(addEntry).toHaveBeenCalledWith('My observation', [])
  })

  it('includes extra tags when tag input is filled', () => {
    const addEntry = vi.fn()
    render(<AlchemicalJournal journal={makeJournal({ addEntry })} />)
    fireEvent.change(
      screen.getByPlaceholderText(/Note your inner experience/i),
      { target: { value: 'Luminous void' } }
    )
    fireEvent.change(
      screen.getByPlaceholderText(/Extra tags/i),
      { target: { value: 'dream, vision' } }
    )
    fireEvent.click(screen.getByRole('button', { name: /Save/i }))
    expect(addEntry).toHaveBeenCalledWith('Luminous void', ['dream', 'vision'])
  })

  it('shows empty-state message when filtered is empty', () => {
    render(<AlchemicalJournal journal={makeJournal({ filtered: [] })} />)
    expect(screen.getByText(/No entries yet/i)).toBeTruthy()
  })

  it('renders entries from filtered list', () => {
    const entry = makeEntry()
    render(<AlchemicalJournal journal={makeJournal({ filtered: [entry] })} />)
    expect(screen.getByText('Deep stillness observed.')).toBeTruthy()
  })

  it('renders entry tags as badges', () => {
    const entry = makeEntry({ tags: ['Albedo', 'EA-1'] })
    render(<AlchemicalJournal journal={makeJournal({ filtered: [entry] })} />)
    expect(screen.getByText('Albedo')).toBeTruthy()
    expect(screen.getByText('EA-1')).toBeTruthy()
  })

  it('calls deleteEntry when ✕ button clicked', () => {
    const deleteEntry = vi.fn()
    const entry = makeEntry()
    render(<AlchemicalJournal journal={makeJournal({ filtered: [entry], deleteEntry })} />)
    fireEvent.click(screen.getByTitle('Delete entry'))
    expect(deleteEntry).toHaveBeenCalledWith('entry-1')
  })

  it('renders search input', () => {
    render(<AlchemicalJournal journal={makeJournal()} />)
    expect(screen.getByPlaceholderText(/Search entries/i)).toBeTruthy()
  })

  it('calls setQuery when search input changes', () => {
    const setQuery = vi.fn()
    render(<AlchemicalJournal journal={makeJournal({ setQuery })} />)
    fireEvent.change(screen.getByPlaceholderText(/Search entries/i), {
      target: { value: 'albedo' },
    })
    expect(setQuery).toHaveBeenCalledWith('albedo')
  })

  it('shows null ea1Score gracefully (no percentage rendered for null)', () => {
    const entry = makeEntry({ ea1Score: null })
    render(<AlchemicalJournal journal={makeJournal({ filtered: [entry] })} />)
    // Should not throw; should not render "null%"
    expect(screen.queryByText(/null%/)).toBeNull()
  })
})
