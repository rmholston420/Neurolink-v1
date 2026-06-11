import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import App from '../App'
import type { NeurolinkState } from '../types'

// Mock the SSE hook so App renders without a real EventSource
vi.mock('../hooks/useNeurolinkSSE', () => ({
  useNeurolinkSSE: vi.fn(),
}))

import { useNeurolinkSSE } from '../hooks/useNeurolinkSSE'
const mockHook = vi.mocked(useNeurolinkSSE)

const fullState = (): NeurolinkState => ({
  connected: true,
  source: 'mock',
  region: 'B',
  alchemical_stage: 'Albedo',
  integration_coverage: 0.8,
  engagement_index: 0.7,
  bands: { alpha: 0.3, theta: 0.2, beta: 0.2, delta: 0.2, gamma: 0.1 },
  s_space: { x: 0.1, y: 0.2, z: 0.3 },
  ea1: {
    eligible: true, score: 0.9, criteria_met: 4, criteria_total: 5,
    label: 'Eligible', overlay_mode: 'X4', gates: {}, criteria: {},
    alchemical_stage: 'Albedo', s_space_coords: null,
    s_space_region: 'B', integration_coverage: 0.8,
  },
  last_ts: 1000,
  frame_count: 99,
  poor_contact: false,
  region_v01: 'C',
  alchemical_stage_v01: 'Citrinitas',
  faa: 0.1, fmt: 0.2,
  hr_bpm: 65, hrv_rmssd: 42.5, rr_bpm: 14,
  pitch_deg: 5, roll_deg: -3, motion_rms: 0.01,
  contact_quality: 95,
  focus_state: 'high_focus',
  focus_score: 0.85,
  fatigue_score: 0.1,
  fnirs_oxy: 0.6,
  fnirs_deoxy: 0.4,
})

describe('App', () => {
  it('renders the Neurolink title', () => {
    mockHook.mockReturnValue(null)
    render(<App />)
    expect(screen.getByText(/Neurolink/i)).toBeTruthy()
  })

  it('shows Disconnected badge when state is null', () => {
    mockHook.mockReturnValue(null)
    render(<App />)
    expect(screen.getByText('Disconnected')).toBeTruthy()
  })

  it('shows Connected badge when state.connected is true', () => {
    mockHook.mockReturnValue(fullState())
    render(<App />)
    expect(screen.getByText('Connected')).toBeTruthy()
  })

  it('shows frame count and source when state is present', () => {
    mockHook.mockReturnValue(fullState())
    render(<App />)
    expect(screen.getByText(/Frame #99/)).toBeTruthy()
    expect(screen.getByText(/Source: mock/)).toBeTruthy()
  })

  it('renders all 9 panel card titles', () => {
    mockHook.mockReturnValue(fullState())
    render(<App />)
    const titles = [
      'Band Powers',
      'S-Space / Alchemical Stage',
      'EA-1 Eligibility',
      'Heart Rate & HRV',
      'Focus & Fatigue',
      'Contact Quality',
      'Breathing',
      'Head Pose & Motion',
      'Calibration',
    ]
    for (const t of titles) {
      expect(screen.getByText(t)).toBeTruthy()
    }
  })
})
