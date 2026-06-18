import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import SignalPipelinePanel from '../components/SignalPipelinePanel'
import type { NeurolinkState } from '../types'
import type { ArtifactStats } from '../hooks/useArtifactStats'

const defaultStats: ArtifactStats = {
  totalFrames: 0, rejectedFrames: 0, rejectRate: 0,
  causeCounts: {}, windowSize: 300, reset: vi.fn(),
}

const baseState: NeurolinkState = {
  connected: true,
  source: 'mock',
  region: 'B',
  alchemical_stage: 'Albedo',
  integration_coverage: 0.8,
  engagement_index: 0.7,
  bands: { alpha: 0.3, theta: 0.2, beta: 0.2, delta: 0.2, gamma: 0.1 },
  s_space: null,
  ea1: {
    eligible: false, score: 0.5, criteria_met: 2, criteria_total: 5,
    label: '', overlay_mode: '', gates: {}, criteria: {},
    alchemical_stage: 'Albedo', s_space_coords: null,
    s_space_region: '', integration_coverage: 0,
  },
  last_ts: 0, frame_count: 42,
  poor_contact: false, region_v01: '', alchemical_stage_v01: '',
  faa: null, fmt: null, hr_bpm: null, hrv_rmssd: null, rr_bpm: null,
  pitch_deg: null, roll_deg: null, motion_rms: null,
  contact_quality: null, focus_state: '', focus_score: 0, fatigue_score: 0,
  fnirs_oxy: null, fnirs_deoxy: null,
  eeg_samples: [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
  bad_channels: [],
  artifact_rejected: false,
  artifact_reasons: [],
  artifact_annotations: {},
  artifact_correction_plan: '',
  channel_impedances: {},
  baseline_phase: null,
}

describe('SignalPipelinePanel', () => {
  it('shows "Not connected" when state is null', () => {
    render(<SignalPipelinePanel state={null} stats={defaultStats} />)
    expect(screen.getByText('Not connected')).toBeTruthy()
  })

  it('renders all four stage labels', () => {
    render(<SignalPipelinePanel state={baseState} stats={defaultStats} />)
    expect(screen.getByText('Acquisition')).toBeTruthy()
    expect(screen.getByText('Filtering')).toBeTruthy()
    expect(screen.getByText('Channels')).toBeTruthy()
    expect(screen.getByText('Artifact Gate')).toBeTruthy()
  })

  it('shows channel and sample counts in Stage 0', () => {
    render(<SignalPipelinePanel state={baseState} stats={defaultStats} />)
    expect(screen.getByText(/4 ch/)).toBeTruthy()
    expect(screen.getByText(/3 samp\/frame/)).toBeTruthy()
    expect(screen.getByText(/Frame #42/)).toBeTruthy()
  })

  it('renders Stage 1 static filter labels', () => {
    render(<SignalPipelinePanel state={baseState} stats={defaultStats} />)
    expect(screen.getByText(/HP 0\.5 Hz/i)).toBeTruthy()
    expect(screen.getByText(/Notch 50\/60 Hz/i)).toBeTruthy()
    expect(screen.getByText(/Zero-phase FIR/i)).toBeTruthy()
  })

  it('shows "Frame accepted" when artifact_rejected is false', () => {
    render(<SignalPipelinePanel state={baseState} stats={defaultStats} />)
    expect(screen.getByText('Frame accepted')).toBeTruthy()
  })

  it('shows "Frame rejected" when artifact_rejected is true', () => {
    render(
      <SignalPipelinePanel
        state={{ ...baseState, artifact_rejected: true, artifact_reasons: ['amplitude'] }}
        stats={defaultStats}
      />
    )
    expect(screen.getByText('Frame rejected')).toBeTruthy()
    expect(screen.getByText(/Amp/)).toBeTruthy()
  })

  it('shows spherical-spline note when bad channels present', () => {
    render(
      <SignalPipelinePanel
        state={{ ...baseState, bad_channels: ['AF7'] }}
        stats={defaultStats}
      />
    )
    expect(screen.getByText(/Spherical-spline interpolation applied/i)).toBeTruthy()
  })

  it('shows reject rate percentage from stats', () => {
    render(
      <SignalPipelinePanel
        state={baseState}
        stats={{ ...defaultStats, rejectRate: 0.25 }}
      />
    )
    expect(screen.getByText('25.0%')).toBeTruthy()
  })

  it('shows "Waiting for EEG samples" when connected but eeg_samples empty', () => {
    render(
      <SignalPipelinePanel
        state={{ ...baseState, eeg_samples: [] }}
        stats={defaultStats}
      />
    )
    expect(screen.getByText(/Waiting for EEG samples/i)).toBeTruthy()
  })
})
