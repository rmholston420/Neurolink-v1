// Neurolink frontend type definitions — mirrors backend models/eeg.py

export interface BandPowers {
  alpha: number
  theta: number
  beta: number
  delta: number
  gamma: number
}

export interface SSpaceCoords {
  x: number  // engagement index
  y: number  // integration coverage
  z: number  // gamma index
}

export interface EA1Result {
  eligible: boolean
  score: number
  criteria_met: number
  criteria_total: number
  label: string
  gates: Record<string, boolean>
  criteria: Record<string, unknown>
  overlay_mode: string
  alchemical_stage: string
  s_space_coords: SSpaceCoords | null
  s_space_region: string
  integration_coverage: number
}

export interface NeurolinkState {
  connected: boolean
  source: string
  region: string
  alchemical_stage: string
  integration_coverage: number
  engagement_index: number
  bands: BandPowers
  s_space: SSpaceCoords | null
  ea1: EA1Result
  last_ts: number
  frame_count: number
  poor_contact: boolean
  region_v01: string
  alchemical_stage_v01: string
  faa: number | null
  fmt: number | null
  hr_bpm: number | null
  hrv_rmssd: number | null
  rr_bpm: number | null
  pitch_deg: number | null
  roll_deg: number | null
  motion_rms: number | null
  contact_quality: number | null
  focus_state: string
  focus_score: number
  fatigue_score: number
  fnirs_oxy: number | null
  fnirs_deoxy: number | null
  /**
   * Raw EEG sample window forwarded from the adapter.
   * Shape: [n_channels][n_samples]  e.g. 4 channels × 64 samples.
   * Empty array when the adapter does not provide raw buffers.
   */
  eeg_samples: number[][]
  /** Stage 2: channel names flagged as bad this frame. */
  bad_channels: string[]
  /** Stage 3: true when the epoch-level artifact gate rejected this frame. */
  artifact_rejected: boolean
  /** Stage 3: human-readable list of rejection causes, e.g. ["amplitude", "kurtosis"]. */
  artifact_reasons: string[]
  /**
   * Stage 3b: per-channel artifact type annotations produced by the
   * classification pipeline.  Keys are channel names; values are the
   * artifact classifier label (e.g. 'ocular', 'muscle', 'cardiac', 'clean').
   * Empty object when Stage 3b is disabled or no frame has been processed.
   */
  artifact_annotations: Record<string, string>
  /**
   * Stage 3b: recommended correction plan for this frame.
   * E.g. 'ica_ocular', 'regression_cardiac', 'none'.  Empty string when no
   * plan was generated.
   */
  artifact_correction_plan: string
  /**
   * Per-channel impedance in kΩ.
   * Empty object when the adapter does not expose per-channel impedance.
   */
  channel_impedances: Record<string, number>
  /**
   * Baseline recorder phase: 'warmup' | 'recording' | 'complete'.
   * Null before the first device connects.
   */
  baseline_phase: string | null
}

// ─── SSE sentinel types ─────────────────────────────────────────────────────

/**
 * The four reason codes the backend sends in a settling event.
 * 'settling' is the generic fallback; all others are specific causes.
 */
export type SettlingReason =
  | 'impedance_unstable'
  | 'motion_settling'
  | 'env_not_ready'
  | 'settling'

/**
 * SSE sentinel emitted by the backend on every frame held by the Stage 0
 * acquisition guard.  The frontend should display a contextual waiting
 * indicator keyed on the reason field.
 *
 * Wire format (named SSE event):
 *   event: settling
 *   data: {"reason": "<SettlingReason>"}
 */
export interface SettlingSentinel {
  event: 'settling'
  reason: SettlingReason
}

/**
 * SSE sentinel emitted exactly once when the 150 s baseline recording
 * completes.  The frontend should play a bell sound and unlock the session UI.
 *
 * Wire format (named SSE event):
 *   event: baseline_complete
 *   data: {}
 */
export interface BaselineCompleteSentinel {
  event: 'baseline_complete'
}
