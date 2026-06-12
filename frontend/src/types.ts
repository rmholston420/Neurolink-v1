// Neurolink frontend type definitions — mirrors backend models/eeg.py

export interface BandPowers {
  alpha: number;
  theta: number;
  beta: number;
  delta: number;
  gamma: number;
}

export interface SSpaceCoords {
  x: number; // engagement index
  y: number; // integration coverage
  z: number; // gamma index
}

export interface EA1Result {
  eligible: boolean;
  score: number;
  criteria_met: number;
  criteria_total: number;
  label: string;
  gates: Record<string, boolean>;
  criteria: Record<string, unknown>;
  overlay_mode: string;
  alchemical_stage: string;
  s_space_coords: SSpaceCoords | null;
  s_space_region: string;
  integration_coverage: number;
}

export interface NeurolinkState {
  connected: boolean;
  source: string;
  region: string;
  alchemical_stage: string;
  integration_coverage: number;
  engagement_index: number;
  bands: BandPowers;
  s_space: SSpaceCoords | null;
  ea1: EA1Result;
  last_ts: number;
  frame_count: number;
  poor_contact: boolean;
  region_v01: string;
  alchemical_stage_v01: string;
  faa: number | null;
  fmt: number | null;
  hr_bpm: number | null;
  hrv_rmssd: number | null;
  rr_bpm: number | null;
  pitch_deg: number | null;
  roll_deg: number | null;
  motion_rms: number | null;
  contact_quality: number | null;
  focus_state: string;
  focus_score: number;
  fatigue_score: number;
  fnirs_oxy: number | null;
  fnirs_deoxy: number | null;
  /**
   * Raw EEG sample window forwarded from the adapter.
   * Shape: [n_channels][n_samples]  e.g. 4 channels × 64 samples.
   * Empty array when the adapter does not provide raw buffers.
   */
  eeg_samples: number[][];
}
