/**
 * Shared TypeScript interfaces mirroring the backend Pydantic models.
 * Source of truth: backend/src/neurolink/models/eeg.py
 */

export interface BandPowers {
  alpha: number
  theta: number
  beta: number
  delta: number
  gamma: number
}

export interface SSpaceCoords {
  x: number
  y: number
  z: number
}

export interface IMUPayload {
  pitch_deg: number
  roll_deg: number
  motion_rms: number
}

export interface PPGPayload {
  hr_bpm: number
  hrv_rmssd: number
  ibi_ms: number[]
  sd1: number
  sd2: number
  ellipse_area: number
}

export interface BreathingPayload {
  rr_bpm: number | null
  rr_ppg: number | null
  rr_accel: number | null
}

export interface EA1Criterion {
  value: number | null
  threshold: number | null
  units: string
  met: boolean
}

export interface EA1Result {
  eligible: boolean
  score: number
  criteria_met: number
  criteria_total: number
  label: string
  gates: Record<string, boolean>
  criteria: Record<string, EA1Criterion>
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
}

export interface ConnectRequest {
  adapter_type: 'mock' | 'ble' | 'lsl'
  device_model: 'muse_s_gen1' | 'muse_s_athena' | 'mock'
  address?: string | null
}

export interface ConnectResponse {
  ok: boolean
  source: string
  message: string
}

export interface HealthResponse {
  status: 'ok' | 'degraded'
  adapter_type: string
  adapter_connected: boolean
  hub_frame_count: number
  redis: string
  db: string
}

export interface SessionSummary {
  id: number
  started_at: string | null
  ended_at: string | null
  device_model: string
  adapter_type: string
  frame_count: number
  final_ea1_eligible: boolean | null
}

// ── BLE scan types (Path B) ───────────────────────────────────────────────────

export interface BLEDevice {
  address: string
  name:    string | null
  rssi?:   number
}

export interface BLEScanResponse {
  devices: BLEDevice[]
  scan_duration_sec: number
}

// ── Web Bluetooth status (Path A) ─────────────────────────────────────────────
export type WebBTStatus =
  | 'unsupported'
  | 'idle'
  | 'requesting'
  | 'connecting'
  | 'streaming'
  | 'reconnecting'
  | 'error'
