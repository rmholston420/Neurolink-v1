"""Pydantic v2 data models for Neurolink EEG pipeline."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class BandPowers(BaseModel):
    model_config = ConfigDict(extra="ignore")
    alpha: float = 0.0
    theta: float = 0.0
    beta: float = 0.0
    delta: float = 0.0
    gamma: float = 0.0


class SSpaceCoords(BaseModel):
    model_config = ConfigDict(extra="ignore")
    x: float = 0.0   # engagement index = beta / (alpha + theta)
    y: float = 0.0   # integration coverage = alpha / beta
    z: float = 0.0   # theta fraction (raw)


class PoincareIndices(BaseModel):
    model_config = ConfigDict(extra="ignore")
    sd1: float = 0.0
    sd2: float = 0.0
    sd1_sd2_ratio: float = 0.0
    ellipse_area: float = 0.0


class PPGPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")
    hr_bpm: float = 0.0
    ibi_ms: list[float] = Field(default_factory=list)
    hrv_rmssd: float = 0.0
    hrv_sdnn: float = 0.0
    hrv_pnn50: float = 0.0
    poincare: PoincareIndices | None = None


class BreathingPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")
    rr_bpm: float | None = None
    rr_ppg: float | None = None
    rr_accel: float | None = None


class IMUPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    motion_rms: float = 0.0


class IngestPayload(BaseModel):
    """Full multimodal ingest payload from any hardware adapter."""
    model_config = ConfigDict(extra="ignore")
    # Core EEG
    region: str = "A"
    alchemical_stage: str = "Nigredo"
    ea1_eligible: bool = False
    integration_coverage: float = 0.5
    engagement_index: float = 0.5
    bands: BandPowers = Field(default_factory=BandPowers)
    s_space: SSpaceCoords = Field(default_factory=SSpaceCoords)
    timestamp: float = 0.0
    source: str = "mock"               # "muse_ble" | "muse_lsl" | "athena_ble" | "mock"
    address: str = ""
    # Contact
    poor_contact: bool = False
    contact_quality: float | None = None
    # Derived EEG
    faa: float | None = None           # Frontal Alpha Asymmetry
    fmt: float | None = None           # Frontal Midline Theta
    # Optional multimodal
    ppg: PPGPayload | None = None
    breathing: BreathingPayload | None = None
    imu: IMUPayload | None = None
    # Athena-only
    fnirs_oxy: float | None = None
    fnirs_deoxy: float | None = None


class EA1Criterion(BaseModel):
    model_config = ConfigDict(extra="ignore")
    value: float | None = None
    threshold: float | None = None
    units: str = ""
    met: bool = False


class EA1Result(BaseModel):
    """EA-1 multimodal eligibility score."""
    model_config = ConfigDict(extra="ignore")
    eligible: bool = False
    score: float = 0.0
    criteria_met: int = 0
    criteria_total: int = 5
    label: str = "Ineligible"
    gates: dict[str, bool] = Field(default_factory=lambda: {"s_space": False, "motion": True})
    criteria: dict[str, Any] = Field(default_factory=dict)
    overlay_mode: str = "X0"
    alchemical_stage: str = "Nigredo"
    s_space_coords: SSpaceCoords = Field(default_factory=SSpaceCoords)
    s_space_region: str = "A"
    integration_coverage: float = 0.0


class NeurolinkState(BaseModel):
    """Current live state of the Neurolink hub."""
    model_config = ConfigDict(extra="ignore")
    connected: bool = False
    source: str = "none"
    region: str = "A"
    alchemical_stage: str = "Nigredo"
    integration_coverage: float = 0.0
    engagement_index: float = 0.0
    bands: BandPowers = Field(default_factory=BandPowers)
    s_space: SSpaceCoords = Field(default_factory=SSpaceCoords)
    ea1: EA1Result = Field(default_factory=EA1Result)
    last_ts: float = 0.0
    frame_count: int = 0
    poor_contact: bool = False
    # v0.1 6-region classifier (muse_ble only)
    region_v01: str = "A"
    alchemical_stage_v01: str = "Nigredo"
    # Extended multimodal
    faa: float | None = None
    fmt: float | None = None
    hr_bpm: float | None = None
    hrv_rmssd: float | None = None
    rr_bpm: float | None = None
    pitch_deg: float | None = None
    roll_deg: float | None = None
    motion_rms: float | None = None
    contact_quality: float | None = None
    # Focus + Fatigue
    focus_state: str = "unknown"       # FocusState enum value
    focus_score: float = 0.0
    fatigue_score: float = 0.0
    # Athena-only
    fnirs_oxy: float | None = None
    fnirs_deoxy: float | None = None


# ── API request/response schemas ──────────────────────────────────────────

class ConnectRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    adapter_type: str = "ble"          # "ble" | "lsl" | "mock"
    device_model: str = "muse_s_gen1"  # "muse_s_gen1" | "muse_s_athena"
    address: str | None = None         # BLE MAC address (required for ble mode)


class ConnectResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    ok: bool
    source: str
    message: str


class DisconnectResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    ok: bool


class BandPowerResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    channel: str
    alpha: float | None = None
    theta: float | None = None
    beta: float | None = None
    delta: float | None = None
    gamma: float | None = None
    error: str | None = None


class CalibrateResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    status: str                        # "started" | "complete" | "error"
    baseline_alpha: float | None = None


class SessionSummary(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: int
    started_at: datetime
    ended_at: datetime | None = None
    device_model: str
    adapter_type: str
    frame_count: int
    final_ea1_eligible: bool


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    status: str                        # "ok" | "degraded"
    adapter_type: str
    adapter_connected: bool
    hub_frame_count: int
    redis: str                         # "connected" | "error" | "disabled"
    db: str                            # "connected" | "error"
