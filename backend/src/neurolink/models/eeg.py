"""Pydantic v2 data models for the Neurolink API.

All API request/response and internal data transfer objects live here.
DO NOT add business logic here.
"""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, Field

# ============================================================================
# Sub-models
# ============================================================================


class BandPowers(BaseModel):
    """EEG band power fractions. All values in [0, 1]."""

    alpha: float = 0.0
    theta: float = 0.0
    beta: float = 0.0
    delta: float = 0.0
    gamma: float = 0.0


class SSpaceCoords(BaseModel):
    """S-space (EEG mandala) coordinates."""

    x: float = 0.0  # engagement index
    y: float = 0.0  # integration coverage
    z: float = 0.0  # gamma index


class IMUPayload(BaseModel):
    """IMU-derived head pose and motion data."""

    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    motion_rms: float = 0.0


class PPGPayload(BaseModel):
    """PPG-derived cardiovascular metrics."""

    hr_bpm: float = 0.0
    hrv_rmssd: float = 0.0
    ibi_ms: list[float] = Field(default_factory=list)
    sd1: float = 0.0
    sd2: float = 0.0
    ellipse_area: float = 0.0


class BreathingPayload(BaseModel):
    """Breathing rate estimates."""

    rr_bpm: float | None = None  # fused
    rr_ppg: float | None = None  # from IBI series
    rr_accel: float | None = None  # from accelerometer


class EA1Criterion(BaseModel):
    """Single EA-1 eligibility criterion."""

    value: float | None = None
    threshold: float | None = None
    units: str = ""
    met: bool = False


class EA1Result(BaseModel):
    """EA-1 multimodal eligibility result."""

    eligible: bool = False
    score: float = 0.0
    criteria_met: int = 0
    criteria_total: int = 5
    label: str = "Ineligible"
    gates: dict[str, bool] = Field(default_factory=dict)
    criteria: dict[str, Any] = Field(default_factory=dict)
    overlay_mode: str = "X0"
    alchemical_stage: str = ""
    s_space_coords: SSpaceCoords | None = None
    s_space_region: str = ""
    integration_coverage: float = 0.0


class ArtifactAnnotationPayload(BaseModel):
    """Single artifact annotation produced by Stage 3b ArtifactDetector."""

    artifact_type: str
    confidence: float
    channels: list[str]
    feature_value: float
    feature_name: str
    threshold: float


class ArtifactCorrectionPlanPayload(BaseModel):
    """Serialisable snapshot of the CorrectionPlan built by Stage 3b."""

    hard_reject: bool = False
    apply_ocular_regression: bool = False
    apply_asr: bool = False
    apply_notch: bool = False
    apply_cardiac_regression: bool = False


class StreamHealthPayload(BaseModel):
    """Real-time stream quality metrics, included in every SSE frame.

    Sourced from ``EEGPipeline.health`` (a ``StreamHealth`` dataclass)
    and serialised into ``NeurolinkState`` so all SSE consumers and the
    frontend DeviceStatusBar can display live signal quality.

    Fields
    ------
    frames_total       Total frames processed since last connect.
    frames_rejected    Artifact-rejected frames (Stages 3 / 3b).
    frames_clean       Frames that reached band-power computation.
    packet_loss_pct    Rolling 10-second BLE packet-loss estimate (%).
    last_frame_ts      Wall-clock time of the most recent frame (0 = never).
    avg_tick_ms        Exponential moving-average pipeline tick time (ms).
    """

    frames_total: int = 0
    frames_rejected: int = 0
    frames_clean: int = 0
    packet_loss_pct: float = 0.0
    last_frame_ts: float = 0.0
    avg_tick_ms: float = 0.0


# ============================================================================
# Ingest Payload (internal: EEGPump -> Hub)
# ============================================================================


class IngestPayload(BaseModel):
    """Internal payload passed from EEGPump to hub.update()."""

    source: str = "mock"
    address: str = ""
    timestamp: float = Field(default_factory=time.time)
    bands: BandPowers = Field(default_factory=BandPowers)
    poor_contact: bool = False
    contact_quality: float | None = None
    faa: float | None = None
    fmt: float | None = None
    focus_score: float = 0.0
    fatigue_score: float = 0.0
    ppg: PPGPayload | None = None
    breathing: BreathingPayload | None = None
    imu: IMUPayload | None = None
    fnirs_oxy: float | None = None
    fnirs_deoxy: float | None = None
    eeg_samples: list[list[float]] = Field(default_factory=list)
    bad_channels: list[str] = Field(default_factory=list)
    artifact_rejected: bool = False
    artifact_reasons: list[str] = Field(default_factory=list)
    artifact_annotations: list[ArtifactAnnotationPayload] = Field(default_factory=list)
    artifact_correction_plan: ArtifactCorrectionPlanPayload | None = None
    channel_impedances: dict[str, float] = Field(default_factory=dict)
    baseline_phase: str = "warmup"
    region: str = "A"
    alchemical_stage: str = "Nigredo"
    s_space: SSpaceCoords | None = None
    integration_coverage: float = 0.0
    engagement_index: float = 0.0
    stream_health: StreamHealthPayload | None = None


# ============================================================================
# NeurolinkState (Hub output / SSE payload)
# ============================================================================


class NeurolinkState(BaseModel):
    """Complete EEG state snapshot broadcast by hub and SSE stream."""

    connected: bool = False
    source: str = ""
    region: str = "A"
    alchemical_stage: str = "Nigredo"
    integration_coverage: float = 0.0
    engagement_index: float = 0.0
    bands: BandPowers = Field(default_factory=BandPowers)
    s_space: SSpaceCoords | None = None
    ea1: EA1Result = Field(default_factory=EA1Result)
    last_ts: float = 0.0
    frame_count: int = 0
    poor_contact: bool = False
    region_v01: str = "A"
    alchemical_stage_v01: str = "Nigredo"
    faa: float | None = None
    fmt: float | None = None
    hr_bpm: float | None = None
    hrv_rmssd: float | None = None
    rr_bpm: float | None = None
    pitch_deg: float | None = None
    roll_deg: float | None = None
    motion_rms: float | None = None
    contact_quality: float | None = None
    focus_state: str = "unknown"
    focus_score: float = 0.0
    fatigue_score: float = 0.0
    fnirs_oxy: float | None = None
    fnirs_deoxy: float | None = None
    eeg_samples: list[list[float]] = Field(default_factory=list)
    bad_channels: list[str] = Field(default_factory=list)
    artifact_rejected: bool = False
    artifact_reasons: list[str] = Field(default_factory=list)
    artifact_annotations: list[ArtifactAnnotationPayload] = Field(default_factory=list)
    artifact_correction_plan: ArtifactCorrectionPlanPayload | None = None
    channel_impedances: dict[str, float] = Field(default_factory=dict)
    baseline_phase: str = "warmup"
    # Stream quality metrics — populated by hub.update() from EEGPump.stream_health
    stream_health: StreamHealthPayload | None = None

    @property
    def band_powers(self) -> BandPowers:
        """Alias for bands (backward compatibility with tests)."""
        return self.bands


# ============================================================================
# API Request/Response models
# ============================================================================


class ConnectRequest(BaseModel):
    """POST /api/v1/neurolink/connect request body."""

    adapter_type: str = "mock"
    device_model: str = "muse_s_gen1"
    address: str | None = None


class ConnectResponse(BaseModel):
    ok: bool
    source: str
    message: str = ""


class DisconnectResponse(BaseModel):
    ok: bool


class BandPowerResponse(BaseModel):
    channel: str = "mean"
    alpha: float = 0.0
    theta: float = 0.0
    beta: float = 0.0
    delta: float = 0.0
    gamma: float = 0.0


class CalibrateResponse(BaseModel):
    status: str
    baseline_alpha: float | None = None


class BaselineProgressResponse(BaseModel):
    """Response for GET /api/v1/neurolink/baseline."""

    phase: str
    elapsed_s: float
    remaining_s: float
    total_s: float


class HealthResponse(BaseModel):
    status: str
    adapter_type: str
    adapter_connected: bool
    hub_frame_count: int
    redis: str
    db: str


class SessionSummary(BaseModel):
    id: int
    started_at: Any = None
    ended_at: Any = None
    device_model: str = ""
    adapter_type: str = ""
    frame_count: int = 0
    final_ea1_eligible: bool | None = None
