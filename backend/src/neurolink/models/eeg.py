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
    """Single artifact annotation produced by Stage 3b ArtifactDetector.

    Mirrors ``neurolink.dsp.artifact_detector.ArtifactAnnotation`` as a
    serialisable Pydantic model so the result can travel through
    IngestPayload → NeurolinkState → SSE stream to the frontend.
    """

    artifact_type: str        # ArtifactType.name  e.g. "BLINK", "EMG"
    confidence: float         # 0.0–1.0
    channels: list[str]       # channel names where artifact was detected
    feature_value: float      # raw feature value that triggered detection
    feature_name: str         # human-readable feature label
    threshold: float          # the threshold that was exceeded


class ArtifactCorrectionPlanPayload(BaseModel):
    """Serialisable snapshot of the CorrectionPlan built by Stage 3b.

    Forwarded through IngestPayload → NeurolinkState → SSE stream so the
    frontend and monitoring tools can see which correctors were invoked
    on each frame without needing to parse artifact_annotations.
    """

    hard_reject: bool = False
    apply_ocular_regression: bool = False
    apply_asr: bool = False
    apply_notch: bool = False
    apply_cardiac_regression: bool = False


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
    ppg: PPGPayload | None = None
    breathing: BreathingPayload | None = None
    imu: IMUPayload | None = None
    fnirs_oxy: float | None = None
    fnirs_deoxy: float | None = None
    # Raw EEG sample window: list of channels, each a list of float samples
    # Shape: [n_channels][n_samples]  e.g. [[...64 floats...], ...] x5
    eeg_samples: list[list[float]] = Field(default_factory=list)
    # Stage 2: channels detected or manually flagged as bad this frame
    bad_channels: list[str] = Field(default_factory=list)
    # Stage 3: epoch-level artifact gate decision
    artifact_rejected: bool = False
    artifact_reasons: list[str] = Field(default_factory=list)
    # Stage 3b: multi-type artifact classifier results
    # Empty list when Stage 3b is disabled or no artifacts detected.
    artifact_annotations: list[ArtifactAnnotationPayload] = Field(default_factory=list)
    # Stage 3b: correction plan built from classifier results.
    # None when Stage 3b is disabled.
    artifact_correction_plan: ArtifactCorrectionPlanPayload | None = None
    # Per-channel impedance in kΩ. Only hardware adapters that expose electrode
    # impedance need to populate this; defaults to empty dict.
    channel_impedances: dict[str, float] = Field(default_factory=dict)
    # Baseline phase: "warmup" | "recording" | "complete"
    # Set by EEGPump from BaselineRecorder.phase every tick.
    # Lets the frontend know which phase of the startup baseline is active
    # and when to expect the bell event.
    baseline_phase: str = "warmup"
    # Filled by hub.update()
    region: str = "A"
    alchemical_stage: str = "Nigredo"
    s_space: SSpaceCoords | None = None
    integration_coverage: float = 0.0
    engagement_index: float = 0.0


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
    # Raw EEG sample window forwarded verbatim from IngestPayload.
    eeg_samples: list[list[float]] = Field(default_factory=list)
    # Stage 2: bad channels detected this frame
    bad_channels: list[str] = Field(default_factory=list)
    # Stage 3: epoch-level artifact gate
    artifact_rejected: bool = False
    artifact_reasons: list[str] = Field(default_factory=list)
    # Stage 3b: multi-type artifact classifier results forwarded from IngestPayload.
    # Empty list when Stage 3b is disabled or no artifacts detected.
    artifact_annotations: list[ArtifactAnnotationPayload] = Field(default_factory=list)
    # Stage 3b: correction plan forwarded from IngestPayload.
    # None when Stage 3b is disabled.
    artifact_correction_plan: ArtifactCorrectionPlanPayload | None = None
    # Per-channel impedance in kΩ, forwarded verbatim from IngestPayload.
    channel_impedances: dict[str, float] = Field(default_factory=dict)
    # Baseline phase forwarded verbatim from IngestPayload.
    # "warmup" | "recording" | "complete"
    baseline_phase: str = "warmup"


# ============================================================================
# API Request/Response models
# ============================================================================


class ConnectRequest(BaseModel):
    """POST /api/v1/neurolink/connect request body."""

    adapter_type: str = "mock"  # mock | ble | lsl
    device_model: str = "muse_s_gen1"  # muse_s_gen1 | muse_s_athena | mock
    address: str | None = None  # BLE MAC address (required for BLE mode)


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
    status: str  # "started" | "complete" | "error"
    baseline_alpha: float | None = None


class BaselineProgressResponse(BaseModel):
    """Response for GET /api/v1/neurolink/baseline.

    Lightweight polling alternative to the SSE stream for clients that
    cannot use server-sent events (e.g. simple HTTP clients, React Native
    fetch pollers, CLI tools).

    Fields
    ------
    phase
        Current phase of the baseline calibration session.
        One of: ``"idle"`` | ``"warmup"`` | ``"baseline"`` | ``"complete"``.
        ``"idle"`` means no session has been started yet.
    elapsed_s
        Seconds elapsed since the session started.
        0.0 when phase is ``"idle"``.
    remaining_s
        Seconds remaining until the session completes.
        Computed as ``max(0, TOTAL_DURATION_SEC - elapsed_s)``.
        0.0 when phase is ``"idle"`` or ``"complete"``.
    total_s
        Total session duration in seconds (always 90.0).
    """

    phase: str
    elapsed_s: float
    remaining_s: float
    total_s: float


class HealthResponse(BaseModel):
    status: str  # "ok" | "degraded"
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
