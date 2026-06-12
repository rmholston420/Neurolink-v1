"""Stage 3b — Artifact detector REST endpoints.

Mounted at /api/v1/stage3b by main.py.

Endpoints
---------
GET  /config   Return the active DetectorConfig thresholds.
PUT  /config   Replace/merge thresholds (all fields optional).
GET  /stats    Return per-type detection counters and rates.
POST /reset    Reset all counters (call at session start).

Design notes
------------
* The detector singleton lives on ``app.state.artifact_detector``
  (injected by main.py lifespan — identical pattern to the Stage 3
  artifact_gate and Stage 2 bad_channel_detector).
* All threshold defaults come from DetectorConfig; the router never
  hard-codes numerics.
* Config changes take effect on the very next EEGPump tick — no
  restart required, mirrors the pattern in stage3.py / filters.py.
* PUT /config is used instead of POST to signal idempotency —
  sending the same body twice produces the same result.
"""

from __future__ import annotations

from typing import Annotated

import structlog
from fastapi import APIRouter, Body, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from neurolink.dsp.artifact_config import (
    ARTIFACT_ACCEL_RMS_G,
    ARTIFACT_PK2PK_UV,
)
from neurolink.dsp.artifact_detector import ArtifactDetector, DetectorConfig

log = structlog.get_logger(__name__)

router = APIRouter(prefix="/stage3b", tags=["Stage3b"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class DetectorConfigSchema(BaseModel):
    """All fields optional on PUT — send only what you want to change."""

    # Blink
    blink_frontal_uv: float | None = Field(
        None,
        description=(
            f"Frontal pk2pk threshold for blink detection (µV). "
            f"Default={round(ARTIFACT_PK2PK_UV * 0.80, 1)}."
        ),
    )
    blink_freq_hz_max: float | None = Field(
        None,
        description="Blink energy concentrated below this frequency (Hz). Default=10.0.",
    )
    blink_frontal_ratio: float | None = Field(
        None,
        description=(
            "Frontal pk2pk must be >= this multiple of temporal pk2pk. "
            "Default=2.0."
        ),
    )

    # Horizontal EOG
    heog_asymmetry_uv: float | None = Field(
        None,
        description="|AF7 mean − AF8 mean| threshold for saccade detection (µV). Default=30.0.",
    )
    heog_freq_hz_max: float | None = Field(
        None,
        description="Saccade energy concentrated below this frequency (Hz). Default=4.0.",
    )

    # EMG
    emg_hf_ratio: float | None = Field(
        None,
        description="HF (30–100 Hz) / broadband power ratio threshold for muscle noise. Default=0.30.",
    )
    emg_freq_low_hz: float | None = Field(
        None,
        description="Lower bound of EMG detection band (Hz). Default=30.0.",
    )
    emg_freq_high_hz: float | None = Field(
        None,
        description="Upper bound of EMG detection band (Hz). Default=100.0.",
    )

    # Cardiac
    cardiac_freq_low_hz: float | None = Field(
        None,
        description="Lower bound of cardiac band (Hz). Default=0.8.",
    )
    cardiac_freq_high_hz: float | None = Field(
        None,
        description="Upper bound of cardiac band (Hz). Default=1.8.",
    )
    cardiac_temporal_uv: float | None = Field(
        None,
        description="pk2pk amplitude threshold at temporal channels for cardiac detection (µV). Default=15.0.",
    )

    # Electrode pop
    pop_step_uv: float | None = Field(
        None,
        description="Single-sample step-change threshold for electrode pop detection (µV). Default=60.0.",
    )
    pop_isolation_ratio: float | None = Field(
        None,
        description=(
            "Affected channel pk2pk / median(others) ratio must exceed this "
            "to qualify as an isolated pop. Default=3.0."
        ),
    )

    # Line noise
    line_freq_hz: float | None = Field(
        None,
        description="Nominal power-line frequency (Hz). Use 50.0 for EU/Asia, 60.0 for US/CA/MX/JP. Default=60.0.",
    )
    line_band_hz: float | None = Field(
        None,
        description="±bandwidth around line_freq_hz used to measure line-noise power (Hz). Default=2.0.",
    )
    line_power_ratio: float | None = Field(
        None,
        description="Line-band / broadband power ratio threshold. Default=0.15.",
    )

    # Motion
    motion_accel_rms_g: float | None = Field(
        None,
        description=f"Accelerometer RMS threshold (g). Default={ARTIFACT_ACCEL_RMS_G}.",
    )

    # Feature enable switches
    enable_blink: bool | None = Field(None, description="Enable/disable blink detector.")
    enable_heog: bool | None = Field(None, description="Enable/disable horizontal EOG detector.")
    enable_emg: bool | None = Field(None, description="Enable/disable EMG detector.")
    enable_cardiac: bool | None = Field(None, description="Enable/disable cardiac detector.")
    enable_electrode_pop: bool | None = Field(None, description="Enable/disable electrode-pop detector.")
    enable_line_noise: bool | None = Field(None, description="Enable/disable line-noise detector.")
    enable_motion: bool | None = Field(None, description="Enable/disable IMU motion detector.")


class DetectorConfigResponse(BaseModel):
    """Full DetectorConfig snapshot returned by GET /config and PUT /config."""

    blink_frontal_uv: float
    blink_freq_hz_max: float
    blink_frontal_ratio: float
    heog_asymmetry_uv: float
    heog_freq_hz_max: float
    emg_hf_ratio: float
    emg_freq_low_hz: float
    emg_freq_high_hz: float
    cardiac_freq_low_hz: float
    cardiac_freq_high_hz: float
    cardiac_temporal_uv: float
    pop_step_uv: float
    pop_isolation_ratio: float
    line_freq_hz: float
    line_band_hz: float
    line_power_ratio: float
    motion_accel_rms_g: float
    enable_blink: bool
    enable_heog: bool
    enable_emg: bool
    enable_cardiac: bool
    enable_electrode_pop: bool
    enable_line_noise: bool
    enable_motion: bool


class ArtifactTypeStats(BaseModel):
    count: int
    rate: float = Field(..., description="Fraction of frames where this artifact type was detected (0–1).")


class DetectorStatsResponse(BaseModel):
    total_frames: int
    artifact_types: dict[str, ArtifactTypeStats] = Field(
        ...,
        description=(
            "Per-type detection counts and rates since last reset. "
            "Keys are ArtifactType enum names: BLINK, HORIZONTAL_EOG, "
            "EMG, CARDIAC, ELECTRODE_POP, LINE_NOISE, MOTION."
        ),
    )


# ---------------------------------------------------------------------------
# Dependency — fetch detector from app.state
# ---------------------------------------------------------------------------

def _get_detector(request: Request) -> ArtifactDetector:
    detector = getattr(request.app.state, "artifact_detector", None)
    if detector is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Stage3b artifact detector not initialised — "
                "is the EEGPump running?"
            ),
        )
    return detector


DetectorDep = Annotated[ArtifactDetector, Depends(_get_detector)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg_to_response(cfg: DetectorConfig) -> DetectorConfigResponse:
    return DetectorConfigResponse(
        blink_frontal_uv=cfg.blink_frontal_uv,
        blink_freq_hz_max=cfg.blink_freq_hz_max,
        blink_frontal_ratio=cfg.blink_frontal_ratio,
        heog_asymmetry_uv=cfg.heog_asymmetry_uv,
        heog_freq_hz_max=cfg.heog_freq_hz_max,
        emg_hf_ratio=cfg.emg_hf_ratio,
        emg_freq_low_hz=cfg.emg_freq_low_hz,
        emg_freq_high_hz=cfg.emg_freq_high_hz,
        cardiac_freq_low_hz=cfg.cardiac_freq_low_hz,
        cardiac_freq_high_hz=cfg.cardiac_freq_high_hz,
        cardiac_temporal_uv=cfg.cardiac_temporal_uv,
        pop_step_uv=cfg.pop_step_uv,
        pop_isolation_ratio=cfg.pop_isolation_ratio,
        line_freq_hz=cfg.line_freq_hz,
        line_band_hz=cfg.line_band_hz,
        line_power_ratio=cfg.line_power_ratio,
        motion_accel_rms_g=cfg.motion_accel_rms_g,
        enable_blink=cfg.enable_blink,
        enable_heog=cfg.enable_heog,
        enable_emg=cfg.enable_emg,
        enable_cardiac=cfg.enable_cardiac,
        enable_electrode_pop=cfg.enable_electrode_pop,
        enable_line_noise=cfg.enable_line_noise,
        enable_motion=cfg.enable_motion,
    )


def _stats_to_response(raw: dict) -> DetectorStatsResponse:
    return DetectorStatsResponse(
        total_frames=raw["total_frames"],
        artifact_types={
            name: ArtifactTypeStats(count=v["count"], rate=v["rate"])
            for name, v in raw["artifact_types"].items()
        },
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get(
    "/config",
    response_model=DetectorConfigResponse,
    summary="Get Stage3b detector thresholds",
)
async def get_config(detector: DetectorDep) -> DetectorConfigResponse:
    """Return the current Stage 3b artifact-detector thresholds.

    All values are live — changes made via PUT /config are reflected
    here immediately.
    """
    return _cfg_to_response(detector.get_config())


@router.put(
    "/config",
    response_model=DetectorConfigResponse,
    summary="Update Stage3b detector thresholds",
)
async def update_config(
    detector: DetectorDep,
    body: Annotated[DetectorConfigSchema, Body()],
) -> DetectorConfigResponse:
    """Merge supplied thresholds into the live detector config.

    Send only the fields you want to change; omit the rest.
    Changes take effect on the next EEGPump tick — no restart required.

    Typical use-cases
    -----------------
    * Raise ``blink_frontal_uv`` for a subject who naturally has large
      frontal alpha to reduce false-positive blink detections.
    * Lower ``emg_hf_ratio`` in a noisy environment to be more
      aggressive about rejecting muscle contamination.
    * Disable ``enable_cardiac`` if no temporal channels are connected.
    * Switch ``line_freq_hz`` to 50.0 when deploying in Europe / Asia.
    """
    current = detector.get_config()
    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    if not updates:
        return _cfg_to_response(current)

    new_cfg = DetectorConfig(
        blink_frontal_uv=updates.get("blink_frontal_uv", current.blink_frontal_uv),
        blink_freq_hz_max=updates.get("blink_freq_hz_max", current.blink_freq_hz_max),
        blink_frontal_ratio=updates.get("blink_frontal_ratio", current.blink_frontal_ratio),
        heog_asymmetry_uv=updates.get("heog_asymmetry_uv", current.heog_asymmetry_uv),
        heog_freq_hz_max=updates.get("heog_freq_hz_max", current.heog_freq_hz_max),
        emg_hf_ratio=updates.get("emg_hf_ratio", current.emg_hf_ratio),
        emg_freq_low_hz=updates.get("emg_freq_low_hz", current.emg_freq_low_hz),
        emg_freq_high_hz=updates.get("emg_freq_high_hz", current.emg_freq_high_hz),
        cardiac_freq_low_hz=updates.get("cardiac_freq_low_hz", current.cardiac_freq_low_hz),
        cardiac_freq_high_hz=updates.get("cardiac_freq_high_hz", current.cardiac_freq_high_hz),
        cardiac_temporal_uv=updates.get("cardiac_temporal_uv", current.cardiac_temporal_uv),
        pop_step_uv=updates.get("pop_step_uv", current.pop_step_uv),
        pop_isolation_ratio=updates.get("pop_isolation_ratio", current.pop_isolation_ratio),
        line_freq_hz=updates.get("line_freq_hz", current.line_freq_hz),
        line_band_hz=updates.get("line_band_hz", current.line_band_hz),
        line_power_ratio=updates.get("line_power_ratio", current.line_power_ratio),
        motion_accel_rms_g=updates.get("motion_accel_rms_g", current.motion_accel_rms_g),
        enable_blink=updates.get("enable_blink", current.enable_blink),
        enable_heog=updates.get("enable_heog", current.enable_heog),
        enable_emg=updates.get("enable_emg", current.enable_emg),
        enable_cardiac=updates.get("enable_cardiac", current.enable_cardiac),
        enable_electrode_pop=updates.get("enable_electrode_pop", current.enable_electrode_pop),
        enable_line_noise=updates.get("enable_line_noise", current.enable_line_noise),
        enable_motion=updates.get("enable_motion", current.enable_motion),
    )
    detector.set_config(new_cfg)
    log.info("stage3b_config_via_api", updates=updates)
    return _cfg_to_response(new_cfg)


@router.get(
    "/stats",
    response_model=DetectorStatsResponse,
    summary="Get per-type artifact detection stats",
)
async def get_stats(detector: DetectorDep) -> DetectorStatsResponse:
    """Return per-type detection counters and rates since last reset.

    ``rate`` for each type is ``count / total_frames`` rounded to 4
    decimal places.  A high BLINK rate (> 0.05) during a session
    suggests the subject is fatigued or the blink threshold is too
    low.  A high EMG rate suggests jaw clenching or cable noise.
    """
    return _stats_to_response(detector.get_stats())


@router.post(
    "/reset",
    summary="Reset artifact detection counters",
)
async def reset_stats(detector: DetectorDep) -> dict:
    """Reset all per-type counters to zero.

    Call this at the start of each session so that GET /stats
    reflects activity for the current session only.
    """
    detector.reset_stats()
    log.info("stage3b_stats_reset_via_api")
    return {"ok": True, "message": "Stage3b artifact detector counters reset"}
