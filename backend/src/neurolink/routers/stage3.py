"""Stage 3 -- Artifact gate REST endpoints.

Mounted at /api/v1/stage3 by main.py.

Endpoints
---------
GET  /config   Return the active GateConfig thresholds.
POST /config   Replace/merge thresholds (all fields optional).
GET  /stats    Return running frame counters and rejection rate.
POST /reset    Reset frame counters (call at session start).
"""

from __future__ import annotations

from typing import Annotated

import structlog
from fastapi import APIRouter, Body, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from neurolink.dsp.artifact_config import (
    ARTIFACT_ACCEL_RMS_G,
    ARTIFACT_KURTOSIS_THRESHOLD,
    ARTIFACT_PK2PK_UV,
)
from neurolink.dsp.artifact_gate import ArtifactGate, GateConfig

log = structlog.get_logger(__name__)

router = APIRouter(prefix="/stage3", tags=["Stage3"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class GateConfigSchema(BaseModel):
    """All fields optional on POST -- send only what you want to change."""

    pk2pk_uv: float | None = Field(
        None,
        description=(
            f"Peak-to-peak amplitude threshold (uV). "
            f"Default={ARTIFACT_PK2PK_UV}. "
            "Frames with any EEG channel exceeding this are amplitude-rejected."
        ),
    )
    accel_rms_g: float | None = Field(
        None,
        description=(
            f"IMU accelerometer RMS threshold (g). "
            f"Default={ARTIFACT_ACCEL_RMS_G}. "
            "Frames where RMS motion exceeds this are motion-rejected."
        ),
    )
    kurtosis_threshold: float | None = Field(
        None,
        description=(
            f"Excess-kurtosis threshold (Fisher convention). "
            f"Default={ARTIFACT_KURTOSIS_THRESHOLD}. "
            "Values >threshold indicate EMG burst or electrode pop."
        ),
    )
    enable_amplitude: bool | None = Field(
        None, description="Enable/disable amplitude gate (Stage 3 pass 1)."
    )
    enable_imu: bool | None = Field(
        None, description="Enable/disable IMU motion gate (Stage 3 pass 2)."
    )
    enable_kurtosis: bool | None = Field(
        None, description="Enable/disable kurtosis burst gate (Stage 3 pass 3)."
    )


class GateConfigResponse(BaseModel):
    """Full GateConfig snapshot returned by GET /config and POST /config."""

    pk2pk_uv: float
    accel_rms_g: float
    kurtosis_threshold: float
    enable_amplitude: bool
    enable_imu: bool
    enable_kurtosis: bool


class GateStatsResponse(BaseModel):
    total_frames: int
    rejected_frames: int
    rejection_rate: float = Field(
        ..., description="Fraction of frames rejected since last reset (0-1)."
    )


# ---------------------------------------------------------------------------
# Dependency -- fetch gate from app.state
# ---------------------------------------------------------------------------


def _get_gate(request: Request) -> ArtifactGate:
    gate = getattr(request.app.state, "artifact_gate", None)
    if gate is None:
        raise HTTPException(
            status_code=503,
            detail="Stage3 artifact gate not initialised -- is the EEGPump running?",
        )
    return gate


GateDep = Annotated[ArtifactGate, Depends(_get_gate)]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _cfg_to_response(cfg: GateConfig) -> GateConfigResponse:
    pk2pk_uv: float = cfg.pk2pk_uv if cfg.pk2pk_uv is not None else ARTIFACT_PK2PK_UV
    return GateConfigResponse(
        pk2pk_uv=pk2pk_uv,
        accel_rms_g=cfg.accel_rms_g,
        kurtosis_threshold=cfg.kurtosis_threshold,
        enable_amplitude=cfg.enable_amplitude,
        enable_imu=cfg.enable_imu,
        enable_kurtosis=cfg.enable_kurtosis,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/config", response_model=GateConfigResponse)
async def get_config(gate: GateDep) -> GateConfigResponse:
    """Return the current Stage 3 artifact-gate thresholds."""
    return _cfg_to_response(gate.get_config())


@router.post("/config", response_model=GateConfigResponse)
async def update_config(
    gate: GateDep,
    body: Annotated[GateConfigSchema, Body()],
) -> GateConfigResponse:
    """Merge supplied thresholds into the live config."""
    current = gate.get_config()
    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    if not updates:
        return _cfg_to_response(current)

    new_cfg = GateConfig(
        pk2pk_uv=updates.get("pk2pk_uv", current.pk2pk_uv),
        accel_rms_g=updates.get("accel_rms_g", current.accel_rms_g),
        kurtosis_threshold=updates.get("kurtosis_threshold", current.kurtosis_threshold),
        enable_amplitude=updates.get("enable_amplitude", current.enable_amplitude),
        enable_imu=updates.get("enable_imu", current.enable_imu),
        enable_kurtosis=updates.get("enable_kurtosis", current.enable_kurtosis),
    )
    gate.set_config(new_cfg)
    log.info("stage3_config_via_api", updates=updates)
    return _cfg_to_response(new_cfg)


@router.get("/stats", response_model=GateStatsResponse)
async def get_stats(gate: GateDep) -> GateStatsResponse:
    """Return running frame counters and rejection rate since last reset."""
    s = gate.get_stats()
    return GateStatsResponse(
        total_frames=s["total_frames"],
        rejected_frames=s["rejected_frames"],
        rejection_rate=s["rejection_rate"],
    )


@router.post("/reset")
async def reset_stats(gate: GateDep) -> dict:
    """Reset frame counters. Call at the start of each session."""
    gate.reset_stats()
    log.info("stage3_stats_reset_via_api")
    return {"ok": True, "message": "Stage3 artifact gate counters reset"}
