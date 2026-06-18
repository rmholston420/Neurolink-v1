"""Stage 2 -- Bad channel detection REST endpoints.

Mounted at /api/v1/stage2 by main.py.

Endpoints
---------
GET  /status         Per-channel live snapshot (variance, PSD, flags).
POST /mark           Manually flag or clear a channel as bad.
GET  /config         Return active detector thresholds.
POST /config         Replace detector thresholds.
POST /reset          Reset all running stats (call at session start).
"""

from __future__ import annotations

from typing import Annotated

import structlog
from fastapi import APIRouter, Body, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from neurolink.dsp.bad_channels import BadChannelDetector, DetectorConfig

log = structlog.get_logger(__name__)

router = APIRouter(prefix="/stage2", tags=["Stage2"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class ChannelStatusSchema(BaseModel):
    name: str
    ema_variance: float
    ema_mean_psd: float
    flat_line: bool
    noisy: bool
    manual_bad: bool
    is_bad: bool
    reason: str


class Stage2StatusResponse(BaseModel):
    channels: list[ChannelStatusSchema]
    bad_channels: list[str]


class MarkChannelRequest(BaseModel):
    channel: str = Field(..., description="Channel name: TP9 | AF7 | AF8 | TP10 | AUX")
    bad: bool = Field(..., description="True to flag bad, False to clear")


class DetectorConfigSchema(BaseModel):
    var_threshold: float = Field(0.01, description="Flat-line threshold uV^2")
    psd_ratio_threshold: float = Field(5.0, description="Noisy channel: PSD > N x median")
    ema_alpha: float = Field(0.1, description="EMA smoothing factor")
    fs: float = Field(256.0, description="Sampling rate Hz")
    nperseg: int = Field(128, description="Welch PSD nperseg")


# ---------------------------------------------------------------------------
# Dependency
# ---------------------------------------------------------------------------


def _get_detector(request: Request) -> BadChannelDetector:
    detector = getattr(request.app.state, "bad_channel_detector", None)
    if detector is None:
        raise HTTPException(status_code=503, detail="Stage2 detector not initialised")
    return detector


DetectorDep = Annotated[BadChannelDetector, Depends(_get_detector)]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/status", response_model=Stage2StatusResponse)
async def get_status(detector: DetectorDep) -> Stage2StatusResponse:
    """Return live per-channel stats and current bad-channel list."""
    stats = detector.get_stats()
    channels = [
        ChannelStatusSchema(
            name=s.name,
            ema_variance=round(s.ema_variance, 6),
            ema_mean_psd=round(s.ema_mean_psd, 6),
            flat_line=s.flat_line,
            noisy=s.noisy,
            manual_bad=s.manual_bad,
            is_bad=s.is_bad,
            reason=s.reason(),
        )
        for s in stats
    ]
    bad = [c.name for c in channels if c.is_bad]
    return Stage2StatusResponse(channels=channels, bad_channels=bad)


@router.post("/mark", response_model=Stage2StatusResponse)
async def mark_channel(
    detector: DetectorDep,
    body: Annotated[MarkChannelRequest, Body()],
) -> Stage2StatusResponse:
    """Manually flag or clear a channel."""
    try:
        detector.set_manual_bad(body.channel, body.bad)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return await get_status(detector)


@router.get("/config", response_model=DetectorConfigSchema)
async def get_config(detector: DetectorDep) -> DetectorConfigSchema:
    cfg = detector.get_config()
    return DetectorConfigSchema(
        var_threshold=cfg.var_threshold,
        psd_ratio_threshold=cfg.psd_ratio_threshold,
        ema_alpha=cfg.ema_alpha,
        fs=cfg.fs,
        nperseg=cfg.nperseg,
    )


@router.post("/config", response_model=DetectorConfigSchema)
async def set_config(
    detector: DetectorDep,
    body: Annotated[DetectorConfigSchema, Body()],
) -> DetectorConfigSchema:
    """Update detector thresholds."""
    cfg = DetectorConfig(
        var_threshold=body.var_threshold,
        psd_ratio_threshold=body.psd_ratio_threshold,
        ema_alpha=body.ema_alpha,
        fs=body.fs,
        nperseg=body.nperseg,
    )
    detector.set_config(cfg)
    return await get_config(detector)


@router.post("/reset")
async def reset_detector(detector: DetectorDep) -> dict:
    """Reset all running stats. Call at the start of each session."""
    detector.reset()
    return {"ok": True, "message": "Stage2 detector reset"}
