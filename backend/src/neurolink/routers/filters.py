"""GET/PUT /api/v1/filters — runtime pipeline stage toggle API.

GET  /api/v1/filters          → {stage1_fir: bool, stage2_bad_channels: bool, ...}
PUT  /api/v1/filters          → body: partial or full FilterToggleConfig dict
                                 → returns updated state

Changes take effect on the very next EEGPump tick with no restart.
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from neurolink.dsp.filter_toggles import FilterToggleConfig, get_toggles, set_toggles

router = APIRouter(tags=["filters"])


class FilterToggleRequest(BaseModel):
    """All fields are optional — send only the keys you want to change."""

    stage1_fir: bool | None = None
    stage2_bad_channels: bool | None = None
    stage3_artifact_gate: bool | None = None
    stage4_asr: bool | None = None
    stage4b_baseline: bool | None = None
    stage5_ocular: bool | None = None
    imu_gate: bool | None = None


class FilterToggleResponse(BaseModel):
    stage1_fir: bool
    stage2_bad_channels: bool
    stage3_artifact_gate: bool
    stage4_asr: bool
    stage4b_baseline: bool
    stage5_ocular: bool
    imu_gate: bool


def _to_response(cfg: FilterToggleConfig) -> FilterToggleResponse:
    return FilterToggleResponse(**cfg.to_dict())


@router.get("/filters", response_model=FilterToggleResponse)
async def get_filters() -> FilterToggleResponse:
    """Return the current pipeline stage enable/disable state."""
    return _to_response(get_toggles())


@router.put("/filters", response_model=FilterToggleResponse)
async def update_filters(body: FilterToggleRequest) -> FilterToggleResponse:
    """Merge the supplied toggles into the live config.

    Send only the keys you want to change; omit the rest.
    Changes take effect on the next EEGPump tick.
    """
    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    new_cfg = set_toggles(updates)
    return _to_response(new_cfg)
