"""Calibration endpoint.

POST /api/v1/neurolink/calibrate — start 30-second baseline alpha capture.
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter

from neurolink.dependencies import ServiceDep
from neurolink.models.eeg import CalibrateResponse

log = structlog.get_logger(__name__)

router = APIRouter(prefix="/neurolink", tags=["calibration"])


@router.post("/calibrate", response_model=CalibrateResponse)
async def start_calibration(service: ServiceDep) -> CalibrateResponse:
    """Start a 30-second personal alpha baseline calibration session.

    Returns immediately with {"status": "started"}.
    Calibration runs in the background; hub.baseline_alpha is updated on completion.
    """
    return await service.start_calibration()
