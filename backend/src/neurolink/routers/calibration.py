"""Calibration router."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from neurolink.dependencies import get_neurolink_service
from neurolink.exceptions import CalibrationBusyError
from neurolink.models.eeg import CalibrateResponse
from neurolink.service import NeuroLinkService

router = APIRouter(prefix="/api/v1/neurolink", tags=["calibration"])


@router.post("/calibrate", response_model=CalibrateResponse)
async def calibrate(
    service: NeuroLinkService = Depends(get_neurolink_service),
) -> CalibrateResponse:
    """Start a 30-second alpha baseline calibration session."""
    try:
        return await service.start_calibration()
    except CalibrationBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
