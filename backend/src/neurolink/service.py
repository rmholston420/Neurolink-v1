"""NeuroLinkService — async business logic layer.

All business logic goes here. Routers call only NeuroLinkService methods.
"""
from __future__ import annotations

import asyncio
from typing import Any

import structlog

from neurolink.adapter_factory import create_adapter
from neurolink.calibration import CalibrationSession
from neurolink.eeg_pump import EEGPump
from neurolink.exceptions import AdapterNotConnectedError, CalibrationBusyError
from neurolink.hub import EEGHub
from neurolink.models.eeg import (
    BandPowerResponse,
    CalibrateResponse,
    ConnectRequest,
    ConnectResponse,
    DisconnectResponse,
    EA1Result,
    NeurolinkState,
)

log = structlog.get_logger(__name__)


class NeuroLinkService:
    """Encapsulates all Neurolink business logic."""

    def __init__(self, hub: EEGHub) -> None:
        self._hub = hub
        self._adapter = None  # type: ignore[assignment]
        self._pump: EEGPump | None = None
        self._calibration: CalibrationSession | None = None
        self._session_id: int | None = None
        self._session_repo = None  # type: ignore[assignment] — injected on connect

    async def connect(
        self,
        request: ConnectRequest,
        session_repo=None,  # type: ignore[type-arg]
    ) -> ConnectResponse:
        """Connect to the EEG adapter and start the pump."""
        # Disconnect any existing adapter
        if self._adapter is not None and self._adapter.is_connected:
            await self.disconnect()

        adapter = create_adapter(
            adapter_type=request.adapter_type,
            device_model=request.device_model,
            address=request.address,
        )
        await adapter.connect()
        self._adapter = adapter
        self._pump = EEGPump(adapter, self._hub)
        await self._pump.start()

        # Session log
        if session_repo is not None:
            try:
                entry = await session_repo.create_session(
                    device_model=request.device_model,
                    adapter_type=request.adapter_type,
                    address=request.address,
                )
                self._session_id = entry.id
                self._session_repo = session_repo
            except Exception as exc:
                log.warning("session_log_create_failed", error=str(exc))

        log.info(
            "neurolink_connected",
            adapter_type=request.adapter_type,
            device_model=request.device_model,
        )
        return ConnectResponse(
            ok=True,
            source=adapter.source_name,
            message=f"Connected via {request.adapter_type}",
        )

    async def disconnect(self) -> DisconnectResponse:
        """Stop the pump and disconnect the adapter."""
        if self._pump is not None:
            await self._pump.stop()
            self._pump = None

        if self._adapter is not None:
            state = self._hub.get_state()
            try:
                await self._adapter.disconnect()
            except Exception as exc:
                log.warning("adapter_disconnect_error", error=str(exc))

            # Session log
            if self._session_repo is not None and self._session_id is not None:
                try:
                    await self._session_repo.end_session(
                        session_id=self._session_id,
                        frame_count=state.frame_count,
                        final_region=state.region,
                        final_stage=state.alchemical_stage,
                        final_ea1_eligible=state.ea1.eligible,
                    )
                except Exception as exc:
                    log.warning("session_log_end_failed", error=str(exc))

            self._adapter = None
            self._session_id = None
            self._session_repo = None

        log.info("neurolink_disconnected")
        return DisconnectResponse(ok=True)

    async def get_current_state(self) -> NeurolinkState:
        """Return the current hub state."""
        return self._hub.get_state()

    async def get_band_powers(self, channel: str = "all") -> BandPowerResponse:
        """Return band powers for a specific channel or average."""
        state = self._hub.get_state()
        if not state.connected:
            return BandPowerResponse(channel=channel, error="not_connected")
        bands = state.bands
        return BandPowerResponse(
            channel=channel,
            alpha=bands.alpha,
            theta=bands.theta,
            beta=bands.beta,
            delta=bands.delta,
            gamma=bands.gamma,
        )

    async def get_ea1(self) -> EA1Result:
        """Return the latest EA-1 result."""
        return self._hub.get_ea1()

    async def start_calibration(self) -> CalibrateResponse:
        """Start a 30-second calibration session."""
        if self._calibration is not None and self._calibration.is_running:
            raise CalibrationBusyError("Calibration already running")
        self._calibration = CalibrationSession(self._hub)
        await self._calibration.start()
        return CalibrateResponse(status="started", baseline_alpha=None)

    @property
    def adapter_connected(self) -> bool:
        """Return True if the adapter is active."""
        return self._adapter is not None and self._adapter.is_connected

    @property
    def adapter_type(self) -> str:
        """Return the current adapter type string."""
        if self._adapter is None:
            from neurolink.config import get_settings
            return get_settings().adapter_type
        return self._adapter.source_name
