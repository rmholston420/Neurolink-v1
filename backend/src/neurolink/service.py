"""NeuroLinkService — async business logic layer.

All router handlers delegate here. Never instantiate adapters or hub in routers.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator

import structlog

from neurolink.calibration import TOTAL_DURATION_SEC, CalibrationSession
from neurolink.eeg_pump import EEGPump
from neurolink.exceptions import AdapterAlreadyConnectedError, AdapterNotConnectedError
from neurolink.hardware.base import HardwareAdapter
from neurolink.hub import EEGHub
from neurolink.models.eeg import (
    BandPowerResponse,
    BaselineProgressResponse,
    CalibrateResponse,
    ConnectResponse,
    DisconnectResponse,
    EA1Result,
    NeurolinkState,
    SessionSummary,
)

log = structlog.get_logger(__name__)


class NeuroLinkService:
    """Orchestrates adapter lifecycle, pump, calibration, and state access."""

    def __init__(self, hub: EEGHub) -> None:
        self._hub = hub
        self._adapter: HardwareAdapter | None = None
        self._pump: EEGPump | None = None
        self._calibration_task: asyncio.Task | None = None
        # Kept alive so get_baseline_progress() can read phase/elapsed after
        # start_calibration() returns.
        self._calibration_session: CalibrationSession | None = None
        self._db_session_id: int | None = None
        self._db_session_factory = None
        self._adapter_type: str = "mock"
        self._device_model: str = "mock"

    def set_db_session_factory(self, factory) -> None:
        """Inject DB session factory (called from lifespan)."""
        self._db_session_factory = factory

    async def connect(
        self,
        adapter_type: str = "mock",
        device_model: str = "muse_s_gen1",
        address: str | None = None,
    ) -> ConnectResponse:
        """Connect adapter and start EEG pump."""
        if self._adapter is not None and self._adapter.is_connected:
            raise AdapterAlreadyConnectedError("Adapter already connected. Disconnect first.")

        from neurolink.adapter_factory import create_adapter
        from neurolink.config import get_settings

        settings = get_settings()
        self._adapter_type = adapter_type
        self._device_model = device_model

        self._adapter = create_adapter(
            adapter_type=adapter_type,
            device_model=device_model,
            address=address,
        )

        await self._adapter.connect()
        log.info(
            "neurolink_adapter_connected",
            adapter_type=adapter_type,
            device_model=device_model,
        )

        self._pump = EEGPump(self._adapter, self._hub, publish_hz=settings.publish_hz)
        await self._pump.start()

        await self._create_db_session(adapter_type, device_model, address)

        return ConnectResponse(
            ok=True,
            source=self._adapter.source_name,
            message=f"Connected via {adapter_type} ({device_model})",
        )

    async def disconnect(self) -> DisconnectResponse:
        """Stop pump and disconnect adapter.

        Calls pump.reset() before stop() so that BaselineRecorder is
        returned to WARMUP and hub state is cleared atomically.  A
        subsequent connect() will therefore always start a fresh 150 s
        baseline window rather than inheriting a stale COMPLETE phase
        from the previous session.
        """
        if self._pump:
            self._pump.reset()        # resets BaselineRecorder + hub
            await self._pump.stop()
            self._pump = None

        if self._adapter:
            try:
                await self._adapter.disconnect()
            except Exception as exc:
                log.warning("disconnect_error", error=str(exc))
            finally:
                self._adapter = None

        await self._close_db_session()
        # hub.reset() is now handled inside pump.reset() above.
        # Call it here only as a safety net when there is no pump
        # (e.g. connect() failed before pump was created).
        if self._pump is None and self._adapter is None:
            # pump was None from the start (no-pump disconnect path)
            pass  # hub was already reset by pump.reset() if pump existed

        log.info("neurolink_disconnected")
        return DisconnectResponse(ok=True)

    async def get_current_state(self) -> NeurolinkState:
        """Return current NeurolinkState from hub."""
        return self._hub.get_state()

    async def get_band_powers(self, channel: str = "mean") -> BandPowerResponse:
        """Return band powers for a specific channel (or mean)."""
        state = self._hub.get_state()
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
        """Return latest EA-1 eligibility result."""
        return self._hub.get_ea1()

    async def start_calibration(self) -> CalibrateResponse:
        """Start a 90-second alpha baseline calibration session."""
        if self._adapter is None or not self._adapter.is_connected:
            raise AdapterNotConnectedError("Cannot calibrate: no adapter connected.")

        if self._calibration_task and not self._calibration_task.done():
            # Calibration already running — return started status idempotently
            return CalibrateResponse(status="started", baseline_alpha=None)

        cal_session = CalibrationSession(self._adapter, self._hub)
        # Store on self so get_baseline_progress() can read phase/elapsed.
        self._calibration_session = cal_session

        async def _run_calibration():
            await cal_session.run()

        self._calibration_task = asyncio.create_task(_run_calibration())
        log.info("calibration_task_started")
        return CalibrateResponse(status="started", baseline_alpha=None)

    def get_baseline_progress(self) -> BaselineProgressResponse:
        """Return the current calibration progress without opening a stream.

        Designed for clients that poll rather than consume SSE.  Reads
        ``phase`` and ``elapsed`` directly from the live
        ``CalibrationSession`` instance (updated every loop tick by
        ``CalibrationSession.run()``).

        Returns a snapshot with ``phase="idle"`` and zero timings when
        no session has been started yet, or ``phase="complete"`` once the
        background task has finished.
        """
        sess = self._calibration_session

        if sess is None:
            return BaselineProgressResponse(
                phase="idle",
                elapsed_s=0.0,
                remaining_s=0.0,
                total_s=TOTAL_DURATION_SEC,
            )

        elapsed = sess.elapsed
        remaining = max(0.0, TOTAL_DURATION_SEC - elapsed)

        # If the task finished and the session's own phase hasn't been
        # set to "complete" yet (race between task completion and the
        # loop's last tick), clamp remaining to 0 and reflect completion.
        task = self._calibration_task
        if task is not None and task.done():
            remaining = 0.0
            phase = "complete"
        else:
            phase = sess.phase

        return BaselineProgressResponse(
            phase=phase,
            elapsed_s=round(elapsed, 2),
            remaining_s=round(remaining, 2),
            total_s=TOTAL_DURATION_SEC,
        )

    async def stream_state(self) -> AsyncGenerator[NeurolinkState]:
        """Async generator that yields NeurolinkState at hub fan-out events."""
        q: asyncio.Queue = asyncio.Queue(maxsize=32)
        self._hub.register_sse_queue(q)
        try:
            while True:
                state = await asyncio.wait_for(q.get(), timeout=2.0)
                yield state
        except TimeoutError:
            yield self._hub.get_state()
        except asyncio.CancelledError:
            pass
        finally:
            self._hub.unregister_sse_queue(q)

    async def get_sessions(self, limit: int = 20) -> list[SessionSummary]:
        """Return recent session log entries."""
        if self._db_session_factory is None:
            return []
        from neurolink.db.repository import SessionLogRepository

        async with self._db_session_factory() as db:
            repo = SessionLogRepository(db)
            sessions = await repo.list_recent(limit=limit)
        return [
            SessionSummary(
                id=s.id,
                started_at=s.started_at,
                ended_at=s.ended_at,
                device_model=s.device_model,
                adapter_type=s.adapter_type,
                frame_count=s.frame_count,
                final_ea1_eligible=s.final_ea1_eligible,
            )
            for s in sessions
        ]

    @property
    def is_connected(self) -> bool:
        return self._adapter is not None and self._adapter.is_connected

    @property
    def adapter_type(self) -> str:
        return self._adapter_type

    async def _create_db_session(
        self, adapter_type: str, device_model: str, address: str | None
    ) -> None:
        """Create a session log entry in the DB."""
        if self._db_session_factory is None:
            return
        try:
            from neurolink.db.repository import SessionLogRepository

            async with self._db_session_factory() as db:
                repo = SessionLogRepository(db)
                entry = await repo.create_session(
                    device_model=device_model,
                    adapter_type=adapter_type,
                    address=address,
                )
                self._db_session_id = entry.id
        except Exception as exc:
            log.warning("db_create_session_error", error=str(exc))

    async def _close_db_session(self) -> None:
        """Close and update the current session log entry."""
        if self._db_session_factory is None or self._db_session_id is None:
            return
        try:
            state = self._hub.get_state()
            from neurolink.db.repository import SessionLogRepository

            async with self._db_session_factory() as db:
                repo = SessionLogRepository(db)
                await repo.end_session(
                    session_id=self._db_session_id,
                    frame_count=state.frame_count,
                    final_region=state.region,
                    final_stage=state.alchemical_stage,
                    final_ea1_eligible=state.ea1.eligible,
                )
            self._db_session_id = None
        except Exception as exc:
            log.warning("db_close_session_error", error=str(exc))
