"""Calibration session: 30-second baseline alpha capture.

Ported from Rigpa-v2 calibration_router.py + Rigpa-v3 calibration.py.
"""
from __future__ import annotations

import asyncio
from typing import AsyncGenerator

import numpy as np
import structlog

from neurolink.exceptions import CalibrationBusyError

log = structlog.get_logger(__name__)

_CALIBRATION_SEC: float = 30.0
_MIN_FRAMES: int = 10


class CalibrationSession:
    """Collects alpha power samples over 30 seconds and sets hub.baseline_alpha."""

    def __init__(self, hub) -> None:  # type: ignore[type-arg]
        self._hub = hub
        self._running = False
        self._task: asyncio.Task | None = None  # type: ignore[type-arg]

    @property
    def is_running(self) -> bool:
        return self._running

    async def start(self) -> None:
        """Start the 30-second calibration session.

        Raises CalibrationBusyError if already running.
        """
        if self._running:
            raise CalibrationBusyError("Calibration already running")
        self._running = True
        self._task = asyncio.create_task(self._run())
        log.info("calibration_started", duration_sec=_CALIBRATION_SEC)

    async def _run(self) -> None:
        """Collect alpha samples for 30 seconds, then set baseline."""
        alpha_samples: list[float] = []
        elapsed = 0.0
        interval = 0.25  # sample at 4 Hz

        try:
            while elapsed < _CALIBRATION_SEC:
                await asyncio.sleep(interval)
                elapsed += interval
                state = self._hub.get_state()
                if state.frame_count > 0:
                    alpha_samples.append(state.bands.alpha)

            if len(alpha_samples) >= _MIN_FRAMES:
                baseline = float(np.mean(alpha_samples))
                self._hub.baseline_alpha = baseline
                log.info(
                    "calibration_complete",
                    baseline_alpha=baseline,
                    n_samples=len(alpha_samples),
                )
            else:
                log.warning(
                    "calibration_insufficient_data",
                    n_samples=len(alpha_samples),
                )
        except asyncio.CancelledError:
            log.info("calibration_cancelled")
        finally:
            self._running = False

    async def wait_for_completion(self) -> float | None:
        """Wait for calibration to finish and return baseline_alpha."""
        if self._task is not None:
            await self._task
        return self._hub.baseline_alpha

    def cancel(self) -> None:
        """Cancel an in-progress calibration."""
        if self._task is not None:
            self._task.cancel()
