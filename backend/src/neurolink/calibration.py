"""Calibration session — 30-second baseline alpha capture.

Ported from Rigpa-v2 calibration_router.py + Rigpa-v3 calibration.py.
"""
from __future__ import annotations

import asyncio
import time

import numpy as np
import structlog

from neurolink.hardware.base import HardwareAdapter

log = structlog.get_logger(__name__)

_CALIBRATION_DURATION_SEC: float = 30.0
_MIN_FRAMES: int = 10


class CalibrationSession:
    """Runs a 30-second baseline alpha capture using the active adapter.

    Sets hub.baseline_alpha on completion.
    """

    def __init__(self, adapter: HardwareAdapter, hub) -> None:  # type: ignore[no-untyped-def]
        self._adapter = adapter
        self._hub = hub
        self._running: bool = False
        self._baseline_alpha: float | None = None

    async def run(self) -> float | None:
        """Run the calibration session and return the baseline alpha value.

        Returns:
            Mean alpha band power fraction, or None if insufficient data.
        """
        if self._running:
            log.warning("calibration_already_running")
            return None

        self._running = True
        alpha_samples: list[float] = []
        start = time.monotonic()

        log.info("calibration_started", duration_sec=_CALIBRATION_DURATION_SEC)

        try:
            from neurolink.dsp.bandpower import compute_band_powers_from_buffer

            while time.monotonic() - start < _CALIBRATION_DURATION_SEC:
                sample = await self._adapter.read_sample()
                if sample is None:
                    await asyncio.sleep(0.1)
                    continue

                if sample.eeg_buffer:
                    eeg = np.array(sample.eeg_buffer, dtype=np.float32)
                    bands = compute_band_powers_from_buffer(eeg)
                    alpha = bands.get("alpha", 0.0)
                    if alpha > 0:
                        alpha_samples.append(alpha)

        except asyncio.CancelledError:
            log.warning("calibration_cancelled")
        finally:
            self._running = False

        if len(alpha_samples) < _MIN_FRAMES:
            log.warning("calibration_insufficient_data", n=len(alpha_samples))
            return None

        baseline = float(np.mean(alpha_samples))
        self._baseline_alpha = baseline
        self._hub.baseline_alpha = baseline
        log.info("calibration_complete", baseline_alpha=baseline, n=len(alpha_samples))
        return baseline

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def baseline_alpha(self) -> float | None:
        return self._baseline_alpha
