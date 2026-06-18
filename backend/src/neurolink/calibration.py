"""Calibration session - 90-second baseline alpha capture.

Ported from Rigpa-v2 calibration_router.py + Rigpa-v3 calibration.py.

Protocol
--------
* Total duration  : 90 s  (TOTAL_DURATION_SEC)
* Warmup window   : 0-30 s (WARMUP_SEC)
    EEG is sampled but ALL samples are discarded.  This purges movement
    artefacts from donning the headset and gives the ADS1299 front-end
    time to DC-stabilise on scalp impedance.
* Baseline window : 30-90 s (BASELINE_SEC = 60 s)
    Only samples collected in this window contribute to the mean alpha
    baseline stored on the hub.
"""

from __future__ import annotations

import asyncio
import time

import numpy as np
import structlog

from neurolink.hardware.base import HardwareAdapter

log = structlog.get_logger(__name__)

WARMUP_SEC: float = 30.0  # discard window - artefact purge + DC settle
BASELINE_SEC: float = 60.0  # clean capture window
TOTAL_DURATION_SEC: float = WARMUP_SEC + BASELINE_SEC  # 90 s total
_MIN_FRAMES: int = 30  # minimum accepted baseline samples


class CalibrationSession:
    """Runs a 90-second baseline alpha capture using the active adapter.

    Phase 1 - Warmup (0-30 s)
        Samples collected but discarded.  Clears movement and donning
        artefacts; allows the EEG front-end to DC-stabilise.

    Phase 2 - Baseline capture (30-90 s)
        Alpha band power accumulated into alpha_samples.  The mean of
        this 60-second window is written to hub.baseline_alpha.

    Sets hub.baseline_alpha on completion.
    """

    def __init__(self, adapter: HardwareAdapter, hub) -> None:
        self._adapter = adapter
        self._hub = hub
        self._running: bool = False
        self._baseline_alpha: float | None = None
        # Exposed for the router to stream progress to SSE clients.
        self.elapsed: float = 0.0
        self.phase: str = "idle"  # 'idle' | 'warmup' | 'baseline' | 'complete'

    async def run(self) -> float | None:
        """Run the calibration session and return the baseline alpha value.

        Returns:
            Mean alpha band power fraction (seconds 30-90), or None if
            insufficient data was collected.
        """
        if self._running:
            log.warning("calibration_already_running")
            return None

        self._running = True
        self.elapsed = 0.0
        self.phase = "warmup"
        alpha_samples: list[float] = []
        start = time.monotonic()

        log.info(
            "calibration_started",
            total_sec=TOTAL_DURATION_SEC,
            warmup_sec=WARMUP_SEC,
            baseline_sec=BASELINE_SEC,
        )

        try:
            from neurolink.dsp.bandpower import compute_band_powers_from_buffer

            while True:
                now = time.monotonic()
                self.elapsed = now - start

                if self.elapsed >= TOTAL_DURATION_SEC:
                    break

                # Update phase label for progress consumers
                if self.elapsed < WARMUP_SEC:
                    if self.phase != "warmup":
                        self.phase = "warmup"
                        log.info("calibration_phase", phase="warmup")
                else:
                    if self.phase != "baseline":
                        self.phase = "baseline"
                        log.info(
                            "calibration_phase",
                            phase="baseline",
                            elapsed=round(self.elapsed, 1),
                        )

                sample = await self._adapter.read_sample()
                if sample is None:
                    await asyncio.sleep(0.1)
                    continue

                # Warmup window: collect but discard
                if self.elapsed < WARMUP_SEC:
                    await asyncio.sleep(0.05)
                    continue

                # Baseline window: accumulate alpha samples
                if sample.eeg_buffer:
                    eeg = np.array(sample.eeg_buffer, dtype=np.float32)
                    bands = compute_band_powers_from_buffer(eeg)
                    alpha = bands.get("alpha", 0.0)
                    if alpha > 0:
                        alpha_samples.append(alpha)

                await asyncio.sleep(0.05)

        except asyncio.CancelledError:
            log.warning("calibration_cancelled", elapsed=round(self.elapsed, 1))
        finally:
            self._running = False
            self.phase = "complete"

        if len(alpha_samples) < _MIN_FRAMES:
            log.warning(
                "calibration_insufficient_data",
                n=len(alpha_samples),
                min_required=_MIN_FRAMES,
            )
            return None

        baseline = float(np.mean(alpha_samples))
        self._baseline_alpha = baseline
        self._hub.baseline_alpha = baseline
        log.info(
            "calibration_complete",
            baseline_alpha=baseline,
            n=len(alpha_samples),
            baseline_window_sec=BASELINE_SEC,
        )
        return baseline

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def baseline_alpha(self) -> float | None:
        return self._baseline_alpha
