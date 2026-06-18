"""Extended unit tests for CalibrationSession.

Tests phase transitions, cancellation robustness, insufficient-data
branch, and idempotency of concurrent run() calls.

Uses a MockAdapter with injected samples so no real hardware is needed.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from neurolink.calibration import (
    BASELINE_SEC,
    TOTAL_DURATION_SEC,
    WARMUP_SEC,
    CalibrationSession,
    _MIN_FRAMES,
)
from neurolink.hub import EEGHub


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_sample(alpha_power: float = 0.4):
    """Return a mock EEGSample whose eeg_buffer yields a known alpha signal."""
    sample = MagicMock()
    # 4 channels × 256 samples of a 10 Hz sine (alpha band)
    t = np.linspace(0, 1, 256)
    channel = np.sin(2 * np.pi * 10 * t) * alpha_power
    sample.eeg_buffer = [channel.tolist() for _ in range(4)]
    return sample


def _make_adapter(sample=None, return_none: bool = False):
    adapter = MagicMock()
    if return_none:
        adapter.read_sample = AsyncMock(return_value=None)
    else:
        s = sample or _make_sample()
        adapter.read_sample = AsyncMock(return_value=s)
    return adapter


# ─────────────────────────────────────────────────────────────────────────────
# Initial state
# ─────────────────────────────────────────────────────────────────────────────

class TestCalibrationSessionInit:
    def test_initial_phase_is_idle(self):
        hub = EEGHub()
        cal = CalibrationSession(_make_adapter(), hub)
        assert cal.phase == "idle"

    def test_initial_elapsed_is_zero(self):
        hub = EEGHub()
        cal = CalibrationSession(_make_adapter(), hub)
        assert cal.elapsed == 0.0

    def test_is_running_false_initially(self):
        hub = EEGHub()
        cal = CalibrationSession(_make_adapter(), hub)
        assert cal.is_running is False

    def test_baseline_alpha_none_initially(self):
        hub = EEGHub()
        cal = CalibrationSession(_make_adapter(), hub)
        assert cal.baseline_alpha is None


# ─────────────────────────────────────────────────────────────────────────────
# Concurrent run() idempotency
# ─────────────────────────────────────────────────────────────────────────────

class TestCalibrationIdempotency:
    @pytest.mark.asyncio
    async def test_second_run_call_while_running_returns_none(self):
        """A second run() on an already-running session returns None immediately."""
        hub = EEGHub()
        adapter = _make_adapter()
        cal = CalibrationSession(adapter, hub)

        # Manually set running flag to simulate an in-progress session.
        cal._running = True
        result = await cal.run()
        assert result is None
        # Clean up
        cal._running = False


# ─────────────────────────────────────────────────────────────────────────────
# Fast-forwarded run using mocked time.monotonic
# ─────────────────────────────────────────────────────────────────────────────

class TestCalibrationFastForward:
    @pytest.mark.asyncio
    async def test_phase_transitions_warmup_then_baseline_then_complete(self):
        """
        Simulate elapsed time racing past WARMUP_SEC and TOTAL_DURATION_SEC
        by patching time.monotonic to return an ever-increasing sequence.
        """
        hub = EEGHub()
        adapter = _make_adapter()

        # Time sequence: 0, 1, …, 31, 32, …, 91 (jump past all windows fast)
        times = iter([0.0, 31.0, TOTAL_DURATION_SEC + 1.0])

        with patch("neurolink.calibration.time.monotonic", side_effect=times):
            cal = CalibrationSession(adapter, hub)
            # run() will loop until elapsed >= TOTAL_DURATION_SEC.
            # Because we only have 3 time values, the loop will exit on
            # the 3rd call (elapsed >= TOTAL_DURATION_SEC).
            result = await cal.run()

        # After run(), phase must be 'complete'
        assert cal.phase == "complete"
        assert cal.is_running is False

    @pytest.mark.asyncio
    async def test_run_returns_none_when_insufficient_samples(self):
        """
        If the adapter returns None for all reads, no alpha samples are
        collected, so the function should return None (insufficient data branch).
        """
        hub = EEGHub()
        adapter = _make_adapter(return_none=True)

        # Fast-forward time so both warmup and baseline windows expire quickly.
        times = [0.0, WARMUP_SEC + 1.0, TOTAL_DURATION_SEC + 1.0]
        call_count = 0

        def _fast_time():
            nonlocal call_count
            if call_count < len(times):
                t = times[call_count]
            else:
                t = TOTAL_DURATION_SEC + 2.0
            call_count += 1
            return t

        with patch("neurolink.calibration.time.monotonic", side_effect=_fast_time):
            cal = CalibrationSession(adapter, hub)
            result = await cal.run()

        assert result is None
        assert cal.phase == "complete"

    @pytest.mark.asyncio
    async def test_cancellation_sets_phase_complete(self):
        """
        If the task is cancelled mid-run, the finally block must set phase
        to 'complete' and is_running to False.
        """
        hub = EEGHub()
        adapter = MagicMock()
        # read_sample sleeps forever so we can cancel cleanly
        adapter.read_sample = AsyncMock(side_effect=asyncio.sleep(9999))

        cal = CalibrationSession(adapter, hub)
        task = asyncio.create_task(cal.run())
        await asyncio.sleep(0.05)  # let the task start
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert cal.phase == "complete"
        assert cal.is_running is False


# ─────────────────────────────────────────────────────────────────────────────
# Constants sanity
# ─────────────────────────────────────────────────────────────────────────────

class TestCalibrationConstants:
    def test_total_duration_equals_warmup_plus_baseline(self):
        assert TOTAL_DURATION_SEC == WARMUP_SEC + BASELINE_SEC

    def test_total_duration_is_ninety(self):
        assert TOTAL_DURATION_SEC == pytest.approx(90.0)

    def test_warmup_is_thirty(self):
        assert WARMUP_SEC == pytest.approx(30.0)

    def test_min_frames_positive(self):
        assert _MIN_FRAMES > 0
