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

        cal._running = True
        result = await cal.run()
        assert result is None
        cal._running = False


# ─────────────────────────────────────────────────────────────────────────────
# Fast-forwarded run using mocked time.monotonic
# ─────────────────────────────────────────────────────────────────────────────

class TestCalibrationFastForward:
    @pytest.mark.asyncio
    async def test_phase_transitions_warmup_then_baseline_then_complete(self):
        hub = EEGHub()
        adapter = _make_adapter()

        _times = [0.0, 31.0, TOTAL_DURATION_SEC + 1.0]
        _idx = 0

        def _monotonic():
            nonlocal _idx
            val = _times[min(_idx, len(_times) - 1)]
            _idx += 1
            return val

        with patch("neurolink.calibration.time.monotonic", side_effect=_monotonic):
            cal = CalibrationSession(adapter, hub)
            await cal.run()

        assert cal.phase == "complete"
        assert cal.is_running is False

    @pytest.mark.asyncio
    async def test_run_returns_none_when_insufficient_samples(self):
        hub = EEGHub()
        adapter = _make_adapter(return_none=True)

        _times = [0.0, WARMUP_SEC + 1.0, TOTAL_DURATION_SEC + 1.0]
        _idx = 0

        def _fast_time():
            nonlocal _idx
            val = _times[min(_idx, len(_times) - 1)]
            _idx += 1
            return val

        with patch("neurolink.calibration.time.monotonic", side_effect=_fast_time):
            cal = CalibrationSession(adapter, hub)
            result = await cal.run()

        assert result is None
        assert cal.phase == "complete"

    @pytest.mark.asyncio
    async def test_cancellation_sets_phase_complete(self):
        """
        Cancelling the run() task mid-flight must cause the finally block
        to set phase='complete' and is_running=False.

        We block read_sample on an asyncio.Event so that cancellation
        propagates as CancelledError through the await, rather than
        spawning leaking sleep coroutines.
        """
        hub = EEGHub()
        adapter = MagicMock()

        # A never-set Event means read_sample blocks forever,
        # but is instantly interrupted when the outer task is cancelled.
        _block = asyncio.Event()

        async def _blocking_read(*_a, **_kw):
            await _block.wait()  # blocks until cancelled
            return None  # unreachable

        adapter.read_sample = _blocking_read

        cal = CalibrationSession(adapter, hub)
        task = asyncio.create_task(cal.run())
        await asyncio.sleep(0.05)  # let the task enter the await
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
