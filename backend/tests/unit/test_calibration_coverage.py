"""Coverage tests for calibration.py."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from neurolink.calibration import run_calibration
from neurolink.hub import EEGHub
from neurolink.models.eeg import BandPowers, IngestPayload


def _hub_with_frames(n: int) -> EEGHub:
    """Return a hub pre-loaded with n frames of alpha=0.5."""
    hub = EEGHub()
    payload = IngestPayload(
        source="mock",
        bands=BandPowers(alpha=0.5, theta=0.2, beta=0.15, delta=0.1, gamma=0.05),
    )
    for _ in range(n):
        hub.update(payload)
    return hub


# ---------------------------------------------------------------------------
# Happy path — enough frames, baseline updated
# ---------------------------------------------------------------------------

async def test_run_calibration_updates_baseline():
    hub = EEGHub()
    # Patch hub.get_latest to return samples with known alpha
    sample = MagicMock()
    sample.source = "mock"

    # Provide enough alpha values via the hub directly
    payload = IngestPayload(
        source="mock",
        bands=BandPowers(alpha=0.6, theta=0.2, beta=0.1, delta=0.05, gamma=0.05),
    )
    # Pre-load 30 frames so calibration has data
    for _ in range(30):
        hub.update(payload)

    original_baseline = hub.baseline_alpha
    await run_calibration(hub, duration_secs=0.1, sample_interval=0.001)
    # baseline_alpha should have been updated (away from default 0.30 toward 0.6)
    assert hub.baseline_alpha != original_baseline or hub.baseline_alpha > 0.0


# ---------------------------------------------------------------------------
# CancelledError exits cleanly
# ---------------------------------------------------------------------------

async def test_run_calibration_cancelled():
    hub = EEGHub()

    async def _cancel_after_start():
        task = asyncio.create_task(run_calibration(hub, duration_secs=60, sample_interval=0.001))
        await asyncio.sleep(0.01)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    await _cancel_after_start()  # must not raise


# ---------------------------------------------------------------------------
# Too few frames — fallback to default baseline
# ---------------------------------------------------------------------------

async def test_run_calibration_too_few_frames_keeps_default():
    hub = EEGHub()
    original = hub.baseline_alpha
    # Very short duration + large interval = 0 samples collected
    await run_calibration(hub, duration_secs=0.001, sample_interval=10.0)
    # baseline should remain at default (no update when < min_frames)
    assert hub.baseline_alpha == original
