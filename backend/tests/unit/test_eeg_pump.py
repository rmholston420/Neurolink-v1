"""Unit tests for EEGPump._build_payload — exercises the eeg_samples extraction path."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from neurolink.eeg_pump import EEGPump
from neurolink.hub import EEGHub


def _make_sample(
    n_channels: int = 4,
    n_samples: int = 256,
    freq_hz: float = 10.0,
) -> object:
    """Construct a minimal EEGSample-like object."""
    t = np.linspace(0, n_samples / 256.0, n_samples)
    eeg_buffer = [
        (np.sin(2 * np.pi * freq_hz * t) + 0.05 * np.random.randn(n_samples)).tolist()
        for _ in range(n_channels)
    ]
    sample = MagicMock()
    sample.source = "mock"
    sample.address = ""
    sample.timestamp = 0.0
    sample.eeg_buffer = eeg_buffer
    sample.ppg_buffer = []
    sample.accel_buffer = []
    sample.gyro_buffer = []
    sample.poor_contact = False
    sample.extra = {}
    return sample


class TestEEGPumpBuildPayload:
    async def test_eeg_samples_populated_when_buffer_present(self):
        hub = EEGHub()
        pump = EEGPump(adapter=MagicMock(), hub=hub)
        sample = _make_sample(n_channels=4, n_samples=256)

        payload = await pump._build_payload(sample)

        assert len(payload.eeg_samples) == 4, "Expected 4 channels"
        for ch in payload.eeg_samples:
            assert len(ch) <= 64, f"Window should be \u226464 samples, got {len(ch)}"
            assert len(ch) > 0

    async def test_eeg_samples_empty_when_no_buffer(self):
        hub = EEGHub()
        pump = EEGPump(adapter=MagicMock(), hub=hub)
        sample = MagicMock()
        sample.source = "mock"
        sample.address = ""
        sample.timestamp = 0.0
        sample.eeg_buffer = []  # no buffer
        sample.ppg_buffer = []
        sample.accel_buffer = []
        sample.gyro_buffer = []
        sample.poor_contact = False
        sample.extra = {}

        payload = await pump._build_payload(sample)

        assert payload.eeg_samples == []

    async def test_eeg_samples_window_capped_at_64(self):
        hub = EEGHub()
        pump = EEGPump(adapter=MagicMock(), hub=hub)
        # 512-sample buffer — window should be capped at 64
        sample = _make_sample(n_channels=4, n_samples=512)

        payload = await pump._build_payload(sample)

        for ch in payload.eeg_samples:
            assert len(ch) == 64

    async def test_bands_nonzero_for_clean_sine(self):
        hub = EEGHub()
        pump = EEGPump(adapter=MagicMock(), hub=hub)
        sample = _make_sample(n_channels=4, n_samples=512, freq_hz=10.0)

        payload = await pump._build_payload(sample)

        assert payload.bands.alpha > 0, "Alpha should be nonzero for 10Hz input"
