"""Integration tests: EEGPump → EEGHub → NeurolinkState → SSE queue pipeline."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import numpy as np

from neurolink.eeg_pump import EEGPump
from neurolink.hub import EEGHub
from neurolink.models.eeg import NeurolinkState


def _make_mock_adapter(n_channels: int = 4, n_samples: int = 256):
    """Return a minimal mock hardware adapter producing one EEGSample."""
    t = np.linspace(0, 1, n_samples)
    eeg_buffer = [
        (
            np.sin(2 * np.pi * 10 * t) + 0.1 * np.random.default_rng(i).standard_normal(n_samples)
        ).tolist()
        for i in range(n_channels)
    ]
    sample = MagicMock()
    sample.source = "mock"
    sample.address = ""
    sample.timestamp = 1.0
    sample.eeg_buffer = eeg_buffer
    sample.ppg_buffer = []
    sample.accel_buffer = []
    sample.gyro_buffer = []
    sample.poor_contact = False

    adapter = MagicMock()
    adapter.read_sample = AsyncMock(side_effect=[sample, None])
    return adapter


class TestHubPipeline:
    def test_pump_tick_updates_hub(self):
        hub = EEGHub()
        adapter = _make_mock_adapter()
        pump = EEGPump(adapter=adapter, hub=hub)

        asyncio.run(pump._tick())

        state = hub.get_state()
        assert state.connected is True
        assert state.frame_count == 1

    def test_eeg_samples_flow_through_pipeline(self):
        hub = EEGHub()
        adapter = _make_mock_adapter(n_channels=4, n_samples=256)
        pump = EEGPump(adapter=adapter, hub=hub)

        asyncio.run(pump._tick())

        state = hub.get_state()
        assert len(state.eeg_samples) == 4
        for ch in state.eeg_samples:
            assert 0 < len(ch) <= 64

    def test_sse_queue_receives_state_with_eeg_samples(self):
        hub = EEGHub()
        q: asyncio.Queue = asyncio.Queue(maxsize=10)
        hub.register_sse_queue(q)

        adapter = _make_mock_adapter(n_channels=4, n_samples=256)
        pump = EEGPump(adapter=adapter, hub=hub)

        asyncio.run(pump._tick())

        assert not q.empty()
        state: NeurolinkState = q.get_nowait()
        assert isinstance(state, NeurolinkState)
        assert len(state.eeg_samples) == 4

    def test_snapshot_contains_eeg_samples_after_tick(self):
        hub = EEGHub()
        adapter = _make_mock_adapter()
        pump = EEGPump(adapter=adapter, hub=hub)

        asyncio.run(pump._tick())

        snap = hub.snapshot()
        assert "eeg_samples" in snap
        assert len(snap["eeg_samples"]) == 4

    def test_band_powers_nonzero_after_tick(self):
        hub = EEGHub()
        adapter = _make_mock_adapter(n_channels=4, n_samples=512)
        pump = EEGPump(adapter=adapter, hub=hub)

        asyncio.run(pump._tick())

        state = hub.get_state()
        total = (
            state.bands.alpha
            + state.bands.theta
            + state.bands.beta
            + state.bands.delta
            + state.bands.gamma
        )
        assert total > 0, "Band powers should be nonzero after processing real signal"

    def test_multiple_ticks_increment_frame_count(self):
        hub = EEGHub()
        t = np.linspace(0, 1, 256)
        eeg_buffer = [(np.sin(2 * np.pi * 10 * t)).tolist()] * 4

        def _make_sample():
            s = MagicMock()
            s.source = "mock"
            s.address = ""
            s.timestamp = 1.0
            s.eeg_buffer = eeg_buffer
            s.ppg_buffer = []
            s.accel_buffer = []
            s.gyro_buffer = []
            s.poor_contact = False
            return s

        adapter = MagicMock()
        adapter.read_sample = AsyncMock(
            side_effect=[_make_sample(), _make_sample(), _make_sample(), None]
        )
        pump = EEGPump(adapter=adapter, hub=hub)

        async def _run_ticks():
            for _ in range(3):
                await pump._tick()

        asyncio.run(_run_ticks())

        state = hub.get_state()
        assert state.frame_count == 3
