"""Unit tests for EEGHub."""

from __future__ import annotations

import asyncio

import pytest

from neurolink.hub import EEGHub
from neurolink.models.eeg import BandPowers, IngestPayload, NeurolinkState


class TestHubInitialState:
    def test_connected_false_on_init(self, hub: EEGHub):
        assert hub.get_state().connected is False

    def test_frame_count_zero_on_init(self, hub: EEGHub):
        assert hub.get_state().frame_count == 0

    def test_ea1_eligible_false_on_init(self, hub: EEGHub):
        assert hub.get_ea1().eligible is False


class TestHubUpdate:
    def test_update_increments_frame_count(self, hub: EEGHub, base_payload):
        hub.update(base_payload)
        assert hub.get_state().frame_count == 1

    def test_update_sets_connected(self, hub: EEGHub, base_payload):
        hub.update(base_payload)
        assert hub.get_state().connected is True

    def test_update_stores_bands(self, hub: EEGHub, alpha_dominant_bands):
        payload = IngestPayload(source="mock", bands=alpha_dominant_bands)
        hub.update(payload)
        state = hub.get_state()
        assert abs(state.bands.alpha - 0.55) < 1e-6

    def test_update_passes_eeg_samples(self, hub: EEGHub):
        samples = [[float(i) for i in range(64)]] * 4
        payload = IngestPayload(
            source="mock",
            bands=BandPowers(),
            eeg_samples=samples,
        )
        hub.update(payload)
        state = hub.get_state()
        assert len(state.eeg_samples) == 4
        assert len(state.eeg_samples[0]) == 64
        assert state.eeg_samples[0][0] == 0.0
        assert state.eeg_samples[0][63] == 63.0

    def test_multiple_updates_monotonic_frame_count(self, hub: EEGHub, base_payload):
        for i in range(5):
            hub.update(base_payload)
        assert hub.get_state().frame_count == 5

    def test_reset_clears_state(self, hub: EEGHub, base_payload):
        hub.update(base_payload)
        hub.reset()
        assert hub.get_state().frame_count == 0
        assert hub.get_state().connected is False

    def test_source_propagated(self, hub: EEGHub):
        payload = IngestPayload(source="muse_ble", bands=BandPowers())
        hub.update(payload)
        assert hub.get_state().source == "muse_ble"


class TestHubSSEQueues:
    def test_register_and_fanout(self, hub: EEGHub, base_payload):
        q: asyncio.Queue = asyncio.Queue(maxsize=10)
        hub.register_sse_queue(q)
        hub.update(base_payload)
        assert not q.empty()
        item = q.get_nowait()
        assert isinstance(item, NeurolinkState)

    def test_unregister_stops_fanout(self, hub: EEGHub, base_payload):
        q: asyncio.Queue = asyncio.Queue(maxsize=10)
        hub.register_sse_queue(q)
        hub.unregister_sse_queue(q)
        hub.update(base_payload)
        assert q.empty()

    def test_full_queue_does_not_raise(self, hub: EEGHub, base_payload):
        q: asyncio.Queue = asyncio.Queue(maxsize=1)
        hub.register_sse_queue(q)
        # Fill queue
        hub.update(base_payload)
        # Second update — queue is full, should log warning but not raise
        hub.update(base_payload)


class TestHubSnapshot:
    def test_snapshot_returns_dict(self, hub: EEGHub, base_payload):
        hub.update(base_payload)
        snap = hub.snapshot()
        assert isinstance(snap, dict)
        assert "connected" in snap
        assert snap["connected"] is True

    def test_snapshot_contains_eeg_samples(self, hub: EEGHub):
        samples = [[1.0, 2.0, 3.0]] * 4
        payload = IngestPayload(source="mock", bands=BandPowers(), eeg_samples=samples)
        hub.update(payload)
        snap = hub.snapshot()
        assert snap["eeg_samples"] == samples
