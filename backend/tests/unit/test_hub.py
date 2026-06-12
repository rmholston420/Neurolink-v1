"""Unit tests for EEGHub — state management, classifiers, SSE fan-out."""

from __future__ import annotations

import asyncio

import pytest

from neurolink.hub import EEGHub
from neurolink.models.eeg import BandPowers, IngestPayload, NeurolinkState


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

class TestEEGHubState:
    def test_initial_state_not_connected(self, hub):
        state = hub.get_state()
        assert state.connected is False

    def test_update_returns_neurolink_state(self, hub, base_payload):
        state = hub.update(base_payload)
        assert isinstance(state, NeurolinkState)

    def test_update_sets_connected_true(self, hub, base_payload):
        state = hub.update(base_payload)
        assert state.connected is True

    def test_frame_count_increments(self, hub, base_payload):
        hub.update(base_payload)
        hub.update(base_payload)
        state = hub.get_state()
        assert state.frame_count == 2

    def test_get_state_returns_latest(self, hub, base_payload, alpha_dominant_bands):
        hub.update(base_payload)
        p2 = IngestPayload(source="mock", bands=alpha_dominant_bands)
        hub.update(p2)
        state = hub.get_state()
        assert state.frame_count == 2

    def test_reset_clears_state(self, hub, base_payload):
        hub.update(base_payload)
        hub.reset()
        state = hub.get_state()
        assert state.connected is False
        assert state.frame_count == 0

    def test_source_propagated(self, hub):
        p = IngestPayload(source="muse_ble",
                          bands=BandPowers(alpha=0.3, theta=0.3, beta=0.2, delta=0.1, gamma=0.1))
        state = hub.update(p)
        assert state.source == "muse_ble"


# ---------------------------------------------------------------------------
# EA-1
# ---------------------------------------------------------------------------

class TestEEGHubEA1:
    def test_get_ea1_returns_result(self, hub, base_payload):
        hub.update(base_payload)
        ea1 = hub.get_ea1()
        assert hasattr(ea1, "eligible")
        assert hasattr(ea1, "score")

    def test_get_ea1_initial_not_eligible(self, hub):
        ea1 = hub.get_ea1()
        # Before any update, default EA1Result should be not-eligible
        assert ea1.eligible is False


# ---------------------------------------------------------------------------
# SSE fan-out
# ---------------------------------------------------------------------------

class TestEEGHubSSEFanout:
    @pytest.mark.asyncio
    async def test_registered_queue_receives_state(self, hub, base_payload):
        q: asyncio.Queue = asyncio.Queue(maxsize=8)
        hub.register_sse_queue(q)
        hub.update(base_payload)
        state = q.get_nowait()
        assert isinstance(state, NeurolinkState)
        hub.unregister_sse_queue(q)

    @pytest.mark.asyncio
    async def test_unregistered_queue_no_receive(self, hub, base_payload):
        q: asyncio.Queue = asyncio.Queue(maxsize=8)
        hub.register_sse_queue(q)
        hub.unregister_sse_queue(q)
        hub.update(base_payload)
        assert q.empty()

    @pytest.mark.asyncio
    async def test_multiple_clients_all_receive(self, hub, base_payload):
        queues = [asyncio.Queue(maxsize=8) for _ in range(3)]
        for q in queues:
            hub.register_sse_queue(q)
        hub.update(base_payload)
        for q in queues:
            assert not q.empty()
            hub.unregister_sse_queue(q)

    @pytest.mark.asyncio
    async def test_full_queue_does_not_raise(self, hub, base_payload):
        """A full SSE queue should log a warning but not crash the hub."""
        q: asyncio.Queue = asyncio.Queue(maxsize=1)
        hub.register_sse_queue(q)
        # Fill the queue first
        q.put_nowait(hub.get_state())
        # Now try to fan-out into a full queue — should not raise
        hub.update(base_payload)
        hub.unregister_sse_queue(q)
