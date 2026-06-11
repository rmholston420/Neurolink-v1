"""Unit tests for EEGHub SSE fanout, reset, and module-level delegates."""

from __future__ import annotations

import asyncio

from neurolink.hub import EEGHub, get_state, reset, snapshot, update
from neurolink.models.eeg import BandPowers, IngestPayload, NeurolinkState


def _payload(**kwargs) -> IngestPayload:
    return IngestPayload(
        source="mock",
        bands=BandPowers(alpha=0.3, theta=0.2, beta=0.15, delta=0.2, gamma=0.1),
        **kwargs,
    )


def test_hub_update_increments_frame_count():
    hub = EEGHub()
    assert hub.get_state().frame_count == 0
    hub.update(_payload())
    assert hub.get_state().frame_count == 1
    hub.update(_payload())
    assert hub.get_state().frame_count == 2


def test_hub_reset_clears_state():
    hub = EEGHub()
    hub.update(_payload())
    assert hub.get_state().frame_count == 1
    hub.reset()
    assert hub.get_state().frame_count == 0
    assert hub.get_state().connected is False


def test_hub_fanout_puts_state_on_queue():
    hub = EEGHub()
    q: asyncio.Queue = asyncio.Queue(maxsize=10)
    hub.register_sse_queue(q)
    hub.update(_payload())
    assert not q.empty()
    state = q.get_nowait()
    assert isinstance(state, NeurolinkState)
    assert state.frame_count == 1
    hub.unregister_sse_queue(q)


def test_hub_unregister_removes_queue():
    hub = EEGHub()
    q: asyncio.Queue = asyncio.Queue(maxsize=10)
    hub.register_sse_queue(q)
    hub.unregister_sse_queue(q)
    hub.update(_payload())
    assert q.empty()


def test_hub_snapshot_returns_dict():
    hub = EEGHub()
    hub.update(_payload())
    snap = hub.snapshot()
    assert isinstance(snap, dict)
    assert "frame_count" in snap
    assert snap["frame_count"] == 1


def test_hub_get_latest_initially_none():
    hub = EEGHub()
    assert hub.get_latest() is None


def test_module_level_delegates():
    reset()
    update(_payload())
    state = get_state()
    assert state.frame_count == 1
    snap = snapshot()
    assert snap["frame_count"] == 1
    reset()
    assert get_state().frame_count == 0
