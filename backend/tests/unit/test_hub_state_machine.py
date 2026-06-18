"""EEGHub state-machine tests.

Covers:
  - update() increments frame_count and persists state
  - SSE fan-out (register/unregister/full-queue drop)
  - notify_baseline_complete()
  - emit_settling() with reason codes
  - get_stats() counters
  - reset() atomically clears everything
  - snapshot() serialises to dict
  - set_latest_sample() / get_latest()
  - backward-compatible aliases
"""

from __future__ import annotations

import asyncio
import queue as stdlib_queue

import pytest

from neurolink.hub import EEGHub, get_hub, reset
from neurolink.models.eeg import BandPowers, IngestPayload


def _payload(
    alpha: float = 0.3,
    theta: float = 0.2,
    source: str = "mock",
) -> IngestPayload:
    return IngestPayload(
        source=source,
        bands=BandPowers(alpha=alpha, theta=theta, beta=0.15, delta=0.2, gamma=0.05),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Core update / state
# ─────────────────────────────────────────────────────────────────────────────

class TestEEGHubUpdate:
    def test_initial_frame_count_is_zero(self):
        hub = EEGHub()
        assert hub.get_state().frame_count == 0

    def test_update_increments_frame_count(self):
        hub = EEGHub()
        hub.update(_payload())
        assert hub.get_state().frame_count == 1

    def test_multiple_updates_increment_correctly(self):
        hub = EEGHub()
        for _ in range(5):
            hub.update(_payload())
        assert hub.get_state().frame_count == 5

    def test_update_sets_connected_true(self):
        hub = EEGHub()
        hub.update(_payload())
        assert hub.get_state().connected is True

    def test_update_stores_source(self):
        hub = EEGHub()
        hub.update(_payload(source="muse_ble"))
        assert hub.get_state().source == "muse_ble"

    def test_update_stores_band_powers(self):
        hub = EEGHub()
        hub.update(_payload(alpha=0.55, theta=0.18))
        state = hub.get_state()
        assert state.bands.alpha == pytest.approx(0.55)
        assert state.bands.theta == pytest.approx(0.18)

    def test_get_state_returns_snapshot_not_reference(self):
        """Mutating the returned state must not affect the hub."""
        hub = EEGHub()
        hub.update(_payload())
        state1 = hub.get_state()
        hub.update(_payload(alpha=0.99))
        # state1 should still reflect the first update
        assert state1.bands.alpha != pytest.approx(0.99)

    def test_frames_processed_counter(self):
        hub = EEGHub()
        for _ in range(7):
            hub.update(_payload())
        stats = hub.get_stats()
        assert stats["frames_processed"] == 7


# ─────────────────────────────────────────────────────────────────────────────
# EA-1 result
# ─────────────────────────────────────────────────────────────────────────────

class TestEEGHubEA1:
    def test_get_ea1_returns_ea1result(self):
        hub = EEGHub()
        ea1 = hub.get_ea1()
        assert hasattr(ea1, "eligible")

    def test_get_ea1_updated_after_ingest(self):
        hub = EEGHub()
        hub.update(_payload(alpha=0.55, theta=0.3))
        ea1 = hub.get_ea1()
        assert ea1 is not None


# ─────────────────────────────────────────────────────────────────────────────
# SSE queue management
# ─────────────────────────────────────────────────────────────────────────────

class TestSSEQueues:
    @pytest.mark.asyncio
    async def test_fanout_reaches_registered_queue(self):
        hub = EEGHub()
        q: asyncio.Queue = asyncio.Queue(maxsize=10)
        hub.register_sse_queue(q)
        hub.update(_payload())
        assert not q.empty()
        hub.unregister_sse_queue(q)

    @pytest.mark.asyncio
    async def test_unregister_removes_queue(self):
        hub = EEGHub()
        q: asyncio.Queue = asyncio.Queue(maxsize=10)
        hub.register_sse_queue(q)
        hub.unregister_sse_queue(q)
        hub.update(_payload())
        assert q.empty()

    def test_unregister_nonexistent_queue_is_safe(self):
        hub = EEGHub()
        q: asyncio.Queue = asyncio.Queue(maxsize=10)
        hub.unregister_sse_queue(q)  # must not raise

    @pytest.mark.asyncio
    async def test_full_queue_drops_frame_without_raising(self):
        hub = EEGHub()
        q: asyncio.Queue = asyncio.Queue(maxsize=1)
        q.put_nowait("placeholder")  # fill queue
        hub.register_sse_queue(q)
        hub.update(_payload())  # must not raise
        hub.unregister_sse_queue(q)

    def test_stdlib_queue_fanout_works(self):
        """Hub supports stdlib.Queue (used by sync tests)."""
        hub = EEGHub()
        q: stdlib_queue.Queue = stdlib_queue.Queue(maxsize=10)
        hub.register_sse_queue(q)
        hub.update(_payload())
        assert not q.empty()
        hub.unregister_sse_queue(q)

    def test_backward_compatible_register_alias(self):
        hub = EEGHub()
        q: asyncio.Queue = asyncio.Queue(maxsize=5)
        hub.register_sse_client(q)  # alias
        assert q in hub._sse_queues
        hub.unregister_sse_client(q)
        assert q not in hub._sse_queues


# ─────────────────────────────────────────────────────────────────────────────
# Sentinel events
# ─────────────────────────────────────────────────────────────────────────────

class TestSentinelEvents:
    @pytest.mark.asyncio
    async def test_notify_baseline_complete_pushes_event(self):
        hub = EEGHub()
        q: asyncio.Queue = asyncio.Queue(maxsize=10)
        hub.register_sse_queue(q)
        hub.notify_baseline_complete()
        item = await asyncio.wait_for(q.get(), timeout=1.0)
        assert isinstance(item, dict)
        assert item.get("event") == "baseline_complete"
        hub.unregister_sse_queue(q)

    @pytest.mark.asyncio
    async def test_emit_settling_pushes_event_with_reason(self):
        hub = EEGHub()
        q: asyncio.Queue = asyncio.Queue(maxsize=10)
        hub.register_sse_queue(q)
        hub.emit_settling(reason="impedance_unstable")
        item = await asyncio.wait_for(q.get(), timeout=1.0)
        assert item.get("event") == "settling"
        assert item.get("reason") == "impedance_unstable"
        hub.unregister_sse_queue(q)

    @pytest.mark.asyncio
    async def test_emit_settling_default_reason(self):
        hub = EEGHub()
        q: asyncio.Queue = asyncio.Queue(maxsize=10)
        hub.register_sse_queue(q)
        hub.emit_settling()
        item = await asyncio.wait_for(q.get(), timeout=1.0)
        assert item.get("reason") == "settling"
        hub.unregister_sse_queue(q)

    def test_emit_settling_increments_counter(self):
        hub = EEGHub()
        hub.emit_settling()
        hub.emit_settling(reason="motion_settling")
        stats = hub.get_stats()
        assert stats["settling_events_emitted"] == 2

    @pytest.mark.asyncio
    async def test_full_queue_drops_baseline_event_without_raising(self):
        hub = EEGHub()
        q: asyncio.Queue = asyncio.Queue(maxsize=1)
        q.put_nowait("placeholder")
        hub.register_sse_queue(q)
        hub.notify_baseline_complete()  # must not raise
        hub.unregister_sse_queue(q)


# ─────────────────────────────────────────────────────────────────────────────
# Reset
# ─────────────────────────────────────────────────────────────────────────────

class TestHubReset:
    def test_reset_clears_frame_count(self):
        hub = EEGHub()
        for _ in range(5):
            hub.update(_payload())
        hub.reset()
        assert hub.get_state().frame_count == 0

    def test_reset_clears_connected_flag(self):
        hub = EEGHub()
        hub.update(_payload())
        hub.reset()
        assert hub.get_state().connected is False

    def test_reset_restores_default_baseline_alpha(self):
        hub = EEGHub()
        hub.baseline_alpha = 0.99
        hub.reset()
        assert hub.baseline_alpha == pytest.approx(0.30, abs=0.01)

    def test_reset_clears_frames_processed_counter(self):
        hub = EEGHub()
        for _ in range(4):
            hub.update(_payload())
        hub.reset()
        assert hub.get_stats()["frames_processed"] == 0

    def test_reset_clears_settling_counter(self):
        hub = EEGHub()
        hub.emit_settling()
        hub.emit_settling()
        hub.reset()
        assert hub.get_stats()["settling_events_emitted"] == 0

    def test_reset_clears_latest_sample(self):
        hub = EEGHub()
        from unittest.mock import MagicMock
        hub.set_latest_sample(MagicMock())
        hub.reset()
        assert hub.get_latest() is None


# ─────────────────────────────────────────────────────────────────────────────
# snapshot / get_latest / set_latest_sample
# ─────────────────────────────────────────────────────────────────────────────

class TestHubAccessors:
    def test_snapshot_returns_dict(self):
        hub = EEGHub()
        snap = hub.snapshot()
        assert isinstance(snap, dict)
        assert "frame_count" in snap

    def test_snapshot_after_update_reflects_data(self):
        hub = EEGHub()
        hub.update(_payload(alpha=0.77))
        snap = hub.snapshot()
        assert snap["bands"]["alpha"] == pytest.approx(0.77)

    def test_get_latest_initially_none(self):
        hub = EEGHub()
        assert hub.get_latest() is None

    def test_set_and_get_latest_sample(self):
        from unittest.mock import MagicMock
        hub = EEGHub()
        sample = MagicMock()
        hub.set_latest_sample(sample)
        assert hub.get_latest() is sample


# ─────────────────────────────────────────────────────────────────────────────
# Module-level delegates
# ─────────────────────────────────────────────────────────────────────────────

class TestModuleLevelDelegates:
    def test_get_hub_returns_eeg_hub(self):
        h = get_hub()
        assert isinstance(h, EEGHub)

    def test_module_reset_clears_singleton(self):
        import neurolink.hub as hub_mod
        hub_mod._hub.update(_payload())
        hub_mod.reset()
        assert hub_mod.get_state().frame_count == 0
