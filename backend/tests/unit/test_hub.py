"""Unit tests for hub.py — EEGHub singleton.

Covers:
  • emit_settling() — sentinel shape, fan-out, counter increment
  • get_stats() — keys, types, values
  • notify_baseline_complete() — sentinel shape
  • update() — state propagation, frame counter
  • SSE queue management (register / unregister)
  • reset() clears all counters
  • module-level delegates (emit_settling, get_stats, reset)
  • Thread safety for emit_settling + concurrent SSE writes
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import neurolink.hub as hub_module
from neurolink.hub import EEGHub
from neurolink.models.eeg import BandPowers, IngestPayload, NeurolinkState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def hub() -> EEGHub:
    """Fresh EEGHub instance (not the singleton) for isolation."""
    return EEGHub()


@pytest.fixture
def q() -> asyncio.Queue:
    return asyncio.Queue(maxsize=128)


def _minimal_payload(**kwargs) -> IngestPayload:
    """Minimal valid IngestPayload with sensible defaults."""
    defaults: dict[str, Any] = dict(
        source="muse_ble",
        bands=BandPowers(delta=0.2, theta=0.2, alpha=0.2, beta=0.2, gamma=0.2),
    )
    defaults.update(kwargs)
    return IngestPayload(**defaults)


# ---------------------------------------------------------------------------
# emit_settling() — sentinel structure
# ---------------------------------------------------------------------------

class TestEmitSettling:
    def test_default_reason_settling(self, hub, q):
        hub.register_sse_queue(q)
        hub.emit_settling()
        item = q.get_nowait()
        assert item["event"] == "settling"
        assert item["reason"] == "settling"

    def test_impedance_unstable_reason(self, hub, q):
        hub.register_sse_queue(q)
        hub.emit_settling(reason="impedance_unstable")
        item = q.get_nowait()
        assert item["reason"] == "impedance_unstable"

    def test_motion_settling_reason(self, hub, q):
        hub.register_sse_queue(q)
        hub.emit_settling(reason="motion_settling")
        assert q.get_nowait()["reason"] == "motion_settling"

    def test_env_not_ready_reason(self, hub, q):
        hub.register_sse_queue(q)
        hub.emit_settling(reason="env_not_ready")
        assert q.get_nowait()["reason"] == "env_not_ready"

    def test_unknown_reason_forwarded_as_is(self, hub, q):
        hub.register_sse_queue(q)  
        hub.emit_settling(reason="future_code_xyz")
        assert q.get_nowait()["reason"] == "future_code_xyz"

    def test_event_key_is_settling(self, hub, q):
        hub.register_sse_queue(q)
        hub.emit_settling(reason="impedance_unstable")
        assert q.get_nowait()["event"] == "settling"

    def test_sentinel_is_dict_not_neurolink_state(self, hub, q):
        hub.register_sse_queue(q)
        hub.emit_settling()
        item = q.get_nowait()
        assert isinstance(item, dict)
        assert not isinstance(item, NeurolinkState)

    def test_increments_settling_counter(self, hub):
        before = hub.get_stats()["settling_events_emitted"]
        hub.emit_settling()
        assert hub.get_stats()["settling_events_emitted"] == before + 1

    def test_counter_increments_with_multiple_calls(self, hub):
        for _ in range(5):
            hub.emit_settling()
        assert hub.get_stats()["settling_events_emitted"] == 5

    def test_fanout_to_multiple_queues(self, hub):
        q1: asyncio.Queue = asyncio.Queue(maxsize=128)
        q2: asyncio.Queue = asyncio.Queue(maxsize=128)
        hub.register_sse_queue(q1)
        hub.register_sse_queue(q2)
        hub.emit_settling(reason="motion_settling")
        assert q1.get_nowait()["reason"] == "motion_settling"
        assert q2.get_nowait()["reason"] == "motion_settling"

    def test_no_queues_does_not_raise(self, hub):
        hub.emit_settling()  # no queues registered — should be silent

    def test_full_queue_does_not_raise(self, hub):
        """A full SSE queue must drop the event, not raise."""
        full_q: asyncio.Queue = asyncio.Queue(maxsize=1)
        full_q.put_nowait({"placeholder": True})
        hub.register_sse_queue(full_q)
        hub.emit_settling()  # must not raise

    def test_emit_settling_does_not_update_neurolink_state(self, hub):
        state_before = hub.get_state().frame_count
        hub.emit_settling()
        assert hub.get_state().frame_count == state_before


# ---------------------------------------------------------------------------
# get_stats()
# ---------------------------------------------------------------------------

class TestGetStats:
    def test_returns_dict(self, hub):
        assert isinstance(hub.get_stats(), dict)

    def test_required_keys_present(self, hub):
        keys = hub.get_stats().keys()
        for k in (
            "frames_processed",
            "settling_events_emitted",
            "sse_client_count",
            "frame_count",
            "baseline_phase",
        ):
            assert k in keys, f"Missing key: {k}"

    def test_initial_frames_processed_zero(self, hub):
        assert hub.get_stats()["frames_processed"] == 0

    def test_initial_settling_events_zero(self, hub):
        assert hub.get_stats()["settling_events_emitted"] == 0

    def test_initial_sse_client_count_zero(self, hub):
        assert hub.get_stats()["sse_client_count"] == 0

    def test_sse_client_count_after_register(self, hub, q):
        hub.register_sse_queue(q)
        assert hub.get_stats()["sse_client_count"] == 1

    def test_sse_client_count_after_unregister(self, hub, q):
        hub.register_sse_queue(q)
        hub.unregister_sse_queue(q)
        assert hub.get_stats()["sse_client_count"] == 0

    def test_frames_processed_after_update(self, hub):
        with patch("neurolink.hub.EEGHub._schedule_redis_push"):
            hub.update(_minimal_payload())
        assert hub.get_stats()["frames_processed"] == 1

    def test_frame_count_increments_with_update(self, hub):
        with patch("neurolink.hub.EEGHub._schedule_redis_push"):
            hub.update(_minimal_payload())
            hub.update(_minimal_payload())
        assert hub.get_stats()["frame_count"] == 2

    def test_settling_events_increments(self, hub):
        hub.emit_settling()
        hub.emit_settling()
        assert hub.get_stats()["settling_events_emitted"] == 2

    def test_stats_after_reset(self, hub):
        hub.emit_settling()
        with patch("neurolink.hub.EEGHub._schedule_redis_push"):
            hub.update(_minimal_payload())
        hub.reset()
        s = hub.get_stats()
        assert s["frames_processed"] == 0
        assert s["settling_events_emitted"] == 0
        assert s["frame_count"] == 0


# ---------------------------------------------------------------------------
# notify_baseline_complete()
# ---------------------------------------------------------------------------

class TestNotifyBaselineComplete:
    def test_event_key_is_baseline_complete(self, hub, q):
        hub.register_sse_queue(q)
        hub.notify_baseline_complete()
        item = q.get_nowait()
        assert item["event"] == "baseline_complete"

    def test_sentinel_is_dict(self, hub, q):
        hub.register_sse_queue(q)
        hub.notify_baseline_complete()
        assert isinstance(q.get_nowait(), dict)

    def test_no_queues_does_not_raise(self, hub):
        hub.notify_baseline_complete()

    def test_fanout_to_multiple_queues(self, hub):
        q1: asyncio.Queue = asyncio.Queue(maxsize=128)
        q2: asyncio.Queue = asyncio.Queue(maxsize=128)
        hub.register_sse_queue(q1)
        hub.register_sse_queue(q2)
        hub.notify_baseline_complete()
        assert q1.get_nowait()["event"] == "baseline_complete"
        assert q2.get_nowait()["event"] == "baseline_complete"

    def test_baseline_sentinel_is_not_settling_event(self, hub, q):
        hub.register_sse_queue(q)
        hub.notify_baseline_complete()
        item = q.get_nowait()
        assert item.get("event") != "settling"


# ---------------------------------------------------------------------------
# update() — state propagation
# ---------------------------------------------------------------------------

class TestUpdate:
    def test_returns_neurolink_state(self, hub):
        with patch("neurolink.hub.EEGHub._schedule_redis_push"):
            result = hub.update(_minimal_payload())
        assert isinstance(result, NeurolinkState)

    def test_connected_flag_true_after_update(self, hub):
        with patch("neurolink.hub.EEGHub._schedule_redis_push"):
            state = hub.update(_minimal_payload())
        assert state.connected is True

    def test_frame_count_starts_at_1(self, hub):
        with patch("neurolink.hub.EEGHub._schedule_redis_push"):
            state = hub.update(_minimal_payload())
        assert state.frame_count == 1

    def test_frame_count_monotonic(self, hub):
        with patch("neurolink.hub.EEGHub._schedule_redis_push"):
            s1 = hub.update(_minimal_payload())
            s2 = hub.update(_minimal_payload())
        assert s2.frame_count == s1.frame_count + 1

    def test_source_propagated(self, hub):
        with patch("neurolink.hub.EEGHub._schedule_redis_push"):
            state = hub.update(_minimal_payload(source="muse_ble"))
        assert state.source == "muse_ble"

    def test_get_state_reflects_latest_update(self, hub):
        with patch("neurolink.hub.EEGHub._schedule_redis_push"):
            hub.update(_minimal_payload())
            hub.update(_minimal_payload())
        assert hub.get_state().frame_count == 2

    def test_update_fans_out_to_sse_queue(self, hub, q):
        hub.register_sse_queue(q)
        with patch("neurolink.hub.EEGHub._schedule_redis_push"):
            hub.update(_minimal_payload())
        item = q.get_nowait()
        assert isinstance(item, NeurolinkState)

    def test_artifact_rejected_propagated(self, hub):
        with patch("neurolink.hub.EEGHub._schedule_redis_push"):
            state = hub.update(_minimal_payload(artifact_rejected=True))
        assert state.artifact_rejected is True

    def test_artifact_reasons_propagated(self, hub):
        with patch("neurolink.hub.EEGHub._schedule_redis_push"):
            state = hub.update(_minimal_payload(artifact_reasons=["amplitude"]))
        assert "amplitude" in state.artifact_reasons


# ---------------------------------------------------------------------------
# SSE queue management
# ---------------------------------------------------------------------------

class TestSSEQueueManagement:
    def test_register_increases_client_count(self, hub):
        q1: asyncio.Queue = asyncio.Queue()
        hub.register_sse_queue(q1)
        assert hub.get_stats()["sse_client_count"] == 1

    def test_unregister_decreases_client_count(self, hub):
        q1: asyncio.Queue = asyncio.Queue()
        hub.register_sse_queue(q1)
        hub.unregister_sse_queue(q1)
        assert hub.get_stats()["sse_client_count"] == 0

    def test_unregister_unknown_queue_no_exception(self, hub):
        hub.unregister_sse_queue(asyncio.Queue())  # never registered

    def test_multiple_clients_all_receive_settling(self, hub):
        queues = [asyncio.Queue(maxsize=128) for _ in range(3)]
        for q in queues:
            hub.register_sse_queue(q)
        hub.emit_settling(reason="motion_settling")
        for q in queues:
            assert q.get_nowait()["reason"] == "motion_settling"


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_frame_count(self, hub):
        with patch("neurolink.hub.EEGHub._schedule_redis_push"):
            hub.update(_minimal_payload())
        hub.reset()
        assert hub.get_state().frame_count == 0

    def test_reset_clears_settling_counter(self, hub):
        hub.emit_settling()
        hub.reset()
        assert hub.get_stats()["settling_events_emitted"] == 0

    def test_reset_clears_frames_processed(self, hub):
        with patch("neurolink.hub.EEGHub._schedule_redis_push"):
            hub.update(_minimal_payload())
        hub.reset()
        assert hub.get_stats()["frames_processed"] == 0


# ---------------------------------------------------------------------------
# Module-level delegates
# ---------------------------------------------------------------------------

class TestModuleDelegates:
    """Smoke-test the module-level singleton delegates."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        hub_module.reset()
        yield
        hub_module.reset()

    def test_emit_settling_delegate(self):
        hub_module.emit_settling(reason="impedance_unstable")
        assert hub_module.get_stats()["settling_events_emitted"] == 1

    def test_get_stats_delegate_returns_dict(self):
        assert isinstance(hub_module.get_stats(), dict)

    def test_reset_delegate_clears_counters(self):
        hub_module.emit_settling()
        hub_module.reset()
        assert hub_module.get_stats()["settling_events_emitted"] == 0


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

class TestHubThreadSafety:
    def test_concurrent_emit_settling_increments_correctly(self, hub):
        n_threads = 5
        n_calls = 20
        errors: list[Exception] = []

        def worker():
            try:
                for _ in range(n_calls):
                    hub.emit_settling()
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        assert hub.get_stats()["settling_events_emitted"] == n_threads * n_calls

    def test_concurrent_register_unregister_and_emit(self, hub):
        errors: list[Exception] = []

        def emitter():
            try:
                for _ in range(30):
                    hub.emit_settling()
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        def registrar():
            try:
                for _ in range(10):
                    q = asyncio.Queue(maxsize=256)
                    hub.register_sse_queue(q)
                    hub.unregister_sse_queue(q)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = (
            [threading.Thread(target=emitter) for _ in range(3)]
            + [threading.Thread(target=registrar) for _ in range(2)]
        )
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
