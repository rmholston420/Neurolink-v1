"""Unit tests for hub.emit_settling() and hub.get_stats()."""

from __future__ import annotations

import asyncio
import threading

import pytest

from neurolink.hub import EEGHub


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def hub() -> EEGHub:
    """Fresh, isolated hub instance per test (NOT the module singleton)."""
    h = EEGHub()
    return h


@pytest.fixture()
def queue(hub: EEGHub) -> asyncio.Queue:
    """Register and yield a single asyncio.Queue; unregister after test."""
    q: asyncio.Queue = asyncio.Queue(maxsize=64)
    hub.register_sse_queue(q)
    yield q
    hub.unregister_sse_queue(q)


# ---------------------------------------------------------------------------
# emit_settling — basic behaviour
# ---------------------------------------------------------------------------

class TestEmitSettling:
    def test_default_reason_is_settling(self, hub: EEGHub, queue: asyncio.Queue):
        hub.emit_settling()
        item = queue.get_nowait()
        assert item["event"] == "settling"
        assert item["reason"] == "settling"

    def test_impedance_unstable_reason(self, hub: EEGHub, queue: asyncio.Queue):
        hub.emit_settling(reason="impedance_unstable")
        item = queue.get_nowait()
        assert item["reason"] == "impedance_unstable"

    def test_motion_settling_reason(self, hub: EEGHub, queue: asyncio.Queue):
        hub.emit_settling(reason="motion_settling")
        item = queue.get_nowait()
        assert item["reason"] == "motion_settling"

    def test_env_not_ready_reason(self, hub: EEGHub, queue: asyncio.Queue):
        hub.emit_settling(reason="env_not_ready")
        item = queue.get_nowait()
        assert item["reason"] == "env_not_ready"

    def test_unknown_reason_forwarded_as_is(self, hub: EEGHub, queue: asyncio.Queue):
        hub.emit_settling(reason="future_code_v2")
        item = queue.get_nowait()
        assert item["reason"] == "future_code_v2"

    def test_event_key_is_settling(self, hub: EEGHub, queue: asyncio.Queue):
        hub.emit_settling(reason="motion_settling")
        item = queue.get_nowait()
        assert item["event"] == "settling"

    def test_item_is_dict_not_state(self, hub: EEGHub, queue: asyncio.Queue):
        hub.emit_settling()
        item = queue.get_nowait()
        assert isinstance(item, dict)
        # Must NOT look like NeurolinkState (no 'bands' key)
        assert "bands" not in item


# ---------------------------------------------------------------------------
# emit_settling — does NOT update NeurolinkState
# ---------------------------------------------------------------------------

class TestEmitSettlingDoesNotUpdateState:
    def test_frame_count_unchanged_after_settling(self, hub: EEGHub):
        before = hub.get_state().frame_count
        hub.emit_settling(reason="impedance_unstable")
        after = hub.get_state().frame_count
        assert before == after

    def test_settling_does_not_change_connected_flag(self, hub: EEGHub):
        hub.emit_settling()
        assert hub.get_state().connected is False  # default; not touched


# ---------------------------------------------------------------------------
# emit_settling — fan-out to multiple queues
# ---------------------------------------------------------------------------

class TestEmitSettlingFanout:
    def test_reaches_all_registered_queues(self, hub: EEGHub):
        queues = [asyncio.Queue(maxsize=16) for _ in range(3)]
        for q in queues:
            hub.register_sse_queue(q)

        hub.emit_settling(reason="impedance_unstable")

        for q in queues:
            item = q.get_nowait()
            assert item["event"] == "settling"
            assert item["reason"] == "impedance_unstable"

        for q in queues:
            hub.unregister_sse_queue(q)

    def test_does_not_reach_unregistered_queue(self, hub: EEGHub):
        q_reg = asyncio.Queue(maxsize=8)
        q_unreg = asyncio.Queue(maxsize=8)
        hub.register_sse_queue(q_reg)
        hub.register_sse_queue(q_unreg)
        hub.unregister_sse_queue(q_unreg)

        hub.emit_settling()

        assert not q_reg.empty()    # registered → received
        assert q_unreg.empty()      # unregistered → nothing
        hub.unregister_sse_queue(q_reg)

    def test_full_queue_does_not_raise(self, hub: EEGHub):
        """A full SSE queue must be silently dropped, not raise."""
        q = asyncio.Queue(maxsize=1)
        q.put_nowait({"dummy": True})  # fill it
        hub.register_sse_queue(q)
        hub.emit_settling()  # must not raise
        hub.unregister_sse_queue(q)


# ---------------------------------------------------------------------------
# emit_settling — counter incremented
# ---------------------------------------------------------------------------

class TestEmitSettlingCounter:
    def test_counter_increments_on_each_call(self, hub: EEGHub):
        assert hub.get_stats()["settling_events_emitted"] == 0
        hub.emit_settling()
        assert hub.get_stats()["settling_events_emitted"] == 1
        hub.emit_settling(reason="motion_settling")
        assert hub.get_stats()["settling_events_emitted"] == 2

    def test_counter_resets_on_hub_reset(self, hub: EEGHub):
        hub.emit_settling()
        hub.emit_settling()
        hub.reset()
        assert hub.get_stats()["settling_events_emitted"] == 0


# ---------------------------------------------------------------------------
# get_stats
# ---------------------------------------------------------------------------

class TestGetStats:
    def test_returns_dict(self, hub: EEGHub):
        stats = hub.get_stats()
        assert isinstance(stats, dict)

    def test_all_expected_keys_present(self, hub: EEGHub):
        stats = hub.get_stats()
        expected = {
            "frames_processed",
            "settling_events_emitted",
            "sse_client_count",
            "frame_count",
            "baseline_phase",
        }
        assert expected <= stats.keys()

    def test_initial_values(self, hub: EEGHub):
        stats = hub.get_stats()
        assert stats["frames_processed"] == 0
        assert stats["settling_events_emitted"] == 0
        assert stats["sse_client_count"] == 0
        assert stats["frame_count"] == 0

    def test_sse_client_count_reflects_registered_queues(self, hub: EEGHub):
        assert hub.get_stats()["sse_client_count"] == 0
        q1 = asyncio.Queue()
        q2 = asyncio.Queue()
        hub.register_sse_queue(q1)
        hub.register_sse_queue(q2)
        assert hub.get_stats()["sse_client_count"] == 2
        hub.unregister_sse_queue(q1)
        assert hub.get_stats()["sse_client_count"] == 1

    def test_frames_processed_does_not_increment_on_settling(self, hub: EEGHub):
        hub.emit_settling()
        hub.emit_settling()
        assert hub.get_stats()["frames_processed"] == 0

    def test_settling_and_frames_counters_independent(self, hub: EEGHub):
        hub.emit_settling()
        s = hub.get_stats()
        assert s["settling_events_emitted"] == 1
        assert s["frames_processed"] == 0


# ---------------------------------------------------------------------------
# Module-level delegates
# ---------------------------------------------------------------------------

class TestModuleLevelDelegates:
    def test_module_emit_settling_exists(self):
        import neurolink.hub as hub_mod
        assert callable(hub_mod.emit_settling)

    def test_module_get_stats_exists(self):
        import neurolink.hub as hub_mod
        assert callable(hub_mod.get_stats)

    def test_module_get_stats_returns_dict(self):
        import neurolink.hub as hub_mod
        result = hub_mod.get_stats()
        assert isinstance(result, dict)
        assert "settling_events_emitted" in result


# ---------------------------------------------------------------------------
# Thread-safety
# ---------------------------------------------------------------------------

class TestEmitSettlingThreadSafety:
    def test_concurrent_emit_no_exception(self, hub: EEGHub):
        queues = [asyncio.Queue(maxsize=256) for _ in range(4)]
        for q in queues:
            hub.register_sse_queue(q)

        errors: list[Exception] = []

        def _emitter():
            try:
                for _ in range(25):
                    hub.emit_settling(reason="motion_settling")
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=_emitter) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        # 4 threads × 25 calls = 100 emissions
        assert hub.get_stats()["settling_events_emitted"] == 100

        for q in queues:
            hub.unregister_sse_queue(q)
