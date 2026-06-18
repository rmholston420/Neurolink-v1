"""Tests for hub.emit_settling() — updated to match emit_settling(reason=...) signature."""

from __future__ import annotations

import threading
from queue import Queue

import pytest

import neurolink.hub as hub_module
from neurolink.hub import EEGHub

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_hub():
    hub_module.reset()
    yield
    hub_module.reset()


# ---------------------------------------------------------------------------
# emit_settling() — core behaviour
# ---------------------------------------------------------------------------


class TestEmitSettlingSignature:
    """emit_settling(reason=...) must accept all documented reason codes."""

    VALID_REASONS = [
        "impedance_unstable",
        "motion_settling",
        "env_not_ready",
        "settling",
    ]

    @pytest.mark.parametrize("reason", VALID_REASONS)
    def test_known_reason_no_exception(self, reason: str):
        hub = EEGHub()
        hub.emit_settling(reason=reason)  # must not raise

    def test_unknown_reason_forwarded_not_raised(self):
        hub = EEGHub()
        hub.emit_settling(reason="custom_unknown_reason")  # tolerated

    def test_positional_reason_accepted(self):
        """Allow positional call: emit_settling('impedance_unstable')."""
        hub = EEGHub()
        hub.emit_settling("impedance_unstable")  # no raise

    def test_sentinel_is_dict_not_state(self):
        hub = EEGHub()
        q: Queue = Queue(maxsize=0)
        hub.register_sse_client(q)
        hub.emit_settling(reason="settling")
        item = q.get_nowait()
        assert isinstance(item, dict)
        assert item.get("event") == "settling"

    def test_reason_preserved_in_sentinel(self):
        hub = EEGHub()
        q: Queue = Queue(maxsize=0)
        hub.register_sse_client(q)
        hub.emit_settling(reason="impedance_unstable")
        item = q.get_nowait()
        assert item.get("reason") == "impedance_unstable"

    def test_event_key_always_settling(self):
        hub = EEGHub()
        q: Queue = Queue(maxsize=0)
        hub.register_sse_client(q)
        for reason in self.VALID_REASONS:
            hub.emit_settling(reason=reason)
        seen_events = set()
        while not q.empty():
            seen_events.add(q.get_nowait().get("event"))
        assert seen_events == {"settling"}


# ---------------------------------------------------------------------------
# Counter
# ---------------------------------------------------------------------------


class TestSettlingCounter:
    def test_counter_increments_per_call(self):
        hub = EEGHub()
        for i in range(1, 6):
            hub.emit_settling(reason="settling")
            stats = hub.get_stats()
            assert stats["settling_events_emitted"] == i

    def test_counter_resets_with_hub_reset(self):
        hub = EEGHub()
        hub.emit_settling(reason="settling")
        hub.emit_settling(reason="settling")
        hub.reset()
        assert hub.get_stats()["settling_events_emitted"] == 0

    def test_module_level_counter(self):
        hub_module.emit_settling(reason="motion_settling")
        hub_module.emit_settling(reason="motion_settling")
        assert hub_module.get_stats()["settling_events_emitted"] == 2


# ---------------------------------------------------------------------------
# Fan-out
# ---------------------------------------------------------------------------


class TestSettlingFanOut:
    def test_single_client_receives_sentinel(self):
        hub = EEGHub()
        q: Queue = Queue(maxsize=0)
        hub.register_sse_client(q)
        hub.emit_settling(reason="env_not_ready")
        assert not q.empty()
        item = q.get_nowait()
        assert item["event"] == "settling"
        assert item["reason"] == "env_not_ready"

    def test_three_clients_all_receive(self):
        hub = EEGHub()
        queues = [Queue(maxsize=0) for _ in range(3)]
        for q in queues:
            hub.register_sse_client(q)
        hub.emit_settling(reason="motion_settling")
        for q in queues:
            assert not q.empty()
            item = q.get_nowait()
            assert item["event"] == "settling"

    def test_no_clients_silent(self):
        hub = EEGHub()
        hub.emit_settling(reason="settling")  # no queues registered — no exception

    def test_full_queue_dropped_not_raised(self):
        hub = EEGHub()
        q: Queue = Queue(maxsize=1)
        hub.register_sse_client(q)
        q.put_nowait({"event": "dummy"})  # fill the queue
        hub.emit_settling(reason="settling")  # should not raise even though full

    def test_emit_settling_does_not_update_frame_count(self):
        hub = EEGHub()
        hub.emit_settling(reason="settling")
        hub.emit_settling(reason="settling")
        assert hub.get_state().frame_count == 0

    def test_unregistered_client_no_longer_receives(self):
        hub = EEGHub()
        q: Queue = Queue(maxsize=0)
        hub.register_sse_client(q)
        hub.unregister_sse_client(q)
        hub.emit_settling(reason="settling")
        assert q.empty()


# ---------------------------------------------------------------------------
# Distinguish from baseline_complete
# ---------------------------------------------------------------------------


class TestSettlingVsBaseline:
    def test_settling_event_not_confused_with_baseline_complete(self):
        hub = EEGHub()
        q: Queue = Queue(maxsize=0)
        hub.register_sse_client(q)
        hub.emit_settling(reason="impedance_unstable")
        hub.notify_baseline_complete()
        events: list[dict] = []
        while not q.empty():
            events.append(q.get_nowait())
        event_types = [e.get("event") for e in events]
        assert "settling" in event_types
        assert "baseline_complete" in event_types
        assert event_types.count("settling") == 1
        assert event_types.count("baseline_complete") == 1


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestSettlingThreadSafety:
    def test_concurrent_emit_settling_exact_count(self):
        hub = EEGHub()
        n_threads, n_calls = 5, 20
        threads = [
            threading.Thread(
                target=lambda: [hub.emit_settling(reason="settling") for _ in range(n_calls)]
            )
            for _ in range(n_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert hub.get_stats()["settling_events_emitted"] == n_threads * n_calls

    def test_concurrent_register_emit_no_exception(self):
        hub = EEGHub()
        errors: list[Exception] = []

        def register_loop():
            for _ in range(20):
                q: Queue = Queue(maxsize=0)
                hub.register_sse_client(q)

        def emit_loop():
            for _ in range(30):
                try:
                    hub.emit_settling(reason="settling")
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=register_loop) for _ in range(3)] + [
            threading.Thread(target=emit_loop) for _ in range(3)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
