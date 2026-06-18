"""Branch-coverage tests for hub.py."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from neurolink.hub import EEGHub, get_ea1, get_hub, get_state, reset, snapshot, update
from neurolink.models.eeg import BandPowers, IngestPayload


def _payload(
    source: str = "mock",
    alpha: float = 0.3,
    theta: float = 0.2,
    beta: float = 0.15,
    delta: float = 0.1,
    gamma: float = 0.05,
    faa: float | None = None,
    fmt: float | None = None,
    ppg=None,
    breathing=None,
    imu=None,
) -> IngestPayload:
    return IngestPayload(
        source=source,
        address="mock",
        timestamp=0.0,
        bands=BandPowers(alpha=alpha, theta=theta, beta=beta, delta=delta, gamma=gamma),
        poor_contact=False,
        faa=faa,
        fmt=fmt,
        ppg=ppg,
        breathing=breathing,
        imu=imu,
    )


# ---------------------------------------------------------------------------
# update() — basic path, frame_count increments
# ---------------------------------------------------------------------------


def test_update_increments_frame_count():
    hub = EEGHub()
    p = _payload()
    state = hub.update(p)
    assert state.frame_count == 1
    state2 = hub.update(p)
    assert state2.frame_count == 2


# ---------------------------------------------------------------------------
# update() — muse_ble branch fires v0.1 classifier
# ---------------------------------------------------------------------------


def test_update_muse_ble_uses_v01_classifier():
    hub = EEGHub()
    p = _payload(source="muse_ble", alpha=0.3, theta=0.2, beta=0.15, delta=0.1, gamma=0.05)
    state = hub.update(p)
    # region_v01 / alchemical_stage_v01 should be set (not default fallback)
    assert state.region_v01 is not None
    assert state.alchemical_stage_v01 is not None


# ---------------------------------------------------------------------------
# update() — None ppg / breathing / imu branches (no AttributeError)
# ---------------------------------------------------------------------------


def test_update_none_ppg_imu_breathing():
    hub = EEGHub()
    state = hub.update(_payload(ppg=None, breathing=None, imu=None))
    assert state.hr_bpm is None
    assert state.hrv_rmssd is None
    assert state.rr_bpm is None
    assert state.pitch_deg is None
    assert state.roll_deg is None
    assert state.motion_rms is None


# ---------------------------------------------------------------------------
# get_latest / set_latest_sample
# ---------------------------------------------------------------------------


def test_get_latest_initially_none():
    hub = EEGHub()
    assert hub.get_latest() is None


def test_set_and_get_latest_sample():
    hub = EEGHub()
    sample = MagicMock()
    hub.set_latest_sample(sample)
    assert hub.get_latest() is sample


# ---------------------------------------------------------------------------
# SSE register / unregister / fanout
# ---------------------------------------------------------------------------


def test_register_and_unregister_sse_queue():
    hub = EEGHub()
    q = asyncio.Queue()
    hub.register_sse_queue(q)
    assert q in hub._sse_queues
    hub.unregister_sse_queue(q)
    assert q not in hub._sse_queues


def test_unregister_nonexistent_queue_no_error():
    hub = EEGHub()
    q = asyncio.Queue()
    hub.unregister_sse_queue(q)  # must not raise ValueError


def test_fanout_delivers_state_to_queue():
    hub = EEGHub()
    q = asyncio.Queue(maxsize=16)
    hub.register_sse_queue(q)
    hub.update(_payload())
    assert not q.empty()
    item = q.get_nowait()
    assert item.frame_count == 1


def test_fanout_queue_full_does_not_raise():
    """When a queue is full, put_nowait raises QueueFull — hub must not propagate it."""
    hub = EEGHub()
    q = asyncio.Queue(maxsize=1)
    q.put_nowait("filler")  # fill the queue
    hub.register_sse_queue(q)
    # update() calls _fanout() which will hit QueueFull; must not raise
    hub.update(_payload())


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------


def test_reset_clears_state():
    hub = EEGHub()
    hub.update(_payload())
    hub.reset()
    assert hub.get_state().frame_count == 0
    assert hub.get_latest() is None
    assert hub.baseline_alpha == 0.30


# ---------------------------------------------------------------------------
# snapshot()
# ---------------------------------------------------------------------------


def test_snapshot_returns_dict():
    hub = EEGHub()
    d = hub.snapshot()
    assert isinstance(d, dict)
    assert "frame_count" in d


# ---------------------------------------------------------------------------
# get_ea1()
# ---------------------------------------------------------------------------


def test_get_ea1_returns_ea1_result():
    from neurolink.models.eeg import EA1Result

    hub = EEGHub()
    result = hub.get_ea1()
    assert isinstance(result, EA1Result)


# ---------------------------------------------------------------------------
# Module-level delegates
# ---------------------------------------------------------------------------


def test_module_level_get_state():
    state = get_state()
    assert hasattr(state, "frame_count")


def test_module_level_get_ea1():
    from neurolink.models.eeg import EA1Result

    assert isinstance(get_ea1(), EA1Result)


def test_module_level_snapshot():
    d = snapshot()
    assert isinstance(d, dict)


def test_module_level_reset():
    reset()
    assert get_state().frame_count == 0


def test_module_level_update():
    reset()
    state = update(_payload())
    assert state.frame_count >= 1


def test_get_hub_returns_singleton():
    h1 = get_hub()
    h2 = get_hub()
    assert h1 is h2
