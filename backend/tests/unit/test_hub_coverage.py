"""Coverage gap-filling tests for EEGHub and hub module-level delegates."""

from __future__ import annotations

import asyncio

import neurolink.hub as hub_mod
from neurolink.hub import EEGHub
from neurolink.models.eeg import (
    BandPowers,
    BreathingPayload,
    IMUPayload,
    IngestPayload,
    PPGPayload,
)


def _payload(**kwargs) -> IngestPayload:
    defaults = dict(
        source="mock",
        bands=BandPowers(alpha=0.3, theta=0.2, beta=0.15, delta=0.2, gamma=0.1),
    )
    defaults.update(kwargs)
    return IngestPayload(**defaults)


# ---------------------------------------------------------------------------
# Module-level delegates
# ---------------------------------------------------------------------------

def test_module_update_returns_state():
    state = hub_mod.update(_payload())
    assert state.frame_count >= 1


def test_module_get_state():
    state = hub_mod.get_state()
    assert state is not None


def test_module_get_ea1():
    ea1 = hub_mod.get_ea1()
    assert hasattr(ea1, "eligible")


def test_module_snapshot_returns_dict():
    d = hub_mod.snapshot()
    assert isinstance(d, dict)
    assert "frame_count" in d


def test_module_reset():
    hub_mod.update(_payload())
    hub_mod.reset()
    assert hub_mod.get_state().frame_count == 0


# ---------------------------------------------------------------------------
# set_latest_sample / get_latest
# ---------------------------------------------------------------------------

def test_set_and_get_latest_sample():
    hub = EEGHub()
    assert hub.get_latest() is None
    sample = object()
    hub.set_latest_sample(sample)  # type: ignore[arg-type]
    assert hub.get_latest() is sample


# ---------------------------------------------------------------------------
# _fanout QueueFull drop
# ---------------------------------------------------------------------------

async def test_fanout_full_queue_drops_frame():
    hub = EEGHub()
    q = asyncio.Queue(maxsize=1)
    q.put_nowait(object())  # fill it
    hub.register_sse_queue(q)
    state = hub.update(_payload())
    assert state.frame_count == 1
    assert q.qsize() == 1  # original item still there; new frame dropped


# ---------------------------------------------------------------------------
# _schedule_redis_push — no running loop (sync context)
# ---------------------------------------------------------------------------

def test_schedule_redis_push_no_loop_is_silent():
    hub = EEGHub()
    state = hub.update(_payload())
    assert state.frame_count == 1


# ---------------------------------------------------------------------------
# unregister_sse_queue — queue not registered
# ---------------------------------------------------------------------------

def test_unregister_nonexistent_queue_is_noop():
    hub = EEGHub()
    q = asyncio.Queue()
    hub.unregister_sse_queue(q)  # must not raise


# ---------------------------------------------------------------------------
# hub.update with muse_ble source (v0.1 classifier branch)
# ---------------------------------------------------------------------------

def test_update_muse_ble_source_runs_v01_classifier():
    hub = EEGHub()
    state = hub.update(_payload(source="muse_ble"))
    assert state.region_v01 is not None
    assert state.alchemical_stage_v01 is not None


# ---------------------------------------------------------------------------
# hub.update with ppg / breathing / imu payloads
# ---------------------------------------------------------------------------

def test_update_with_ppg_populates_hr_bpm():
    hub = EEGHub()
    state = hub.update(_payload(ppg=PPGPayload(hr_bpm=65.0, hrv_rmssd=42.0)))
    assert state.hr_bpm == 65.0
    assert state.hrv_rmssd == 42.0


def test_update_with_breathing_populates_rr_bpm():
    hub = EEGHub()
    state = hub.update(_payload(breathing=BreathingPayload(rr_bpm=14.0)))
    assert state.rr_bpm == 14.0


def test_update_with_imu_populates_motion_fields():
    hub = EEGHub()
    state = hub.update(_payload(imu=IMUPayload(pitch_deg=5.0, roll_deg=-3.0, motion_rms=0.02)))
    assert state.pitch_deg == 5.0
    assert state.roll_deg == -3.0
    assert state.motion_rms == 0.02


# ---------------------------------------------------------------------------
# hub.reset clears sample + fatigue
# ---------------------------------------------------------------------------

def test_reset_clears_latest_sample_and_frame_count():
    hub = EEGHub()
    hub.update(_payload())
    hub.set_latest_sample(object())  # type: ignore[arg-type]
    hub.reset()
    assert hub.get_state().frame_count == 0
    assert hub.get_latest() is None
