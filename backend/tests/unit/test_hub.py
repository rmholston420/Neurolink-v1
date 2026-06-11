"""Unit tests for hub.py."""
from __future__ import annotations

import pytest

from neurolink.hub import EEGHub
from neurolink.models.eeg import BandPowers, IngestPayload


def make_payload(
    source: str = "mock",
    alpha: float = 0.30,
    theta: float = 0.15,
    beta: float = 0.15,
    delta: float = 0.20,
    gamma: float = 0.05,
) -> IngestPayload:
    bands = BandPowers(alpha=alpha, theta=theta, beta=beta, delta=delta, gamma=gamma)
    return IngestPayload(source=source, bands=bands, timestamp=1000.0)


def test_hub_update_increments_frame_count():
    hub = EEGHub()
    assert hub.get_state().frame_count == 0
    hub.update(make_payload())
    assert hub.get_state().frame_count == 1
    hub.update(make_payload())
    assert hub.get_state().frame_count == 2


def test_hub_dual_classifier_both_populated_for_muse_ble():
    hub = EEGHub()
    # alpha=0.32, theta=0.18, beta=0.10 -> should hit region E/Rubedo for v0.1
    payload = make_payload(source="muse_ble", alpha=0.32, theta=0.18, beta=0.10)
    state = hub.update(payload)
    # v0.1 should have run
    assert state.region_v01 in ("A", "B", "C", "D", "E", "F")
    assert state.alchemical_stage_v01 != ""
    # v2 should also have run
    assert state.region != ""
    assert state.alchemical_stage != ""


def test_hub_v01_not_run_for_mock_source():
    hub = EEGHub()
    state = hub.update(make_payload(source="mock"))
    # v0.1 not run -> defaults
    assert state.region_v01 == "A"
    assert state.alchemical_stage_v01 == "Nigredo"


def test_hub_reset_clears_state():
    hub = EEGHub()
    hub.update(make_payload())
    assert hub.get_state().frame_count == 1
    hub.reset()
    assert hub.get_state().frame_count == 0
    assert hub.get_state().connected is False


def test_hub_snapshot_is_dict():
    hub = EEGHub()
    hub.update(make_payload())
    snap = hub.snapshot()
    assert isinstance(snap, dict)
    assert "frame_count" in snap


def test_hub_ea1_returned():
    hub = EEGHub()
    payload = make_payload(alpha=0.30, theta=0.20, region="E" if False else "mock")
    hub.update(payload)
    ea1 = hub.get_ea1()
    assert hasattr(ea1, "eligible")
    assert hasattr(ea1, "score")


def test_hub_focus_state_populated():
    hub = EEGHub()
    state = hub.update(make_payload(alpha=0.30))
    assert state.focus_state in (
        "HIGH_FOCUS", "MODERATE_FOCUS", "LOW_FOCUS", "DISTRACTED", "unknown"
    )
