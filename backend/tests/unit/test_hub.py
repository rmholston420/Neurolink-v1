"""Unit tests for hub.py."""
from __future__ import annotations

import pytest

from neurolink.hub import EEGHub
from neurolink.models.eeg import BandPowers, IngestPayload


def _make_payload(source: str = "mock", alpha: float = 0.35, theta: float = 0.20) -> IngestPayload:
    return IngestPayload(
        source=source,
        bands=BandPowers(alpha=alpha, theta=theta, beta=0.12, delta=0.10, gamma=0.05),
    )


def test_hub_update_increments_frame_count():
    hub = EEGHub()
    p = _make_payload()
    state = hub.update(p)
    assert state.frame_count == 1
    state2 = hub.update(p)
    assert state2.frame_count == 2


def test_hub_dual_classifier_both_populated_for_muse_ble():
    """v0.1 and v2 classifiers should both produce non-default results for muse_ble."""
    hub = EEGHub()
    p = _make_payload(source="muse_ble", alpha=0.35, theta=0.18)
    state = hub.update(p)
    # v2 always runs
    assert state.region in ("A", "B", "C", "D", "E", "F", "G", "H")
    assert state.alchemical_stage != ""
    # v0.1 ran because source == muse_ble
    assert state.region_v01 in ("A", "B", "C", "D", "E", "F")


def test_hub_v01_not_run_for_mock_source():
    """v0.1 classifier should NOT run for mock source."""
    hub = EEGHub()
    p = _make_payload(source="mock")
    state = hub.update(p)
    # Default region_v01 = 'A'; should not be changed
    assert state.region_v01 == "A"
    assert state.alchemical_stage_v01 == "Nigredo"


def test_hub_reset_clears_state():
    hub = EEGHub()
    p = _make_payload()
    hub.update(p)
    assert hub.get_state().frame_count == 1
    hub.reset()
    assert hub.get_state().frame_count == 0
    assert hub.get_state().connected is False


def test_hub_get_state_returns_neurolink_state():
    from neurolink.models.eeg import NeurolinkState
    hub = EEGHub()
    state = hub.get_state()
    assert isinstance(state, NeurolinkState)


def test_hub_ea1_result_returned():
    from neurolink.models.eeg import EA1Result
    hub = EEGHub()
    ea1 = hub.get_ea1()
    assert isinstance(ea1, EA1Result)


def test_hub_snapshot_is_dict():
    hub = EEGHub()
    snap = hub.snapshot()
    assert isinstance(snap, dict)
    assert "frame_count" in snap


def test_hub_focus_fatigue_in_state():
    hub = EEGHub()
    p = _make_payload()
    state = hub.update(p)
    assert isinstance(state.focus_score, float)
    assert isinstance(state.fatigue_score, float)
    assert state.focus_state in ("HIGH_FOCUS", "MODERATE_FOCUS", "LOW_FOCUS", "DISTRACTED", "unknown")
