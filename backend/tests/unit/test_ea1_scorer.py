"""Unit tests for ea1_scorer.py."""
from __future__ import annotations

import pytest

from neurolink.ea1_scorer import score
from neurolink.models.eeg import BandPowers, IMUPayload, IngestPayload


def _make_payload(**kwargs) -> IngestPayload:
    defaults = dict(
        source="mock",
        bands=BandPowers(alpha=0.35, theta=0.20, beta=0.12, delta=0.10, gamma=0.05),
        region="D",
        alchemical_stage="Rubedo",
        poor_contact=False,
        contact_quality=0.9,
        imu=IMUPayload(motion_rms=0.1),
    )
    defaults.update(kwargs)
    return IngestPayload(**defaults)


def test_ea1_eligible_when_all_criteria_met():
    payload = _make_payload()
    result = score(payload)
    assert result.eligible is True
    assert result.score > 0.0
    assert result.label == "Eligible"


def test_ea1_ineligible_poor_contact():
    payload = _make_payload(poor_contact=True, contact_quality=0.1)
    result = score(payload)
    assert result.eligible is False


def test_ea1_motion_gate_blocks_eligibility():
    payload = _make_payload(imu=IMUPayload(motion_rms=1.0))
    result = score(payload)
    assert result.eligible is False


def test_ea1_ineligible_low_alpha():
    payload = _make_payload(
        bands=BandPowers(alpha=0.10, theta=0.20, beta=0.12, delta=0.10, gamma=0.05)
    )
    result = score(payload)
    assert result.eligible is False


def test_ea1_ineligible_wrong_region():
    payload = _make_payload(region="A", alchemical_stage="Nigredo")
    result = score(payload)
    assert result.eligible is False


def test_ea1_score_proportional_to_criteria_met():
    # 5 criteria met -> score = 1.0
    p5 = _make_payload()
    r5 = score(p5)
    # 1 criterion met -> lower score
    p1 = _make_payload(
        bands=BandPowers(alpha=0.05, theta=0.05, beta=0.05, delta=0.05, gamma=0.05),
        region="A",
        imu=IMUPayload(motion_rms=1.0),
        poor_contact=True,
        contact_quality=0.1,
    )
    r1 = score(p1)
    assert r5.score > r1.score


def test_ea1_criteria_total_is_5():
    payload = _make_payload()
    result = score(payload)
    assert result.criteria_total == 5
