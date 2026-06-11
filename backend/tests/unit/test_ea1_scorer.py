"""Unit tests for ea1_scorer.py."""
from __future__ import annotations

import pytest

from neurolink.ea1_scorer import score
from neurolink.models.eeg import BandPowers, IMUPayload, IngestPayload


def make_payload(
    alpha: float = 0.30,
    theta: float = 0.20,
    beta: float = 0.10,
    delta: float = 0.15,
    gamma: float = 0.05,
    region: str = "E",
    poor_contact: bool = False,
    motion_rms: float | None = None,
    contact_quality: float | None = None,
) -> IngestPayload:
    bands = BandPowers(alpha=alpha, theta=theta, beta=beta, delta=delta, gamma=gamma)
    imu = IMUPayload(motion_rms=motion_rms) if motion_rms is not None else None
    return IngestPayload(
        bands=bands,
        region=region,
        poor_contact=poor_contact,
        contact_quality=contact_quality,
        imu=imu,
    )


def test_ea1_eligible_when_all_criteria_met():
    payload = make_payload(
        alpha=0.30, theta=0.20, region="E",
        poor_contact=False, motion_rms=0.1, contact_quality=0.9,
    )
    result = score(payload)
    assert result.eligible is True
    assert result.criteria_met == 5
    assert result.score == 1.0
    assert result.label == "Eligible"


def test_ea1_ineligible_poor_contact():
    payload = make_payload(
        alpha=0.30, theta=0.20, region="E",
        poor_contact=True,
    )
    result = score(payload)
    # s_space gate fails when poor_contact=True
    assert result.gates["s_space"] is False


def test_ea1_motion_gate_blocks_eligibility():
    payload = make_payload(
        alpha=0.30, theta=0.20, region="E",
        poor_contact=False, motion_rms=1.0,  # > 0.5 threshold
    )
    result = score(payload)
    assert result.gates["motion"] is False


def test_ea1_ineligible_when_all_criteria_fail():
    payload = make_payload(
        alpha=0.05, theta=0.05, region="A",
        poor_contact=True, motion_rms=2.0, contact_quality=0.1,
    )
    result = score(payload)
    assert result.eligible is False
    assert result.criteria_met == 0
    assert result.score == 0.0


def test_ea1_score_proportional_to_criteria_met():
    # 3 of 5 criteria met -> score = 0.60 -> eligible
    payload = make_payload(
        alpha=0.30, theta=0.20, region="E",
        poor_contact=False, motion_rms=None, contact_quality=None,
    )
    result = score(payload)
    assert result.criteria_met >= 3
    assert result.eligible is True


def test_ea1_overlay_mode():
    payload = make_payload(alpha=0.30, theta=0.20, region="E")
    result = score(payload)
    assert result.overlay_mode.startswith("X")
