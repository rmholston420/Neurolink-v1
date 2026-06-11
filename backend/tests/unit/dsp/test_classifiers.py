"""Unit tests for dsp/classifiers.py."""
from __future__ import annotations

import pytest

from neurolink.dsp.classifiers import classify_v01, classify_v2
from neurolink.models.eeg import BandPowers


# --- v0.1 tests ---

def test_classify_v01_region_A_default():
    region, stage = classify_v01(alpha=0.10, theta=0.10, beta=0.10, delta=0.20, gamma=0.05)
    assert region == "A"
    assert stage == "Nigredo"


def test_classify_v01_region_B_beta_driven():
    region, stage = classify_v01(alpha=0.10, theta=0.05, beta=0.40, delta=0.10, gamma=0.05)
    assert region == "B"
    assert stage == "Albedo"


def test_classify_v01_region_C_settling():
    region, stage = classify_v01(alpha=0.28, theta=0.10, beta=0.10, delta=0.20, gamma=0.05)
    assert region == "C"
    assert stage == "Albedo"


def test_classify_v01_region_D_flow():
    region, stage = classify_v01(alpha=0.30, theta=0.25, beta=0.20, delta=0.10, gamma=0.05)
    assert region == "D"
    assert stage == "Citrinitas"


def test_classify_v01_region_E_for_high_alpha_theta():
    region, stage = classify_v01(alpha=0.32, theta=0.18, beta=0.10, delta=0.15, gamma=0.05)
    assert region == "E"
    assert stage == "Rubedo"


def test_classify_v01_region_F_for_delta_gt_50_pct():
    region, stage = classify_v01(alpha=0.10, theta=0.10, beta=0.10, delta=0.55, gamma=0.05)
    assert region == "F"
    assert stage == "Coagulatio"


def test_classify_v01_multiplicatio_escalation():
    """Rubedo -> Multiplicatio when alpha >= 0.35 and theta >= 0.15 and FAA >= -0.05."""
    region, stage = classify_v01(
        alpha=0.37, theta=0.18, beta=0.10, delta=0.10, gamma=0.05,
        faa=0.1,
    )
    assert region == "E"
    assert stage == "Multiplicatio"


def test_classify_v01_no_multiplicatio_with_negative_faa():
    region, stage = classify_v01(
        alpha=0.37, theta=0.18, beta=0.10, delta=0.10, gamma=0.05,
        faa=-0.20,
    )
    assert region == "E"
    assert stage == "Rubedo"  # no escalation


# --- v2 tests ---

def test_classify_v02_rubedo_threshold():
    bands = BandPowers(alpha=0.32, theta=0.17, beta=0.18, delta=0.20, gamma=0.05)
    region, stage = classify_v2(bands)
    # alpha>=0.30, theta>=0.15, beta<=0.20 -> Rubedo
    assert stage == "Rubedo"


def test_classify_v02_nigredo_default():
    bands = BandPowers(alpha=0.05, theta=0.05, beta=0.10, delta=0.10, gamma=0.02)
    region, stage = classify_v2(bands)
    assert stage == "Nigredo"


def test_classify_v02_albedo_beta_dominant():
    bands = BandPowers(alpha=0.10, theta=0.05, beta=0.40, delta=0.10, gamma=0.05)
    region, stage = classify_v2(bands)
    assert stage == "Albedo"
