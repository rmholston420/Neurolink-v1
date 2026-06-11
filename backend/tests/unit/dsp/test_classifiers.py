"""Unit tests for dsp/classifiers.py."""
from __future__ import annotations

import pytest

from neurolink.dsp.classifiers import classify_v01, classify_v2, compute_s_space
from neurolink.models.eeg import BandPowers


# ── classify_v01 tests ───────────────────────────────────────────────────────

def test_classify_v01_region_A_default():
    region, stage = classify_v01(alpha=0.1, theta=0.05, beta=0.1, delta=0.1, gamma=0.05)
    assert region == "A"
    assert stage == "Nigredo"


def test_classify_v01_region_B_high_beta():
    region, stage = classify_v01(alpha=0.1, theta=0.05, beta=0.35, delta=0.1, gamma=0.05)
    assert region == "B"
    assert stage == "Albedo"


def test_classify_v01_region_E_for_high_alpha_theta():
    region, stage = classify_v01(alpha=0.32, theta=0.18, beta=0.10, delta=0.1, gamma=0.05)
    assert region == "E"
    assert stage in ("Rubedo", "Multiplicatio")


def test_classify_v01_multiplicatio_escalation():
    """Should escalate to Multiplicatio with alpha>=0.35, theta>=0.15, faa>=-0.05."""
    region, stage = classify_v01(
        alpha=0.38, theta=0.18, beta=0.08, delta=0.1, gamma=0.02, faa=0.0
    )
    assert region == "E"
    assert stage == "Multiplicatio"


def test_classify_v01_region_D_flow():
    region, stage = classify_v01(alpha=0.29, theta=0.21, beta=0.10, delta=0.1, gamma=0.05)
    assert region == "D"
    assert stage == "Citrinitas"


def test_classify_v01_region_C_alpha_settling():
    region, stage = classify_v01(alpha=0.26, theta=0.10, beta=0.10, delta=0.1, gamma=0.05)
    assert region == "C"
    assert stage == "Albedo"


def test_classify_v01_region_F_for_delta_gt_50_pct():
    region, stage = classify_v01(alpha=0.05, theta=0.05, beta=0.05, delta=0.55, gamma=0.02)
    assert region == "F"
    assert stage == "Coagulatio"


# ── classify_v2 tests ───────────────────────────────────────────────────────

def test_classify_v02_nigredo_default():
    bands = BandPowers(alpha=0.1, theta=0.05, beta=0.1, delta=0.1, gamma=0.05)
    region, stage = classify_v2(bands)
    assert stage == "Nigredo"


def test_classify_v02_albedo_high_beta():
    bands = BandPowers(alpha=0.1, theta=0.05, beta=0.35, delta=0.1, gamma=0.05)
    region, stage = classify_v2(bands)
    assert stage == "Albedo"


def test_classify_v02_rubedo_threshold():
    """Rubedo: alpha>=0.30, theta>=0.15, beta<=0.20."""
    bands = BandPowers(alpha=0.31, theta=0.16, beta=0.15, delta=0.1, gamma=0.05)
    region, stage = classify_v2(bands)
    assert stage in ("Rubedo", "Multiplicatio")


def test_classify_v02_multiplicatio():
    bands = BandPowers(alpha=0.35, theta=0.16, beta=0.12, delta=0.1, gamma=0.05)
    region, stage = classify_v2(bands)
    assert stage == "Multiplicatio"


def test_classify_v02_coagulatio():
    bands = BandPowers(alpha=0.05, theta=0.05, beta=0.05, delta=0.50, gamma=0.02)
    region, stage = classify_v2(bands)
    assert stage == "Coagulatio"


def test_classify_v02_sublimatio():
    bands = BandPowers(alpha=0.05, theta=0.05, beta=0.05, delta=0.05, gamma=0.25)
    region, stage = classify_v2(bands)
    assert stage == "Sublimatio"


def test_classify_v02_solutio():
    bands = BandPowers(alpha=0.1, theta=0.30, beta=0.1, delta=0.1, gamma=0.05)
    region, stage = classify_v2(bands)
    assert stage == "Solutio"


# ── compute_s_space ────────────────────────────────────────────────────────

def test_compute_s_space_values_in_range():
    bands = BandPowers(alpha=0.3, theta=0.2, beta=0.15, delta=0.25, gamma=0.1)
    coords = compute_s_space(bands)
    assert 0.0 <= coords.x <= 10.0
    assert 0.0 <= coords.y <= 10.0
    assert 0.0 <= coords.z <= 1.0
