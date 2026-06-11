"""Unit tests for dsp.classifiers — all 8 regions + s-space."""

from __future__ import annotations

from neurolink.dsp.classifiers import classify_v01, classify_v2, compute_s_space
from neurolink.models.eeg import BandPowers


def _bands(alpha=0.0, theta=0.0, beta=0.0, delta=0.0, gamma=0.0) -> BandPowers:
    return BandPowers(alpha=alpha, theta=theta, beta=beta, delta=delta, gamma=gamma)


# ── v2 classifier ────────────────────────────────────────────────────────────

def test_v2_nigredo_default():
    region, stage = classify_v2(_bands(alpha=0.1, theta=0.05, beta=0.1, delta=0.1, gamma=0.05))
    assert region == "A"
    assert stage == "Nigredo"


def test_v2_coagulatio_heavy_delta():
    region, stage = classify_v2(_bands(delta=0.50))
    assert region == "F"
    assert stage == "Coagulatio"


def test_v2_sublimatio_gamma_dominant():
    region, stage = classify_v2(_bands(gamma=0.25, alpha=0.05, theta=0.05, beta=0.05))
    assert region == "G"
    assert stage == "Sublimatio"


def test_v2_calcinatio_high_beta():
    region, stage = classify_v2(_bands(beta=0.45, delta=0.1))
    assert region == "H"
    assert stage == "Calcinatio"


def test_v2_multiplicatio():
    region, stage = classify_v2(_bands(alpha=0.35, theta=0.18, beta=0.10, delta=0.1, gamma=0.05))
    assert region == "E"
    assert stage == "Multiplicatio"


def test_v2_rubedo():
    region, stage = classify_v2(_bands(alpha=0.31, theta=0.16, beta=0.10, delta=0.1, gamma=0.05))
    assert region == "E"
    assert stage == "Rubedo"


def test_v2_solutio_high_theta():
    region, stage = classify_v2(_bands(theta=0.30, alpha=0.10, delta=0.1))
    assert region == "D"
    assert stage == "Solutio"


def test_v2_albedo_moderate_beta():
    region, stage = classify_v2(_bands(beta=0.30, delta=0.1, gamma=0.05))
    assert region == "C"
    assert stage == "Albedo"


# ── v01 classifier ───────────────────────────────────────────────────────────

def test_v01_region_e_rubedo():
    region, stage = classify_v01(alpha=0.32, theta=0.16, beta=0.10, delta=0.1, gamma=0.05)
    assert region == "E"
    assert stage == "Rubedo"


def test_v01_region_f_delta():
    region, stage = classify_v01(alpha=0.1, theta=0.1, beta=0.1, delta=0.55, gamma=0.05)
    assert region == "F"
    assert stage == "Coagulatio"


def test_v01_region_a_default():
    region, stage = classify_v01(alpha=0.1, theta=0.05, beta=0.05, delta=0.1, gamma=0.05)
    assert region == "A"
    assert stage == "Nigredo"


# ── S-space ──────────────────────────────────────────────────────────────────

def test_s_space_coords_in_range():
    coords = compute_s_space(_bands(alpha=0.3, theta=0.2, beta=0.15, delta=0.2, gamma=0.1))
    assert 0.0 <= coords.x <= 10.0
    assert 0.0 <= coords.y <= 10.0
    assert 0.0 <= coords.z <= 1.0


def test_s_space_zero_bands():
    coords = compute_s_space(_bands())
    assert coords.x >= 0.0
    assert coords.y == 0.0
