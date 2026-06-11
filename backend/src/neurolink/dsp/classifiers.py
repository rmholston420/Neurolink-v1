"""EEG state classifiers.

Dual classifier system:
- v0.1: 6-region S-space classifier (Rigpa-v2 muse_compute.classify)
- v2: 8-region alchemical classifier (Rigpa-v2 classifier.py)

All functions are pure.
"""
from __future__ import annotations

from neurolink.models.eeg import BandPowers, SSpaceCoords

# ── v0.1: 6-region S-space classifier ──────────────────────────────────────
#
# Region map (6 regions A-F):
#  A: Nigredo    - default / undefined state
#  B: Albedo     - beta-dominant (active mind)
#  C: Albedo     - alpha settling
#  D: Citrinitas - alpha+theta flow
#  E: Rubedo     - deep alpha+theta (contemplative)
#  F: Coagulatio - delta-dominant (sleep/artifact)
#
# Stage escalation within region E:
#  Rubedo -> Multiplicatio when alpha>=0.35, theta>=0.15, faa>=-0.05

_V01_BETA_THRESHOLD: float = 0.30
_V01_ALPHA_C: float = 0.25
_V01_ALPHA_D: float = 0.28
_V01_ALPHA_E: float = 0.30
_V01_THETA_D: float = 0.20
_V01_THETA_E: float = 0.15
_V01_DELTA_F: float = 0.50
_V01_ALPHA_MULT: float = 0.35
_V01_THETA_MULT: float = 0.15
_V01_FAA_MULT: float = -0.05


def classify_v01(
    alpha: float,
    theta: float,
    beta: float,
    delta: float,
    gamma: float,
    faa: float | None = None,
    fmt: float | None = None,
) -> tuple[str, str]:
    """6-region S-space v0.1 classifier.

    Returns:
        (region, alchemical_stage) tuple
    """
    # F: delta dominates (sleep/artifact)
    if delta > _V01_DELTA_F:
        return "F", "Coagulatio"

    # B: beta dominant (active/distracted)
    if beta > _V01_BETA_THRESHOLD:
        return "B", "Albedo"

    # E: deep contemplative state
    if alpha >= _V01_ALPHA_E and theta >= _V01_THETA_E:
        # Check for Multiplicatio escalation
        if (
            alpha >= _V01_ALPHA_MULT
            and theta >= _V01_THETA_MULT
            and (faa is None or faa >= _V01_FAA_MULT)
        ):
            return "E", "Multiplicatio"
        return "E", "Rubedo"

    # D: alpha+theta flow
    if alpha >= _V01_ALPHA_D and theta >= _V01_THETA_D:
        return "D", "Citrinitas"

    # C: alpha settling
    if alpha >= _V01_ALPHA_C:
        return "C", "Albedo"

    # A: default
    return "A", "Nigredo"


# ── v2: 8-region alchemical classifier ─────────────────────────────────────
#
# 8 alchemical stages:
#  Nigredo: low-signal baseline
#  Albedo: beta-dominant, agitated
#  Citrinitas: alpha rising
#  Rubedo: balanced alpha+theta+low-beta
#  Multiplicatio: deep flow (Rubedo + faa > 0)
#  Coagulatio: delta surge
#  Sublimatio: gamma burst
#  Solutio: theta-dominant (early meditation)

_V2_ALPHA_RUBEDO: float = 0.30
_V2_THETA_RUBEDO: float = 0.15
_V2_BETA_RUBEDO_MAX: float = 0.20
_V2_BETA_ALBEDO: float = 0.30
_V2_ALPHA_CITRINITAS: float = 0.22
_V2_DELTA_COAGULATIO: float = 0.45
_V2_GAMMA_SUBLIMATIO: float = 0.20
_V2_THETA_SOLUTIO: float = 0.25
_V2_ALPHA_MULTIPLICATIO: float = 0.32

# Region to stage mapping for v2
_V2_REGION_MAP: dict[str, str] = {
    "Nigredo": "A",
    "Albedo": "B",
    "Citrinitas": "C",
    "Rubedo": "D",
    "Multiplicatio": "E",
    "Coagulatio": "F",
    "Sublimatio": "G",
    "Solutio": "H",
}


def classify_v2(bands: BandPowers) -> tuple[str, str]:
    """8-region alchemical v2 classifier.

    Returns:
        (region, alchemical_stage) tuple
    """
    alpha = bands.alpha
    theta = bands.theta
    beta = bands.beta
    delta = bands.delta
    gamma = bands.gamma

    # Coagulatio: delta surge
    if delta > _V2_DELTA_COAGULATIO:
        return "F", "Coagulatio"

    # Sublimatio: gamma burst
    if gamma > _V2_GAMMA_SUBLIMATIO:
        return "G", "Sublimatio"

    # Albedo: beta dominant
    if beta > _V2_BETA_ALBEDO:
        return "B", "Albedo"

    # Rubedo: balanced alpha+theta (deep meditation)
    if alpha >= _V2_ALPHA_RUBEDO and theta >= _V2_THETA_RUBEDO and beta <= _V2_BETA_RUBEDO_MAX:
        # Multiplicatio escalation
        if alpha >= _V2_ALPHA_MULTIPLICATIO:
            return "E", "Multiplicatio"
        return "D", "Rubedo"

    # Solutio: theta dominant
    if theta > _V2_THETA_SOLUTIO:
        return "H", "Solutio"

    # Citrinitas: alpha rising
    if alpha >= _V2_ALPHA_CITRINITAS:
        return "C", "Citrinitas"

    # Nigredo: baseline
    return "A", "Nigredo"


def compute_s_space(bands: BandPowers) -> SSpaceCoords:
    """Compute S-space coordinates from band powers.

    x = engagement index  = beta / (alpha + theta)
    y = integration cover = alpha / beta
    z = theta fraction    = theta (raw)

    Returns:
        SSpaceCoords with x, y, z values in [0, inf) (clamped to [0, 10]).
    """
    _eps = 1e-6
    x = bands.beta / (bands.alpha + bands.theta + _eps)
    y = bands.alpha / (bands.beta + _eps)
    z = bands.theta
    # Clamp to reasonable range
    x = min(max(x, 0.0), 10.0)
    y = min(max(y, 0.0), 10.0)
    z = min(max(z, 0.0), 1.0)
    return SSpaceCoords(x=x, y=y, z=z)
