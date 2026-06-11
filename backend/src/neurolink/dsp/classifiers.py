"""EEG classifiers v0.1 and v2.

v0.1: rule-based 6-region classifier ported from Rigpa-v2 classifier_v01.py.
v2: extended 8-region classifier ported from Rigpa-v3 classifiers.py.
S-space projection for visualisation.

All functions are pure; no side effects.
"""

from __future__ import annotations

from neurolink.models.eeg import BandPowers, SSpaceCoords

# ============================================================================
# v0.1 Classifier (6 regions: A-F)
# ============================================================================
# Threshold table (FIXED — from Rigpa-v2 classifier_v01.py)

_V01_ALPHA_E: float = 0.30
_V01_THETA_E: float = 0.15
_V01_ALPHA_D: float = 0.22
_V01_THETA_D: float = 0.18
_V01_ALPHA_C: float = 0.22
_V01_BETA_B: float = 0.30
_V01_DELTA_F: float = 0.50
_V01_GAMMA_G: float = 0.20
_V01_MULTIPLICATIO_ALPHA: float = 0.35
_V01_MULTIPLICATIO_THETA: float = 0.15
_V01_MULTIPLICATIO_FAA: float = -0.05


def classify_v01(
    alpha: float,
    theta: float,
    beta: float,
    delta: float,
    gamma: float,
    faa: float | None = None,
    fmt: float | None = None,
) -> tuple[str, str]:
    """Classify EEG state using v0.1 rule-based 6-region model.

    Args:
        alpha: Alpha band power fraction
        theta: Theta band power fraction
        beta: Beta band power fraction
        delta: Delta band power fraction
        gamma: Gamma band power fraction
        faa: Frontal Alpha Asymmetry (optional, used for Multiplicatio gate)
        fmt: Frontal Midline Theta (optional, currently unused)

    Returns:
        (region_label, alchemical_stage) tuple.
    """
    # Region F: deep sleep / delta dominance
    if delta >= _V01_DELTA_F:
        return "F", "Coagulatio"

    # Region E: deep meditation (high alpha + theta)
    if alpha >= _V01_ALPHA_E and theta >= _V01_THETA_E:
        # Escalation to Multiplicatio: very high alpha + faa gate
        if (
            alpha >= _V01_MULTIPLICATIO_ALPHA
            and theta >= _V01_MULTIPLICATIO_THETA
            and (faa is None or faa >= _V01_MULTIPLICATIO_FAA)
        ):
            return "E", "Multiplicatio"
        return "E", "Rubedo"

    # Region D: flow state (theta-dominant)
    if theta >= _V01_THETA_D and alpha < _V01_ALPHA_E:
        return "D", "Citrinitas"

    # Region C: alpha onset (moderate alpha)
    if alpha >= _V01_ALPHA_C and beta < _V01_BETA_B:
        return "C", "Albedo"

    # Region B: active/aroused (high beta)
    if beta >= _V01_BETA_B:
        return "B", "Albedo"

    # Region A: default / mixed
    return "A", "Nigredo"


# ============================================================================
# v2 Classifier (8 regions: A-H)
# ============================================================================

_V2_STAGES: dict[str, tuple[str, str]] = {
    # (region, stage)
    "Multiplicatio": ("E", "Multiplicatio"),
    "Rubedo": ("E", "Rubedo"),
    "Citrinitas": ("D", "Citrinitas"),
    "Solutio": ("D", "Solutio"),
    "Albedo": ("C", "Albedo"),
    "Nigredo": ("A", "Nigredo"),
    "Coagulatio": ("F", "Coagulatio"),
    "Sublimatio": ("G", "Sublimatio"),
    "Calcinatio": ("H", "Calcinatio"),
}

# v2 thresholds (FIXED)
_V2_ALPHA_RUBEDO: float = 0.30
_V2_THETA_RUBEDO: float = 0.15
_V2_BETA_RUBEDO_MAX: float = 0.25
_V2_ALPHA_MULTIPLICATIO: float = 0.33
_V2_BETA_ALBEDO: float = 0.28
_V2_THETA_SOLUTIO: float = 0.25
_V2_DELTA_COAGULATIO: float = 0.45
_V2_GAMMA_SUBLIMATIO: float = 0.20
_V2_BETA_CALCINATIO: float = 0.40


def classify_v2(bands: BandPowers) -> tuple[str, str]:
    """Classify EEG state using v2 extended 8-region model.

    Args:
        bands: BandPowers instance

    Returns:
        (region_label, alchemical_stage) tuple.
    """
    a, th, b, d, g = bands.alpha, bands.theta, bands.beta, bands.delta, bands.gamma

    # Coagulatio: heavy delta (sleep/drowsiness)
    if d >= _V2_DELTA_COAGULATIO:
        return "F", "Coagulatio"

    # Sublimatio: gamma-dominant (high cognitive load)
    if g >= _V2_GAMMA_SUBLIMATIO and g > a and g > th:
        return "G", "Sublimatio"

    # Calcinatio: very high beta (anxiety/hyperarousal)
    if b >= _V2_BETA_CALCINATIO:
        return "H", "Calcinatio"

    # Multiplicatio: highest meditation state
    if a >= _V2_ALPHA_MULTIPLICATIO and th >= _V2_THETA_RUBEDO and b <= _V2_BETA_RUBEDO_MAX:
        return "E", "Multiplicatio"

    # Rubedo: deep meditation
    if a >= _V2_ALPHA_RUBEDO and th >= _V2_THETA_RUBEDO and b <= _V2_BETA_RUBEDO_MAX:
        return "E", "Rubedo"

    # Solutio: high theta (deep focus/flow)
    if th >= _V2_THETA_SOLUTIO and a < _V2_ALPHA_RUBEDO:
        return "D", "Solutio"

    # Albedo: moderate beta (relaxed alertness)
    if b >= _V2_BETA_ALBEDO:
        return "C", "Albedo"

    # Citrinitas: balanced alpha-theta
    if a >= 0.20 and th >= 0.10:
        return "D", "Citrinitas"

    # Nigredo: default
    return "A", "Nigredo"


# ============================================================================
# S-Space Projection
# ============================================================================


def compute_s_space(bands: BandPowers) -> SSpaceCoords:
    """Project band powers into 3D S-space coordinates.

    S-space: 2D mandala coordinate system derived from EEG band ratios.

    X-axis: engagement index (beta-alpha balance), 0-10
    Y-axis: integration coverage (alpha-theta coherence), 0-10
    Z-axis: gamma index (cognitive load), 0-1

    Args:
        bands: BandPowers instance

    Returns:
        SSpaceCoords(x, y, z)
    """
    a, th, b, d, g = bands.alpha, bands.theta, bands.beta, bands.delta, bands.gamma

    # X: engagement = beta / (alpha + 1e-6), normalised to [0, 10]
    engagement_raw = b / (a + 1e-6)
    x = float(min(10.0, max(0.0, engagement_raw * 3.0)))

    # Y: integration = alpha * theta ratio, normalised to [0, 10]
    integration_raw = (a * th) / (b + d + 1e-6)
    y = float(min(10.0, max(0.0, integration_raw * 20.0)))

    # Z: gamma index, normalised to [0, 1]
    z = float(min(1.0, g / (g + a + 1e-6)))

    return SSpaceCoords(x=x, y=y, z=z)
