"""EEG state classifiers: v0.1 6-region S-space and v2 8-region alchemical.

Ported verbatim from Rigpa-v2 muse_compute.py and classifier.py.
Region and stage maps are firmware-level constants - do not modify.
"""
from __future__ import annotations

from neurolink.models.eeg import BandPowers, SSpaceCoords

# ──────────────────────────────────────────────────────────────────────
# v0.1 6-Region Classifier (Rigpa-v2 muse_compute.py)
# ──────────────────────────────────────────────────────────────────────

_REGION_TO_STAGE_V01: dict[str, str] = {
    "A": "Nigredo",     # scattered/default
    "B": "Albedo",      # alerting/beta-driven
    "C": "Albedo",      # settling
    "D": "Citrinitas",  # flow
    "E": "Rubedo",      # meditation
    "F": "Coagulatio",  # delta-contaminated
}


def classify_v01(
    alpha: float,
    theta: float,
    beta: float,
    delta: float,
    gamma: float,  # noqa: ARG001 — included for API symmetry
    faa: float | None = None,
    fmt: float | None = None,  # noqa: ARG001
) -> tuple[str, str]:
    """Classify EEG state into one of 6 S-space regions (v0.1).

    Returns (region, alchemical_stage) tuple.
    Multiplicatio escalation: Rubedo -> Multiplicatio when
    alpha >= 0.35 AND theta >= 0.15 AND (faa is None OR faa >= -0.05).
    """
    # Region assignment (order matters — F first, then specificity)
    if delta > 0.50:
        region = "F"
    elif alpha < 0.15 and theta < 0.15:
        region = "A"
    elif beta > 0.35 and alpha < 0.20:
        region = "B"
    elif alpha >= 0.25 and theta < 0.20:
        region = "C"
    elif alpha >= 0.25 and theta >= 0.20 and beta >= 0.15:
        region = "D"
    elif alpha >= 0.30 and theta >= 0.15 and beta < 0.20:
        region = "E"
    else:
        region = "A"

    stage = _REGION_TO_STAGE_V01.get(region, "Nigredo")

    # Multiplicatio escalation
    if (
        region == "E"
        and alpha >= 0.35
        and theta >= 0.15
        and (faa is None or faa >= -0.05)
    ):
        stage = "Multiplicatio"

    return region, stage


# ──────────────────────────────────────────────────────────────────────
# v2 8-Region Alchemical Classifier (Rigpa-v2 classifier.py)
# ──────────────────────────────────────────────────────────────────────

_ALCHEMICAL_THRESHOLDS = [
    # (alpha_min, theta_min, beta_max, stage_name)
    (0.40, 0.20, 0.15, "Coagulatio"),
    (0.35, 0.18, 0.18, "Multiplicatio"),
    (0.30, 0.15, 0.20, "Rubedo"),
    (0.25, 0.15, 0.25, "Citrinitas"),
    (0.20, 0.10, 0.35, "Albedo"),
]

_REGION_V2_MAP = {
    "Coagulatio": "H",
    "Multiplicatio": "G",
    "Rubedo": "F",
    "Citrinitas": "E",
    "Albedo": "D",
    "Nigredo": "A",
}


def compute_s_space(bands: BandPowers) -> SSpaceCoords:
    """Compute S-space coordinates from band powers.

    x = engagement index = beta / (alpha + theta)
    y = integration coverage = alpha / beta
    z = theta fraction (raw)
    """
    denom_x = bands.alpha + bands.theta
    x = bands.beta / denom_x if denom_x > 0 else 0.0
    y = bands.alpha / bands.beta if bands.beta > 0 else 0.0
    z = bands.theta
    return SSpaceCoords(x=x, y=y, z=z)


def classify_alchemical_stage(bands: BandPowers) -> str:
    """Classify band powers to an alchemical stage (v2)."""
    for alpha_min, theta_min, beta_max, stage in _ALCHEMICAL_THRESHOLDS:
        if (
            bands.alpha >= alpha_min
            and bands.theta >= theta_min
            and bands.beta <= beta_max
        ):
            return stage
    # Delta contamination check
    if bands.delta > 0.50:
        return "Coagulatio"
    # Beta-dominant alerting
    if bands.beta > 0.35:
        return "Albedo"
    return "Nigredo"


def classify_region_v2(s_space: SSpaceCoords, stage: str) -> str:
    """Map alchemical stage to an 8-region label."""
    return _REGION_V2_MAP.get(stage, "A")


def classify_v2(bands: BandPowers) -> tuple[str, str]:
    """Run the v2 8-region alchemical classifier.

    Returns (region, alchemical_stage).
    """
    stage = classify_alchemical_stage(bands)
    s_space = compute_s_space(bands)
    region = classify_region_v2(s_space, stage)
    return region, stage
