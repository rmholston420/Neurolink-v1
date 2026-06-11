"""EA-1 multimodal eligibility scorer.

Ported from Rigpa-v2 ea1_scorer.py.
5-criterion eligibility system for advanced contemplative protocols.
All functions are pure.
"""
from __future__ import annotations

from neurolink.models.eeg import EA1Criterion, EA1Result, IngestPayload, SSpaceCoords

# ── Thresholds ────────────────────────────────────────────────────────────────────
ALPHA_THRESHOLD: float = 0.30
THETA_THRESHOLD: float = 0.15
MOTION_RMS_GATE: float = 0.5   # above this = too much motion
CONTACT_QUALITY_MIN: float = 0.5  # below this = poor contact

# S-space gating — must be in region D or E for eligibility
_ELIGIBLE_REGIONS = {"D", "E"}

# Overlay mode mapping: criteria_met -> mode label
_OVERLAY_MODES: list[str] = ["X0", "X1", "X2", "X3", "X4", "X5"]

# Alchemical stage -> overlay mode override
_STAGE_OVERLAY: dict[str, str] = {
    "Rubedo": "X4",
    "Multiplicatio": "X5",
    "Citrinitas": "X3",
    "Solutio": "X3",
}


def score(payload: IngestPayload) -> EA1Result:
    """Compute EA-1 eligibility from an IngestPayload.

    Criteria:
    1. alpha_power >= 0.30
    2. theta_power >= 0.15
    3. s_space region in {D, E} (v2 classifier region from payload)
    4. motion_rms < 0.5 (if IMU data present)
    5. contact_quality >= 0.5 (if contact data present)

    Args:
        payload: IngestPayload from hub.update()

    Returns:
        EA1Result with eligibility, score, and per-criterion details.
    """
    bands = payload.bands

    # Criterion 1: alpha power
    alpha_met = bands.alpha >= ALPHA_THRESHOLD
    crit_alpha = EA1Criterion(
        value=bands.alpha,
        threshold=ALPHA_THRESHOLD,
        units="fraction",
        met=alpha_met,
    )

    # Criterion 2: theta power
    theta_met = bands.theta >= THETA_THRESHOLD
    crit_theta = EA1Criterion(
        value=bands.theta,
        threshold=THETA_THRESHOLD,
        units="fraction",
        met=theta_met,
    )

    # Criterion 3: S-space gating
    region = payload.region
    s_space_met = region in _ELIGIBLE_REGIONS
    crit_sspace = EA1Criterion(
        value=None,
        threshold=None,
        units="region",
        met=s_space_met,
    )

    # Criterion 4: motion gating
    motion_rms = payload.imu.motion_rms if payload.imu else 0.0
    motion_met = motion_rms < MOTION_RMS_GATE
    crit_motion = EA1Criterion(
        value=motion_rms,
        threshold=MOTION_RMS_GATE,
        units="g",
        met=motion_met,
    )

    # Criterion 5: contact quality
    contact_quality = payload.contact_quality
    if contact_quality is not None:
        contact_met = not payload.poor_contact and contact_quality >= CONTACT_QUALITY_MIN
    else:
        # No contact data — only fail if poor_contact is explicitly set
        contact_met = not payload.poor_contact
    crit_contact = EA1Criterion(
        value=contact_quality,
        threshold=CONTACT_QUALITY_MIN,
        units="fraction",
        met=contact_met,
    )

    criteria_list = [alpha_met, theta_met, s_space_met, motion_met, contact_met]
    criteria_met = sum(criteria_list)
    total = len(criteria_list)
    eligible = all(criteria_list)

    raw_score = criteria_met / total

    # Gates
    gates = {
        "s_space": s_space_met,
        "motion": motion_met,
    }

    # Overlay mode
    stage = payload.alchemical_stage
    if eligible:
        overlay_mode = _STAGE_OVERLAY.get(stage, f"X{criteria_met}")
    else:
        idx = min(criteria_met, len(_OVERLAY_MODES) - 1)
        overlay_mode = _OVERLAY_MODES[idx]

    label = "Eligible" if eligible else "Ineligible"

    return EA1Result(
        eligible=eligible,
        score=raw_score,
        criteria_met=criteria_met,
        criteria_total=total,
        label=label,
        gates=gates,
        criteria={
            "alpha_power": crit_alpha.model_dump(),
            "theta_power": crit_theta.model_dump(),
            "s_space": crit_sspace.model_dump(),
            "motion": crit_motion.model_dump(),
            "contact_quality": crit_contact.model_dump(),
        },
        overlay_mode=overlay_mode,
        alchemical_stage=stage,
        s_space_coords=payload.s_space,
        s_space_region=region,
        integration_coverage=payload.integration_coverage,
    )
