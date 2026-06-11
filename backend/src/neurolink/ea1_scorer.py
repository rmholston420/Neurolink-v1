"""EA-1 multimodal eligibility scorer.

Ported from Rigpa-v2 ea1_scorer.py.
5-criterion gated scorer for advanced contemplative practice protocol.
"""
from __future__ import annotations

from neurolink.models.eeg import EA1Criterion, EA1Result, IngestPayload, SSpaceCoords

# Default thresholds
_ALPHA_THRESHOLD: float = 0.25
_THETA_THRESHOLD: float = 0.15
_MOTION_RMS_MAX: float = 0.5
_CONTACT_QUALITY_MIN: float = 0.5
_S_SPACE_GATE_REGIONS: frozenset[str] = frozenset({"C", "D", "E"})
_ELIGIBILITY_THRESHOLD: float = 0.60  # score >= 0.60 -> eligible (3+ / 5)


def score(payload: IngestPayload) -> EA1Result:
    """Score EA-1 multimodal eligibility from an ingest payload.

    5 criteria:
    1. alpha_power >= 0.25
    2. theta_power >= 0.15
    3. s_space gate: region in {C,D,E} and not poor_contact
    4. motion gate: motion_rms < 0.5 (or None -> pass)
    5. contact_quality >= 0.5 (or None -> pass)

    Score = criteria_met / 5. Eligible if score >= 0.60 (3+).
    """
    criteria: dict[str, EA1Criterion] = {}

    # Criterion 1: alpha power
    alpha_val = payload.bands.alpha
    c1_met = alpha_val >= _ALPHA_THRESHOLD
    criteria["alpha_power"] = EA1Criterion(
        value=alpha_val,
        threshold=_ALPHA_THRESHOLD,
        units="fraction",
        met=c1_met,
    )

    # Criterion 2: theta power
    theta_val = payload.bands.theta
    c2_met = theta_val >= _THETA_THRESHOLD
    criteria["theta_power"] = EA1Criterion(
        value=theta_val,
        threshold=_THETA_THRESHOLD,
        units="fraction",
        met=c2_met,
    )

    # Criterion 3: S-space gate
    s_space_ok = payload.region in _S_SPACE_GATE_REGIONS and not payload.poor_contact
    criteria["s_space_gate"] = EA1Criterion(
        value=None,
        threshold=None,
        units="region",
        met=s_space_ok,
    )

    # Criterion 4: motion gate
    motion_rms = payload.imu.motion_rms if payload.imu is not None else None
    if motion_rms is None:
        c4_met = True  # no IMU data -> pass
    else:
        c4_met = motion_rms < _MOTION_RMS_MAX
    criteria["motion_gate"] = EA1Criterion(
        value=motion_rms,
        threshold=_MOTION_RMS_MAX,
        units="g_rms",
        met=c4_met,
    )

    # Criterion 5: contact quality
    contact_quality = payload.contact_quality
    if contact_quality is None:
        c5_met = True  # no contact quality -> pass
    else:
        c5_met = contact_quality >= _CONTACT_QUALITY_MIN
    criteria["contact_quality"] = EA1Criterion(
        value=contact_quality,
        threshold=_CONTACT_QUALITY_MIN,
        units="fraction",
        met=c5_met,
    )

    criteria_list = [c1_met, c2_met, s_space_ok, c4_met, c5_met]
    criteria_met = sum(criteria_list)
    score_val = criteria_met / 5.0
    eligible = score_val >= _ELIGIBILITY_THRESHOLD
    overlay_mode = f"X{criteria_met}"
    label = "Eligible" if eligible else "Ineligible"

    return EA1Result(
        eligible=eligible,
        score=score_val,
        criteria_met=criteria_met,
        criteria_total=5,
        label=label,
        gates={"s_space": s_space_ok, "motion": c4_met},
        criteria={k: v.model_dump() for k, v in criteria.items()},
        overlay_mode=overlay_mode,
        alchemical_stage=payload.alchemical_stage,
        s_space_coords=payload.s_space,
        s_space_region=payload.region,
        integration_coverage=payload.integration_coverage,
    )
