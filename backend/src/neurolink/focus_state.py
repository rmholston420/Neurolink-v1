"""Focus state classifier: maps focus score to FocusState enum.

Ported from Rigpa-v3 focus_state.py.
All functions are pure.
"""

from __future__ import annotations

from enum import StrEnum


class FocusState(StrEnum):
    """Focus state classification."""

    HIGH_FOCUS = "HIGH_FOCUS"
    MODERATE_FOCUS = "MODERATE_FOCUS"
    LOW_FOCUS = "LOW_FOCUS"
    DISTRACTED = "DISTRACTED"
    UNKNOWN = "unknown"


# Thresholds (from Rigpa-v3 focus_state.py)
_HIGH_FOCUS_THRESHOLD: float = 0.75
_MODERATE_FOCUS_THRESHOLD: float = 0.50
_LOW_FOCUS_THRESHOLD: float = 0.25


def classify_focus(score: float) -> FocusState:
    """Map a normalised focus score [0, 1] to a FocusState enum.

    Args:
        score: Normalised focus score in [0, 1].

    Returns:
        FocusState enum value.
    """
    if score >= _HIGH_FOCUS_THRESHOLD:
        return FocusState.HIGH_FOCUS
    if score >= _MODERATE_FOCUS_THRESHOLD:
        return FocusState.MODERATE_FOCUS
    if score >= _LOW_FOCUS_THRESHOLD:
        return FocusState.LOW_FOCUS
    return FocusState.DISTRACTED


def compute_focus_score(bands_alpha: float, bands_beta: float, baseline_alpha: float) -> float:
    """Compute a normalised focus score from band powers.

    Focus is driven by alpha suppression relative to baseline,
    penalised by beta (mind-wandering).

    Args:
        bands_alpha: Current alpha band power fraction
        bands_beta: Current beta band power fraction
        baseline_alpha: Per-subject alpha baseline from calibration

    Returns:
        Focus score in [0, 1].
    """
    if baseline_alpha <= 0:
        baseline_alpha = 0.3  # sensible fallback

    # Inverse of alpha ratio (alpha suppression = focus)
    alpha_ratio = bands_alpha / (baseline_alpha + 1e-6)
    alpha_component = max(0.0, 1.0 - alpha_ratio)

    # Beta penalty (high beta = mind-wandering)
    beta_penalty = min(bands_beta, 0.5) / 0.5

    score = (alpha_component * 0.7) + ((1.0 - beta_penalty) * 0.3)
    return float(max(0.0, min(1.0, score)))
