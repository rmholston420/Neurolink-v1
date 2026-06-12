"""Focus state classifier: maps focus score to FocusState enum.

Ported from Rigpa-v3 focus_state.py.
All functions are pure.
"""

from __future__ import annotations

from enum import StrEnum

import structlog

log = structlog.get_logger(__name__)


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

# Minimum focus score required for EEG gate to pass through
_BLOCKING_THRESHOLD: float = _LOW_FOCUS_THRESHOLD

# Module-level focus score cache (written by hub, read by EEG gate)
_current_focus_score: float = 0.0


def set_current_focus_score(score: float) -> None:
    """Update the cached focus score (called by hub.update on every frame)."""
    global _current_focus_score
    _current_focus_score = float(score)


def is_blocking() -> bool:
    """Return True if the current focus score is below the blocking threshold.

    Used by the EEG gate middleware to decide whether to gate output.
    A low focus score (DISTRACTED state) triggers the gate.

    Returns:
        True if focus is too low (gate should block), False if focus passes.
    """
    return _current_focus_score < _BLOCKING_THRESHOLD


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

    In contemplative practice, focused meditation is characterised by
    alpha relative to the subject's own baseline (present, settled mind)
    combined with beta engagement (active, clear attention).  High beta
    with moderate alpha is the hallmark of focused, non-drowsy practice;
    low alpha AND low beta together indicate distraction or drowsiness.

    Score = alpha_ratio_component * 0.6 + beta_engagement * 0.4

    where:
      alpha_ratio_component = min(bands_alpha / baseline_alpha, 1.5) / 1.5
          (alpha relative to calibrated baseline, capped to avoid overflow)
      beta_engagement       = min(bands_beta, 0.5) / 0.5
          (normalised beta, capped at 0.5 — above that is noise/artefact)

    Args:
        bands_alpha: Current alpha band power fraction in [0, 1]
        bands_beta:  Current beta band power fraction in [0, 1]
        baseline_alpha: Per-subject alpha baseline from calibration

    Returns:
        Focus score in [0, 1].
    """
    if baseline_alpha <= 0:
        baseline_alpha = 0.3  # sensible fallback

    # Alpha relative to subject baseline (capped at 1.5× to stay in [0, 1])
    alpha_ratio = bands_alpha / (baseline_alpha + 1e-6)
    alpha_component = min(alpha_ratio, 1.5) / 1.5

    # Beta engagement: active attention indicator
    beta_engagement = min(bands_beta, 0.5) / 0.5

    score = (alpha_component * 0.6) + (beta_engagement * 0.4)
    return float(max(0.0, min(1.0, score)))
