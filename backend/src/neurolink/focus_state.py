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
    """Return True if the current focus score is below the blocking threshold."""
    return _current_focus_score < _BLOCKING_THRESHOLD


def classify_focus(score: float) -> FocusState:
    """Map a normalised focus score [0, 1] to a FocusState enum."""
    if score >= _HIGH_FOCUS_THRESHOLD:
        return FocusState.HIGH_FOCUS
    if score >= _MODERATE_FOCUS_THRESHOLD:
        return FocusState.MODERATE_FOCUS
    if score >= _LOW_FOCUS_THRESHOLD:
        return FocusState.LOW_FOCUS
    return FocusState.DISTRACTED


def compute_focus_score(
    alpha: float = 0.0,
    beta: float = 0.0,
    baseline_alpha: float = 0.3,
    # Legacy positional aliases kept for hub.py compatibility
    bands_alpha: float | None = None,
    bands_beta: float | None = None,
) -> float:
    """Compute a normalised focus score from band powers.

    Accepts either the new keyword-arg style:
        compute_focus_score(alpha=0.4, beta=0.3, baseline_alpha=0.3)
    or the legacy positional style used by hub.py:
        compute_focus_score(bands.alpha, bands.beta, self.baseline_alpha)

    Args:
        alpha: Current alpha band power fraction in [0, 1]
        beta:  Current beta band power fraction in [0, 1]
        baseline_alpha: Per-subject alpha baseline from calibration
        bands_alpha: Legacy alias for alpha (overrides alpha if provided)
        bands_beta:  Legacy alias for beta (overrides beta if provided)

    Returns:
        Focus score in [0, 1].
    """
    # Resolve legacy aliases
    if bands_alpha is not None:
        alpha = bands_alpha
    if bands_beta is not None:
        beta = bands_beta

    if baseline_alpha <= 0:
        baseline_alpha = 0.3  # sensible fallback

    # Alpha relative to subject baseline (capped at 1.5x to stay in [0, 1])
    alpha_ratio = alpha / (baseline_alpha + 1e-6)
    alpha_component = min(alpha_ratio, 1.5) / 1.5

    # Beta engagement: active attention indicator
    beta_engagement = min(beta, 0.5) / 0.5

    score = (alpha_component * 0.6) + (beta_engagement * 0.4)
    return float(max(0.0, min(1.0, score)))
