"""Focus state classifier.

Ported from Rigpa-v3 focus_state.py.
"""
from __future__ import annotations

from enum import Enum


class FocusState(str, Enum):
    """Focus state levels derived from EA-1 score."""

    HIGH_FOCUS = "HIGH_FOCUS"
    MODERATE_FOCUS = "MODERATE_FOCUS"
    LOW_FOCUS = "LOW_FOCUS"
    DISTRACTED = "DISTRACTED"
    UNKNOWN = "unknown"


# Thresholds for focus state classification
_HIGH_FOCUS_THRESHOLD: float = 0.75
_MODERATE_FOCUS_THRESHOLD: float = 0.50
_LOW_FOCUS_THRESHOLD: float = 0.25


def classify_focus(score: float) -> FocusState:
    """Classify focus state from a normalised score in [0, 1].

    Args:
        score: focus score in [0, 1]

    Returns:
        FocusState enum value
    """
    if score >= _HIGH_FOCUS_THRESHOLD:
        return FocusState.HIGH_FOCUS
    elif score >= _MODERATE_FOCUS_THRESHOLD:
        return FocusState.MODERATE_FOCUS
    elif score >= _LOW_FOCUS_THRESHOLD:
        return FocusState.LOW_FOCUS
    else:
        return FocusState.DISTRACTED


def is_blocking(state: FocusState) -> bool:
    """Return True if the focus state should block advanced protocols."""
    return state in (FocusState.LOW_FOCUS, FocusState.DISTRACTED, FocusState.UNKNOWN)
