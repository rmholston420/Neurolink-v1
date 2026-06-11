"""Unit tests for focus_state.py."""
from __future__ import annotations

import pytest

from neurolink.focus_state import FocusState, classify_focus, is_blocking


def test_classify_focus_high_above_075():
    state = classify_focus(0.80)
    assert state == FocusState.HIGH_FOCUS


def test_classify_focus_moderate_between_050_075():
    state = classify_focus(0.60)
    assert state == FocusState.MODERATE_FOCUS


def test_classify_focus_low_between_025_050():
    state = classify_focus(0.35)
    assert state == FocusState.LOW_FOCUS


def test_classify_focus_distracted_below_025():
    state = classify_focus(0.10)
    assert state == FocusState.DISTRACTED


def test_classify_focus_boundary_exactly_075():
    state = classify_focus(0.75)
    assert state == FocusState.HIGH_FOCUS


def test_classify_focus_boundary_exactly_025():
    state = classify_focus(0.25)
    assert state == FocusState.LOW_FOCUS


def test_is_blocking_low_focus():
    assert is_blocking(FocusState.LOW_FOCUS) is True


def test_is_blocking_distracted():
    assert is_blocking(FocusState.DISTRACTED) is True


def test_is_not_blocking_high_focus():
    assert is_blocking(FocusState.HIGH_FOCUS) is False
