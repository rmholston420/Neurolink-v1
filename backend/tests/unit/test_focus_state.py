"""Unit tests for neurolink.focus_state — pure functions, no I/O."""

from __future__ import annotations

from neurolink.focus_state import (
    FocusState,
    classify_focus,
    compute_focus_score,
    is_blocking,
    set_current_focus_score,
)


def test_classify_focus_high():
    assert classify_focus(0.80) == FocusState.HIGH_FOCUS


def test_classify_focus_moderate():
    assert classify_focus(0.60) == FocusState.MODERATE_FOCUS


def test_classify_focus_low():
    assert classify_focus(0.30) == FocusState.LOW_FOCUS


def test_classify_focus_distracted():
    assert classify_focus(0.10) == FocusState.DISTRACTED


def test_classify_focus_boundary_high():
    assert classify_focus(0.75) == FocusState.HIGH_FOCUS


def test_compute_focus_score_suppressed_alpha():
    score = compute_focus_score(bands_alpha=0.05, bands_beta=0.10, baseline_alpha=0.30)
    assert score > 0.5


def test_compute_focus_score_high_beta_penalty():
    low_beta = compute_focus_score(0.05, 0.05, 0.30)
    high_beta = compute_focus_score(0.05, 0.50, 0.30)
    assert high_beta < low_beta


def test_compute_focus_score_zero_baseline_fallback():
    score = compute_focus_score(0.10, 0.10, 0.0)
    assert 0.0 <= score <= 1.0


def test_is_blocking_when_low():
    set_current_focus_score(0.10)
    assert is_blocking() is True


def test_is_blocking_when_high():
    set_current_focus_score(0.80)
    assert is_blocking() is False
