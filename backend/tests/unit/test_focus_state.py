"""Unit tests for focus_state.py."""

from __future__ import annotations

from neurolink.focus_state import FocusState, classify_focus, compute_focus_score


def test_classify_focus_high_above_075():
    state = classify_focus(0.80)
    assert state == FocusState.HIGH_FOCUS


def test_classify_focus_moderate():
    state = classify_focus(0.60)
    assert state == FocusState.MODERATE_FOCUS


def test_classify_focus_low():
    state = classify_focus(0.35)
    assert state == FocusState.LOW_FOCUS


def test_classify_focus_distracted_below_025():
    state = classify_focus(0.15)
    assert state == FocusState.DISTRACTED


def test_classify_focus_boundary_at_075():
    state = classify_focus(0.75)
    assert state == FocusState.HIGH_FOCUS


def test_compute_focus_score_in_range():
    s = compute_focus_score(bands_alpha=0.2, bands_beta=0.15, baseline_alpha=0.3)
    assert 0.0 <= s <= 1.0


def test_compute_focus_score_high_beta_reduces_focus():
    s_low_beta = compute_focus_score(0.2, 0.05, 0.3)
    s_high_beta = compute_focus_score(0.2, 0.45, 0.3)
    assert s_low_beta > s_high_beta
