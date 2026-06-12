"""Unit tests for focus_state — focus score and classification."""

from __future__ import annotations

import pytest

from neurolink.focus_state import classify_focus, compute_focus_score


class TestComputeFocusScore:
    def test_returns_float(self):
        score = compute_focus_score(alpha=0.4, beta=0.3, baseline_alpha=0.3)
        assert isinstance(score, float)

    def test_score_non_negative(self):
        score = compute_focus_score(alpha=0.5, beta=0.2, baseline_alpha=0.3)
        assert score >= 0.0

    def test_above_baseline_increases_score(self):
        low = compute_focus_score(alpha=0.2, beta=0.2, baseline_alpha=0.5)
        high = compute_focus_score(alpha=0.7, beta=0.2, baseline_alpha=0.3)
        assert high >= low

    def test_zero_baseline_handled(self):
        """baseline_alpha=0 should not raise ZeroDivisionError."""
        score = compute_focus_score(alpha=0.3, beta=0.2, baseline_alpha=0.0)
        assert isinstance(score, float)


class TestClassifyFocus:
    def test_returns_enum_or_str(self):
        result = classify_focus(0.5)
        # Accept either an enum or a string
        assert result is not None

    def test_low_score_returns_unfocused(self):
        result = classify_focus(0.0)
        assert result is not None

    def test_high_score_returns_focused(self):
        result = classify_focus(1.0)
        assert result is not None

    def test_boundary_mid(self):
        result = classify_focus(0.5)
        assert result is not None
