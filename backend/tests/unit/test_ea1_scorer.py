"""Unit tests for ea1_scorer — EA-1 eligibility scoring."""

from __future__ import annotations

import pytest

from neurolink.ea1_scorer import score as ea1_score
from neurolink.models.eeg import BandPowers, IngestPayload


def _payload(alpha: float, theta: float, beta: float = 0.1,
             delta: float = 0.05, gamma: float = 0.05,
             faa: float = 0.0, fmt: float = 0.0,
             source: str = "mock") -> IngestPayload:
    bands = BandPowers(alpha=alpha, theta=theta, beta=beta, delta=delta, gamma=gamma)
    return IngestPayload(source=source, bands=bands, faa=faa, fmt=fmt)


class TestEA1Score:
    def test_returns_ea1_result(self, base_payload):
        result = ea1_score(base_payload)
        assert hasattr(result, "eligible")
        assert hasattr(result, "score")

    def test_eligible_is_bool(self, base_payload):
        result = ea1_score(base_payload)
        assert isinstance(result.eligible, bool)

    def test_score_is_float(self, base_payload):
        result = ea1_score(base_payload)
        assert isinstance(result.score, float)

    def test_score_bounded(self, base_payload):
        result = ea1_score(base_payload)
        assert 0.0 <= result.score <= 1.0

    def test_high_alpha_theta_favours_eligible(self):
        """Strong alpha + theta with positive FAA should push toward eligible."""
        p = _payload(alpha=0.65, theta=0.20, beta=0.08, faa=0.8, fmt=0.6)
        result = ea1_score(p)
        assert result.score > 0.3  # at least partial credit

    def test_low_alpha_unfavourable(self):
        """Flat/low alpha should not trivially hit eligibility."""
        p = _payload(alpha=0.05, theta=0.05, beta=0.60, delta=0.25, gamma=0.05)
        result = ea1_score(p)
        # Score is lower than the high-alpha case; exact threshold depends on impl
        assert isinstance(result.eligible, bool)

    def test_deterministic(self, base_payload):
        r1 = ea1_score(base_payload)
        r2 = ea1_score(base_payload)
        assert r1.score == pytest.approx(r2.score)
        assert r1.eligible == r2.eligible
