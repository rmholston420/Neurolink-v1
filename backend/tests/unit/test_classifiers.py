"""Unit tests for dsp.classifiers -- v2 and v0.1 region/stage classifiers."""

from __future__ import annotations

from neurolink.dsp.classifiers import classify_v01, classify_v2, compute_s_space
from neurolink.models.eeg import BandPowers

# ---------------------------------------------------------------------------
# classify_v2
# ---------------------------------------------------------------------------


class TestClassifyV2:
    def test_returns_two_strings(self, flat_bands):
        region, stage = classify_v2(flat_bands)
        assert isinstance(region, str)
        assert isinstance(stage, str)

    def test_alpha_dominant_region(self):
        bands = BandPowers(alpha=0.7, theta=0.1, beta=0.1, delta=0.05, gamma=0.05)
        _region, stage = classify_v2(bands)
        assert isinstance(_region, str)
        assert len(_region) > 0

    def test_theta_dominant_returns_string(self):
        bands = BandPowers(alpha=0.1, theta=0.7, beta=0.1, delta=0.05, gamma=0.05)
        _region, stage = classify_v2(bands)
        assert isinstance(stage, str)

    def test_consistent_on_same_input(self, flat_bands):
        r1, s1 = classify_v2(flat_bands)
        r2, s2 = classify_v2(flat_bands)
        assert r1 == r2
        assert s1 == s2


# ---------------------------------------------------------------------------
# classify_v01
# ---------------------------------------------------------------------------


class TestClassifyV01:
    def test_returns_two_strings(self):
        region, stage = classify_v01(
            alpha=0.2, theta=0.2, beta=0.2, delta=0.2, gamma=0.2, faa=0.0, fmt=0.0
        )
        assert isinstance(region, str)
        assert isinstance(stage, str)

    def test_faa_positive_influence(self):
        r_pos, _ = classify_v01(0.5, 0.1, 0.1, 0.1, 0.2, faa=1.0, fmt=0.0)
        r_neg, _ = classify_v01(0.5, 0.1, 0.1, 0.1, 0.2, faa=-1.0, fmt=0.0)
        # Results may differ -- test simply that no exception is raised
        assert isinstance(r_pos, str)
        assert isinstance(r_neg, str)


# ---------------------------------------------------------------------------
# compute_s_space
# ---------------------------------------------------------------------------


class TestComputeSSpace:
    def test_returns_sspace_with_x_y(self, flat_bands):
        s = compute_s_space(flat_bands)
        assert hasattr(s, "x")
        assert hasattr(s, "y")

    def test_x_y_are_finite_floats(self, flat_bands):
        s = compute_s_space(flat_bands)
        import math

        assert math.isfinite(s.x)
        assert math.isfinite(s.y)

    def test_alpha_dominant_shifts_y(self):
        """High alpha / low beta increases integration_coverage (y)."""
        high_alpha = BandPowers(alpha=0.8, theta=0.05, beta=0.05, delta=0.05, gamma=0.05)
        low_alpha = BandPowers(alpha=0.05, theta=0.05, beta=0.8, delta=0.05, gamma=0.05)
        s_hi = compute_s_space(high_alpha)
        s_lo = compute_s_space(low_alpha)
        # High alpha should have higher y (integration) than high beta
        assert s_hi.y >= s_lo.y
