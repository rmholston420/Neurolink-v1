"""Unit tests for EEG classifiers (v2 + v0.1) and S-space computation."""

from __future__ import annotations

import pytest

from neurolink.dsp.classifiers import classify_v01, classify_v2, compute_s_space
from neurolink.models.eeg import BandPowers


class TestClassifyV2:
    def test_returns_tuple_of_two_strings(self, flat_bands):
        region, stage = classify_v2(flat_bands)
        assert isinstance(region, str)
        assert isinstance(stage, str)

    def test_alpha_dominant_returns_valid_region(self, alpha_dominant_bands):
        region, stage = classify_v2(alpha_dominant_bands)
        assert len(region) > 0
        assert len(stage) > 0

    def test_all_zero_bands_does_not_raise(self):
        bands = BandPowers(alpha=0, theta=0, beta=0, delta=0, gamma=0)
        region, stage = classify_v2(bands)
        assert isinstance(region, str)


class TestComputeSSpace:
    def test_returns_s_space_coords(self, flat_bands):
        coords = compute_s_space(flat_bands)
        assert hasattr(coords, 'x')
        assert hasattr(coords, 'y')
        assert hasattr(coords, 'z')

    def test_coords_are_floats(self, flat_bands):
        coords = compute_s_space(flat_bands)
        assert isinstance(coords.x, float)
        assert isinstance(coords.y, float)
        assert isinstance(coords.z, float)

    def test_high_alpha_affects_engagement(self):
        low_alpha = BandPowers(alpha=0.05, theta=0.4, beta=0.3, delta=0.2, gamma=0.05)
        high_alpha = BandPowers(alpha=0.7, theta=0.1, beta=0.1, delta=0.05, gamma=0.05)
        coords_low = compute_s_space(low_alpha)
        coords_high = compute_s_space(high_alpha)
        # S-space should differ between states
        assert coords_low.x != coords_high.x or coords_low.y != coords_high.y


class TestClassifyV01:
    def test_returns_tuple_of_two_strings(self, flat_bands):
        region, stage = classify_v01(
            alpha=flat_bands.alpha,
            theta=flat_bands.theta,
            beta=flat_bands.beta,
            delta=flat_bands.delta,
            gamma=flat_bands.gamma,
            faa=None,
            fmt=None,
        )
        assert isinstance(region, str)
        assert isinstance(stage, str)

    def test_does_not_raise_with_none_faa_fmt(self, alpha_dominant_bands):
        region, stage = classify_v01(
            alpha=alpha_dominant_bands.alpha,
            theta=alpha_dominant_bands.theta,
            beta=alpha_dominant_bands.beta,
            delta=alpha_dominant_bands.delta,
            gamma=alpha_dominant_bands.gamma,
            faa=None,
            fmt=None,
        )
        assert isinstance(region, str)
