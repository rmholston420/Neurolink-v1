"""Unit tests for dsp.classifiers — classify_v2 and classify_v01."""

from __future__ import annotations

import pytest

from neurolink.dsp.classifiers import classify_v01, classify_v2
from neurolink.models.eeg import BandPowers


class TestClassifyV2:
    def test_alpha_dominant_region(self):
        bands = BandPowers(alpha=0.7, theta=0.1, beta=0.1, delta=0.05, gamma=0.05)
        _region, _stage = classify_v2(bands)
        assert isinstance(_region, str)
        assert len(_region) > 0

    def test_theta_dominant_returns_string(self):
        bands = BandPowers(alpha=0.1, theta=0.7, beta=0.1, delta=0.05, gamma=0.05)
        _region, stage = classify_v2(bands)
        assert isinstance(stage, str)

    def test_beta_dominant_returns_string(self):
        bands = BandPowers(alpha=0.1, theta=0.1, beta=0.7, delta=0.05, gamma=0.05)
        _region, stage = classify_v2(bands)
        assert isinstance(stage, str)
        assert len(stage) > 0

    def test_balanced_bands(self):
        bands = BandPowers(alpha=0.2, theta=0.2, beta=0.2, delta=0.2, gamma=0.2)
        region, stage = classify_v2(bands)
        assert isinstance(region, str)
        assert isinstance(stage, str)

    def test_returns_tuple_of_two_strings(self):
        bands = BandPowers(alpha=0.5, theta=0.2, beta=0.15, delta=0.1, gamma=0.05)
        result = classify_v2(bands)
        assert len(result) == 2
        assert all(isinstance(r, str) for r in result)


class TestClassifyV01:
    def test_returns_tuple_of_two_strings(self):
        region, stage = classify_v01(
            alpha=0.3, theta=0.2, beta=0.15, delta=0.2, gamma=0.15
        )
        assert isinstance(region, str)
        assert isinstance(stage, str)

    def test_high_alpha_region(self):
        region, stage = classify_v01(
            alpha=0.6, theta=0.1, beta=0.1, delta=0.1, gamma=0.1
        )
        assert isinstance(region, str)
        assert isinstance(stage, str)

    def test_high_beta_region(self):
        region, stage = classify_v01(
            alpha=0.1, theta=0.1, beta=0.6, delta=0.1, gamma=0.1
        )
        assert isinstance(region, str)
        assert isinstance(stage, str)
