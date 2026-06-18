"""Unit tests for dsp.spherical_spline.SphericalSplineInterpolator."""
from __future__ import annotations

import numpy as np
import pytest

from neurolink.dsp.spherical_spline import SphericalSplineInterpolator


# Minimal electrode layout on a unit sphere: 4 good + 1 bad
GOOD_COORDS = np.array([
    [0.0,  0.0,  1.0],
    [1.0,  0.0,  0.0],
    [0.0,  1.0,  0.0],
    [-1.0, 0.0,  0.0],
], dtype=float)

BAD_COORD = np.array([[0.0, -1.0, 0.0]], dtype=float)
ALL_COORDS = np.vstack([GOOD_COORDS, BAD_COORD])


@pytest.fixture()
def interpolator() -> SphericalSplineInterpolator:
    return SphericalSplineInterpolator(electrode_coords=ALL_COORDS)


class TestConstruction:
    def test_instantiation(self):
        interp = SphericalSplineInterpolator(electrode_coords=ALL_COORDS)
        assert interp is not None

    def test_electrode_count(self, interpolator):
        assert interpolator.n_electrodes == len(ALL_COORDS)


class TestInterpolation:
    def test_output_shape(self, interpolator):
        data = np.random.default_rng(0).normal(0, 5e-6, size=(len(ALL_COORDS),))
        bad_mask = np.zeros(len(ALL_COORDS), dtype=bool)
        bad_mask[-1] = True
        out = interpolator.interpolate(data, bad_mask)
        assert out.shape == data.shape

    def test_good_channels_unchanged(self, interpolator):
        data = np.random.default_rng(1).normal(0, 5e-6, size=(len(ALL_COORDS),))
        bad_mask = np.zeros(len(ALL_COORDS), dtype=bool)
        bad_mask[-1] = True
        out = interpolator.interpolate(data, bad_mask)
        np.testing.assert_array_almost_equal(out[:-1], data[:-1])

    def test_bad_channel_is_finite(self, interpolator):
        data = np.random.default_rng(2).normal(0, 5e-6, size=(len(ALL_COORDS),))
        bad_mask = np.zeros(len(ALL_COORDS), dtype=bool)
        bad_mask[-1] = True
        out = interpolator.interpolate(data, bad_mask)
        assert np.isfinite(out[-1])

    def test_no_bad_channels_returns_input(self, interpolator):
        data = np.random.default_rng(3).normal(0, 5e-6, size=(len(ALL_COORDS),))
        bad_mask = np.zeros(len(ALL_COORDS), dtype=bool)
        out = interpolator.interpolate(data, bad_mask)
        np.testing.assert_array_almost_equal(out, data)


class TestAllBadRaisesOrReturns:
    def test_all_bad_does_not_crash(self, interpolator):
        data = np.zeros(len(ALL_COORDS))
        bad_mask = np.ones(len(ALL_COORDS), dtype=bool)
        # Must either return zeros/nans gracefully or raise ValueError — not crash hard
        try:
            out = interpolator.interpolate(data, bad_mask)
            assert out.shape == data.shape
        except (ValueError, RuntimeError):
            pass
