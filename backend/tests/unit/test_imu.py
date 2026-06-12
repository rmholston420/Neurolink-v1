"""Unit tests for dsp.imu — motion/orientation processing."""

from __future__ import annotations

import pytest

from neurolink.dsp.imu import compute_motion_rms


class TestComputeMotionRMS:
    def test_zero_acceleration_gives_zero(self):
        result = compute_motion_rms(ax=0.0, ay=0.0, az=0.0)
        assert result == pytest.approx(0.0, abs=1e-9)

    def test_unit_vector_gives_one(self):
        # sqrt(1^2 + 0^2 + 0^2) = 1
        result = compute_motion_rms(ax=1.0, ay=0.0, az=0.0)
        assert result == pytest.approx(1.0)

    def test_positive_values(self):
        result = compute_motion_rms(ax=0.5, ay=0.5, az=0.5)
        assert result > 0.0

    def test_symmetric(self):
        a = compute_motion_rms(ax=1.0, ay=2.0, az=3.0)
        b = compute_motion_rms(ax=3.0, ay=2.0, az=1.0)
        assert a == pytest.approx(b)
