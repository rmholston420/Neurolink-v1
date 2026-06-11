"""Unit tests for dsp/imu.py."""

from __future__ import annotations

import numpy as np

from neurolink.dsp.imu import head_orientation


def test_head_orientation_pitch_roll_bounded():
    """Pitch and roll should be bounded to +-90 degrees."""
    # Near-static head: accel points down (z=1g)
    n = 100
    accel = np.zeros((3, n))
    accel[2] = 1.0  # z = 1g
    result = head_orientation(accel)
    assert -90.0 <= result.pitch_deg <= 90.0
    assert -90.0 <= result.roll_deg <= 90.0


def test_head_orientation_motion_rms_low_when_static():
    """motion_rms should be near 0 for a static head."""
    n = 100
    accel = np.zeros((3, n))
    accel[2] = 1.0  # perfectly static: 1g on z
    result = head_orientation(accel)
    assert result.motion_rms < 0.1


def test_head_orientation_empty_returns_zeros():
    """Empty accel buffer should return default zeros."""
    accel = np.zeros((3, 0))
    result = head_orientation(accel)
    assert result.pitch_deg == 0.0
    assert result.roll_deg == 0.0
    assert result.motion_rms == 0.0


def test_head_orientation_with_gyro():
    """Should not raise when gyro is provided."""
    n = 100
    accel = np.zeros((3, n))
    accel[2] = 1.0
    gyro = np.zeros((3, n))
    result = head_orientation(accel, gyro)
    assert isinstance(result.motion_rms, float)
