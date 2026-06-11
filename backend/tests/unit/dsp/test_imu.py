"""Unit tests for dsp/imu.py."""
from __future__ import annotations

import numpy as np
import pytest

from neurolink.dsp.imu import head_orientation
from neurolink.models.eeg import IMUPayload


def test_head_orientation_pitch_roll_bounded():
    """Pitch and roll should be within +-90 degrees."""
    accel = np.array([[0.0] * 10, [0.0] * 10, [1.0] * 10])  # 3xN
    result = head_orientation(accel)
    assert isinstance(result, IMUPayload)
    assert -90.0 <= result.pitch_deg <= 90.0
    assert -90.0 <= result.roll_deg <= 90.0


def test_head_orientation_upright_head():
    """With pure vertical accel, pitch and roll should be ~0."""
    accel = np.array([[0.0] * 20, [0.0] * 20, [1.0] * 20])
    result = head_orientation(accel)
    assert abs(result.pitch_deg) < 5.0
    assert abs(result.roll_deg) < 5.0


def test_head_orientation_motion_rms_computed():
    """motion_rms should be a non-negative float."""
    accel = np.random.default_rng(42).normal(0, 0.1, (3, 50)) + np.array([[0], [0], [1.0]])
    gyro = np.random.default_rng(99).normal(0, 0.05, (3, 50))
    result = head_orientation(accel, gyro)
    assert result.motion_rms >= 0.0


def test_head_orientation_empty_returns_default():
    accel = np.zeros((3, 0))
    result = head_orientation(accel)
    assert isinstance(result, IMUPayload)
