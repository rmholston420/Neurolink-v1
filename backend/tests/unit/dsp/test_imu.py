"""Unit tests for dsp/imu.py."""

from __future__ import annotations

import math

import numpy as np
import pytest

from neurolink.dsp.imu import compute_motion_rms, head_orientation
from neurolink.models.eeg import IMUPayload


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FS = 256.0
N = 64


def _accel(
    ax: float = 0.0,
    ay: float = 0.0,
    az: float = 1.0,
    n: int = N,
) -> np.ndarray:
    """Constant 3-axis accelerometer array (3, N)."""
    return np.array(
        [[ax] * n, [ay] * n, [az] * n], dtype=np.float32
    )


# ---------------------------------------------------------------------------
# head_orientation()
# ---------------------------------------------------------------------------

class TestHeadOrientation:
    # --- Guard conditions ---
    def test_none_accel_returns_zero_payload(self):
        result = head_orientation(None)
        assert result == IMUPayload(pitch_deg=0.0, roll_deg=0.0, motion_rms=0.0)

    def test_empty_accel_returns_zero_payload(self):
        result = head_orientation(np.zeros((3, 0), dtype=np.float32))
        assert result == IMUPayload(pitch_deg=0.0, roll_deg=0.0, motion_rms=0.0)

    # --- Return type ---
    def test_returns_imu_payload_instance(self):
        result = head_orientation(_accel())
        assert isinstance(result, IMUPayload)

    # --- Pitch and roll range ---
    def test_pitch_clamped_to_90(self):
        """Pure -x direction (nose down) → pitch clamped at -90 deg."""
        arr = _accel(ax=-10.0, ay=0.0, az=0.0)
        result = head_orientation(arr)
        assert -90.0 <= result.pitch_deg <= 90.0

    def test_roll_clamped_to_90(self):
        arr = _accel(ax=0.0, ay=10.0, az=0.0)
        result = head_orientation(arr)
        assert -90.0 <= result.roll_deg <= 90.0

    def test_upright_position_pitch_near_zero(self):
        """ax=0, ay=0, az=1g → pitch ≈ 0, roll ≈ 0."""
        arr = _accel(ax=0.0, ay=0.0, az=1.0)
        result = head_orientation(arr)
        assert result.pitch_deg == pytest.approx(0.0, abs=0.1)
        assert result.roll_deg == pytest.approx(0.0, abs=0.1)

    def test_pure_x_tilt_nonzero_pitch(self):
        """Tilting in x changes pitch."""
        arr_tilted = _accel(ax=0.5, ay=0.0, az=0.866)
        result = head_orientation(arr_tilted)
        assert result.pitch_deg != 0.0

    def test_pure_y_tilt_nonzero_roll(self):
        arr_tilted = _accel(ax=0.0, ay=0.5, az=0.866)
        result = head_orientation(arr_tilted)
        assert result.roll_deg != 0.0

    # --- motion_rms ---
    def test_motion_rms_nonnegative(self):
        result = head_orientation(_accel())
        assert result.motion_rms >= 0.0

    def test_motion_rms_near_zero_for_1g_stationary(self):
        """ax=0, ay=0, az=1.0 → mag=1g every sample → deviation=0 → rms ≈ 0."""
        arr = _accel(ax=0.0, ay=0.0, az=1.0)
        result = head_orientation(arr)
        assert result.motion_rms == pytest.approx(0.0, abs=1e-6)

    def test_motion_rms_elevated_for_shaking(self):
        """Random high-amplitude accelerations → large deviation from 1g."""
        rng = np.random.default_rng(42)
        arr = rng.standard_normal((3, 64)).astype(np.float32) * 3.0
        result = head_orientation(arr)
        assert result.motion_rms > 0.1

    # --- gyro ignored ---
    def test_gyro_argument_accepted_without_error(self):
        accel = _accel()
        gyro = np.zeros((3, N), dtype=np.float32)
        result = head_orientation(accel, gyro=gyro)
        assert isinstance(result, IMUPayload)

    def test_gyro_does_not_change_pitch_roll(self):
        accel = _accel(ax=0.1, ay=0.2, az=0.9)
        r_no_gyro = head_orientation(accel)
        r_with_gyro = head_orientation(accel, gyro=np.ones((3, N), dtype=np.float32))
        assert r_no_gyro.pitch_deg == pytest.approx(r_with_gyro.pitch_deg)
        assert r_no_gyro.roll_deg == pytest.approx(r_with_gyro.roll_deg)

    # --- single-sample ---
    def test_single_sample_does_not_raise(self):
        arr = np.array([[0.0], [0.0], [1.0]], dtype=np.float32)
        result = head_orientation(arr)
        assert isinstance(result, IMUPayload)


# ---------------------------------------------------------------------------
# compute_motion_rms()
# ---------------------------------------------------------------------------

class TestComputeMotionRms:
    def test_returns_float(self):
        assert isinstance(compute_motion_rms(0.0, 0.0, 1.0), float)

    def test_zero_vector_returns_zero(self):
        assert compute_motion_rms(0.0, 0.0, 0.0) == 0.0

    def test_unit_z_returns_one(self):
        assert compute_motion_rms(0.0, 0.0, 1.0) == pytest.approx(1.0)

    def test_pythagoras_3_4_5(self):
        """sqrt(3^2 + 4^2 + 0^2) = 5.0"""
        assert compute_motion_rms(3.0, 4.0, 0.0) == pytest.approx(5.0)

    def test_known_3d_value(self):
        """sqrt(1^2 + 2^2 + 2^2) = 3.0"""
        assert compute_motion_rms(1.0, 2.0, 2.0) == pytest.approx(3.0)

    def test_negative_components(self):
        """Magnitude is always non-negative regardless of sign."""
        assert compute_motion_rms(-3.0, -4.0, 0.0) == pytest.approx(5.0)

    def test_nonnegative(self):
        for ax, ay, az in [(0, 0, 0), (1, -1, 1), (-5, 0, 12)]:
            assert compute_motion_rms(ax, ay, az) >= 0.0

    def test_large_values(self):
        result = compute_motion_rms(100.0, 0.0, 0.0)
        assert result == pytest.approx(100.0)
