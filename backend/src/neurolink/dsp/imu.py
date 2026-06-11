"""IMU processing: head orientation (pitch/roll) and motion RMS gating.

Ported from Rigpa-v3 dsp/imu.py.
All functions are pure.
"""
from __future__ import annotations

import math

import numpy as np

from neurolink.models.eeg import IMUPayload

_G: float = 9.80665  # m/s²


def head_orientation(
    accel: np.ndarray,
    gyro: np.ndarray | None = None,
) -> IMUPayload:
    """Compute pitch, roll, and motion RMS from accelerometer data.

    Args:
        accel: 2-D array of shape (3, n_samples) [x, y, z rows] in g
        gyro: optional 2-D array of shape (3, n_samples) in deg/s

    Returns:
        IMUPayload with pitch_deg, roll_deg, and motion_rms.
        Defaults to zeros if accel is empty.
    """
    if accel.shape[1] == 0:
        return IMUPayload()

    # Mean accel over window (gravity reference)
    ax = float(np.mean(accel[0]))
    ay = float(np.mean(accel[1]))
    az = float(np.mean(accel[2]))

    # Pitch: rotation around y-axis
    # pitch = atan2(ax, sqrt(ay^2 + az^2))
    pitch_rad = math.atan2(ax, math.sqrt(ay ** 2 + az ** 2 + 1e-9))
    pitch_deg = math.degrees(pitch_rad)

    # Roll: rotation around x-axis
    # roll = atan2(-ay, az)
    roll_rad = math.atan2(-ay, az + 1e-9)
    roll_deg = math.degrees(roll_rad)

    # Clamp to +-90 degrees
    pitch_deg = max(-90.0, min(90.0, pitch_deg))
    roll_deg = max(-90.0, min(90.0, roll_deg))

    # Motion RMS: RMS of accel vector magnitude deviation from 1g
    accel_mag = np.sqrt(accel[0] ** 2 + accel[1] ** 2 + accel[2] ** 2)
    motion_rms = float(np.sqrt(np.mean((accel_mag - 1.0) ** 2)))

    # Add gyro RMS if available
    if gyro is not None and gyro.shape[1] > 0:
        gyro_rms = float(np.sqrt(np.mean(gyro ** 2)))
        # Combine accel and gyro motion signals
        motion_rms = float(np.sqrt((motion_rms ** 2 + (gyro_rms / 100.0) ** 2) / 2))

    return IMUPayload(pitch_deg=pitch_deg, roll_deg=roll_deg, motion_rms=motion_rms)
