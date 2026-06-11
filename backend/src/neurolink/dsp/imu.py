"""IMU head orientation and motion detection.

Ported from Rigpa-v3 dsp/imu.py.
Computes pitch, roll, and motion RMS from accelerometer + optional gyro.
"""
from __future__ import annotations

import numpy as np

from neurolink.models.eeg import IMUPayload

_GRAVITY: float = 9.81


def head_orientation(
    accel: np.ndarray,
    gyro: np.ndarray | None = None,
) -> IMUPayload:
    """Compute head pitch, roll, and motion RMS.

    Args:
        accel: Accelerometer array of shape (3, N) [x, y, z] in g.
        gyro: Gyroscope array of shape (3, N) (optional, not used currently).

    Returns:
        IMUPayload with pitch_deg, roll_deg, motion_rms.
    """
    empty = IMUPayload(pitch_deg=0.0, roll_deg=0.0, motion_rms=0.0)

    if accel is None or accel.shape[1] == 0:
        return empty

    # Mean accel values
    ax = float(np.mean(accel[0]))
    ay = float(np.mean(accel[1]))
    az = float(np.mean(accel[2]))

    # Pitch and roll (degrees)
    pitch = float(np.degrees(np.arctan2(-ax, np.sqrt(ay ** 2 + az ** 2))))
    roll = float(np.degrees(np.arctan2(ay, az)))

    # Clamp
    pitch = max(-90.0, min(90.0, pitch))
    roll = max(-90.0, min(90.0, roll))

    # Motion RMS: deviation from 1g (gravity)
    mag = np.sqrt(accel[0] ** 2 + accel[1] ** 2 + accel[2] ** 2)
    deviation = mag - 1.0  # subtract gravity
    motion_rms = float(np.sqrt(np.mean(deviation ** 2)))

    return IMUPayload(
        pitch_deg=pitch,
        roll_deg=roll,
        motion_rms=motion_rms,
    )
