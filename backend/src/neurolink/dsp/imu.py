"""IMU processing: head orientation (pitch/roll) and motion RMS gating."""
from __future__ import annotations

import math

import numpy as np

from neurolink.models.eeg import IMUPayload


def head_orientation(
    accel: np.ndarray,
    gyro: np.ndarray | None = None,
) -> IMUPayload:
    """Compute gravity-referenced pitch/roll and motion RMS from accelerometer.

    Args:
        accel: shape (3, N) or (N, 3) accelerometer data in g units
               Axes: x=forward, y=lateral, z=vertical
        gyro: optional gyroscope data (same shape) — used for motion_rms

    Returns:
        IMUPayload with pitch_deg (±90), roll_deg (±90), motion_rms
    """
    try:
        # Normalise to (3, N)
        if accel.ndim == 1:
            accel = accel.reshape(3, 1)
        elif accel.shape[0] != 3 and accel.shape[1] == 3:
            accel = accel.T

        if accel.shape[0] != 3 or accel.shape[1] == 0:
            return IMUPayload()

        # Mean gravity vector
        gx = float(np.mean(accel[0]))
        gy = float(np.mean(accel[1]))
        gz = float(np.mean(accel[2]))

        # Gravity-referenced pitch and roll (deg)
        pitch_rad = math.atan2(gx, math.sqrt(gy ** 2 + gz ** 2))
        roll_rad = math.atan2(gy, math.sqrt(gx ** 2 + gz ** 2))
        pitch_deg = math.degrees(pitch_rad)
        roll_deg = math.degrees(roll_rad)

        # Clamp to ±90
        pitch_deg = max(-90.0, min(90.0, pitch_deg))
        roll_deg = max(-90.0, min(90.0, roll_deg))

        # Motion RMS from gyroscope (or accel variance as fallback)
        if gyro is not None and gyro.size > 0:
            if gyro.ndim == 1:
                gyro = gyro.reshape(3, 1)
            elif gyro.shape[0] != 3 and gyro.shape[1] == 3:
                gyro = gyro.T
            motion_rms = float(np.sqrt(np.mean(gyro ** 2)))
        else:
            # Use accel variance as motion proxy
            motion_rms = float(np.sqrt(np.mean(np.var(accel, axis=1))))

        return IMUPayload(
            pitch_deg=pitch_deg,
            roll_deg=roll_deg,
            motion_rms=motion_rms,
        )
    except Exception:
        return IMUPayload()
