"""Stage 0 -- IMU Motion Gate.

Records accelerometer/gyroscope as a parallel channel and flags EEG segments
where motion magnitude exceeds a threshold before any neural analysis.

Algorithm
---------
For each EEGSample that carries accel_buffer / gyro_buffer:

  1. Compute per-sample RMS across (ax, ay, az) after subtracting gravity
     using the running mean of the z-axis as a gravity estimate.
  2. Append to a sliding window of length `window_samples`.
  3. Compute window RMS magnitude.
  4. If RMS > threshold_g, set sample.extra["motion_flagged"] = True
     and sample.extra["motion_rms"] = <value>.

The pipeline downstream can then discard or downweight flagged segments.
"""

from __future__ import annotations

import time
from collections import deque

import numpy as np

from neurolink.hardware.base import EEGSample

# Default threshold: 0.15 g  (standing-still baseline is ~0.01-0.05 g)
_DEFAULT_THRESHOLD_G: float = 0.15
# Window length in IMU samples (Muse IMU is ~52 Hz; 26 samples ~= 500 ms)
_DEFAULT_WINDOW: int = 26


class IMUGate:
    """Sliding-window RMS motion detector.

    Mutates EEGSample.extra in-place:
      extra["motion_flagged"] = bool
      extra["motion_rms"]     = float   (g-units)
      extra["motion_ts"]      = float   (unix timestamp)
    """

    def __init__(
        self,
        threshold_g: float = _DEFAULT_THRESHOLD_G,
        window_samples: int = _DEFAULT_WINDOW,
    ) -> None:
        self._threshold = threshold_g
        self._window: deque[float] = deque(maxlen=window_samples)
        self._last_rms: float = 0.0
        self._flagged: bool = False
        self._last_ts: float = 0.0

    def flag_segment(self, sample: EEGSample) -> EEGSample:
        """Evaluate motion in *sample* and annotate extra{} in-place.

        Safe to call even when accel_buffer is None (no-op in that case).
        """
        if sample.accel_buffer is None or len(sample.accel_buffer) < 3:
            sample.extra["motion_flagged"] = False
            sample.extra["motion_rms"] = 0.0
            return sample

        rms = self._compute_rms(sample.accel_buffer)
        self._window.append(rms)
        window_rms = float(np.sqrt(np.mean(np.array(self._window) ** 2)))

        self._last_rms = window_rms
        self._flagged = window_rms > self._threshold
        self._last_ts = time.time()

        sample.extra["motion_flagged"] = self._flagged
        sample.extra["motion_rms"] = round(window_rms, 4)
        sample.extra["motion_ts"] = self._last_ts
        return sample

    @staticmethod
    def _compute_rms(accel_buffer: list[list[float]]) -> float:
        """Compute scalar RMS acceleration magnitude for one buffer snapshot.

        accel_buffer shape: (3, N)  -- rows are [ax, ay, az].
        Gravity is removed by subtracting the mean of the z-axis.
        """
        ax = np.array(accel_buffer[0], dtype=np.float32)
        ay = np.array(accel_buffer[1], dtype=np.float32)
        az = np.array(accel_buffer[2], dtype=np.float32)

        # Crude gravity removal: subtract running mean of z-axis
        az_demeaned = az - float(np.mean(az))

        # Vector magnitude sample-wise, then RMS
        mag = np.sqrt(ax**2 + ay**2 + az_demeaned**2)
        return float(np.sqrt(np.mean(mag**2)))

    @property
    def is_flagged(self) -> bool:
        """True if the most recent window was above the motion threshold."""
        return self._flagged

    @property
    def last_rms(self) -> float:
        return self._last_rms

    @property
    def threshold_g(self) -> float:
        return self._threshold

    def status_dict(self) -> dict:
        return {
            "flagged": self._flagged,
            "motion_rms_g": round(self._last_rms, 4),
            "threshold_g": self._threshold,
            "last_ts": self._last_ts,
        }
