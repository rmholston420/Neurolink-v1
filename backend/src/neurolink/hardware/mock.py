"""Mock EEG adapter — deterministic sine-wave EEG, PPG, and IMU.

Ported from Rigpa-v2 mock_stream.py + Rigpa-v3 hardware/mock.py.
Activated via NEUROLINK_ADAPTER_TYPE=mock.
Requires no hardware or BLE drivers.
"""

from __future__ import annotations

import asyncio
import math
import time

import numpy as np

from neurolink.hardware.base import EEGSample, HardwareAdapter

# Mock signal parameters
_EEG_FS: float = 256.0
_PPG_FS: float = 64.0
_IMU_FS: float = 52.0
_RING_SECS_EEG: float = 4.0
_RING_SECS_PPG: float = 30.0
_RING_SECS_IMU: float = 4.0

_N_EEG: int = int(_EEG_FS * _RING_SECS_EEG)  # 1024
_N_PPG: int = int(_PPG_FS * _RING_SECS_PPG)  # 1920
_N_IMU: int = int(_IMU_FS * _RING_SECS_IMU)  # 208

# Simulate a resting alpha-dominant EEG
_ALPHA_FREQ: float = 10.0  # Hz
_THETA_FREQ: float = 6.0  # Hz
_PPG_HR_FREQ: float = 1.1  # Hz (~66 bpm)


class MockAdapter(HardwareAdapter):
    """Generates deterministic sine-wave EEG, PPG, and IMU data.

    The signal is alpha-dominant with a theta component to produce
    a Rubedo/deep-meditation classification in both classifiers.
    """

    def __init__(self) -> None:
        self._connected: bool = False
        self._start_time: float = 0.0
        self._frame: int = 0

    async def connect(self) -> None:
        """Mark adapter as connected; reset clock."""
        self._connected = True
        self._start_time = time.time()
        self._frame = 0

    async def disconnect(self) -> None:
        """Mark adapter as disconnected."""
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def source_name(self) -> str:
        return "mock"

    async def read_sample(self) -> EEGSample | None:
        """Generate and return one mock EEG sample.

        Simulates a ~4 Hz read cycle with a brief sleep.
        Returns None if not connected.
        """
        if not self._connected:
            return None

        await asyncio.sleep(0.25)  # 4 Hz

        t = time.time() - self._start_time
        self._frame += 1

        # Build fake EEG ring buffer — strong alpha + moderate theta
        t_vec = np.linspace(t, t + _RING_SECS_EEG, _N_EEG)
        alpha = 0.45 * np.sin(2 * math.pi * _ALPHA_FREQ * t_vec)
        theta = 0.25 * np.sin(2 * math.pi * _THETA_FREQ * t_vec)
        noise = 0.05 * np.random.randn(_N_EEG)
        signal = alpha + theta + noise

        # 5 channels (slightly phase-shifted per channel)
        eeg_buf = [(signal + 0.02 * np.random.randn(_N_EEG)).tolist() for _ in range(5)]

        # PPG — simulated heartbeat
        t_ppg = np.linspace(t, t + _RING_SECS_PPG, _N_PPG)
        ppg = 0.8 * np.sin(2 * math.pi * _PPG_HR_FREQ * t_ppg) + 0.1 * np.random.randn(_N_PPG)
        ppg_buf = ppg.tolist()

        # IMU — near-static (head resting)
        accel_buf = [
            (0.01 * np.random.randn(_N_IMU)).tolist(),  # x
            (0.01 * np.random.randn(_N_IMU)).tolist(),  # y
            (1.0 + 0.005 * np.random.randn(_N_IMU)).tolist(),  # z (gravity)
        ]
        gyro_buf = [
            (0.5 * np.random.randn(_N_IMU)).tolist(),
            (0.5 * np.random.randn(_N_IMU)).tolist(),
            (0.5 * np.random.randn(_N_IMU)).tolist(),
        ]

        # Per-channel single-sample value (latest EEG point)
        channels = [float(ch[-1]) for ch in eeg_buf]

        return EEGSample(
            channels=channels,
            timestamp=time.time(),
            source="mock",
            address="mock",
            poor_contact=False,
            eeg_buffer=eeg_buf,
            ppg_buffer=ppg_buf,
            accel_buffer=accel_buf,
            gyro_buffer=gyro_buf,
        )
