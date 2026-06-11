"""Mock hardware adapter for development and CI/CD.

Generates deterministic sine-wave EEG + PPG + IMU data at 4 Hz.
Ported from Rigpa-v2 mock_stream.py + Rigpa-v3 hardware/mock.py.
"""
from __future__ import annotations

import asyncio
import math
import time
from typing import AsyncGenerator

import numpy as np

from neurolink.hardware.base import EEGSample, HardwareAdapter

# Deterministic frequencies for each simulated band
_CHANNEL_NAMES = ["TP9", "AF7", "AF8", "TP10", "AUX"]
_EEG_FS: float = 256.0
_PPG_FS: float = 64.0
_IMU_FS: float = 52.0
_PUBLISH_HZ: float = 4.0
_SAMPLES_PER_FRAME = int(_EEG_FS / _PUBLISH_HZ)  # 64 samples per 4 Hz tick

# Sine-wave frequencies per channel (Hz)
_FREQS = {
    "TP9": 10.0,   # alpha
    "AF7": 9.5,    # alpha
    "AF8": 10.5,   # alpha
    "TP10": 10.0,  # alpha
    "AUX": 6.0,    # theta
}
_AMPLITUDES = {ch: 20.0 for ch in _CHANNEL_NAMES}  # 20 uV peak


class MockAdapter(HardwareAdapter):
    """Mock EEG adapter — no hardware required.

    Generates plausible sine-wave EEG with dominant alpha (10 Hz)
    and a theta component on AUX. PPG is a 60 bpm sine wave.
    IMU is near-static with small noise.
    """

    def __init__(self, publish_hz: float = _PUBLISH_HZ) -> None:
        self._publish_hz = publish_hz
        self._connected = False
        self._t0: float = 0.0
        self._frame: int = 0

    async def connect(self) -> None:
        """Activate mock adapter."""
        self._connected = True
        self._t0 = time.time()
        self._frame = 0

    async def disconnect(self) -> None:
        """Deactivate mock adapter."""
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def source_name(self) -> str:
        return "mock"

    async def stream(self) -> AsyncGenerator[EEGSample, None]:  # type: ignore[override]
        """Yield deterministic EEG frames at publish_hz."""
        interval = 1.0 / self._publish_hz
        while self._connected:
            sample = self._make_sample()
            yield sample
            self._frame += 1
            await asyncio.sleep(interval)

    def _make_sample(self) -> EEGSample:
        """Build a single deterministic EEG sample frame."""
        t_start = self._frame * _SAMPLES_PER_FRAME / _EEG_FS
        t = np.linspace(t_start, t_start + _SAMPLES_PER_FRAME / _EEG_FS, _SAMPLES_PER_FRAME)

        eeg: dict[str, list[float]] = {}
        for ch in _CHANNEL_NAMES:
            freq = _FREQS[ch]
            amp = _AMPLITUDES[ch]
            sig = amp * np.sin(2 * math.pi * freq * t)
            # Add small noise
            sig += np.random.default_rng(seed=self._frame + hash(ch) % 1000).normal(
                0, 1.0, len(t)
            )
            eeg[ch] = sig.tolist()

        # PPG: 60 bpm ~ 1 Hz sine
        ppg_t = np.linspace(t_start, t_start + 6 / _PPG_FS, 6)
        ppg = (1000.0 + 500.0 * np.sin(2 * math.pi * 1.0 * ppg_t)).tolist()

        # IMU: near-static head position (0g forward, 0g lateral, 1g up)
        accel = [0.0, 0.0, 1.0] * 3  # 3 samples x (x,y,z)
        gyro = [0.01, 0.01, 0.0] * 3

        return EEGSample(
            timestamp=time.time(),
            eeg=eeg,
            ppg=ppg,
            accel=accel,
            gyro=gyro,
            source="mock",
            address="mock",
            poor_contact=False,
            contact_quality=1.0,
        )
