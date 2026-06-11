"""Band power computation utilities.

Ported from Rigpa-v2 muse_compute.py + Rigpa-v3 hardware/muse_s/compute.py.
All functions are pure (no I/O, no globals, no side effects).
"""
from __future__ import annotations

import numpy as np

# EEG band boundaries (Hz)
_BANDS: dict[str, tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

_DEFAULT_FS: float = 256.0
_EEG_CHANNELS: int = 5
_RING_SECONDS: float = 4.0
_PPG_FS: float = 64.0
_PPG_SECONDS: float = 30.0
_IMU_FS: float = 52.0
_IMU_SECONDS: float = 4.0


def bandpower(sig: np.ndarray, lo: float, hi: float, fs: float = _DEFAULT_FS) -> float:
    """Compute mean power in [lo, hi] Hz band using numpy rfft.

    Args:
        sig: 1-D signal array
        lo: lower frequency bound (Hz)
        hi: upper frequency bound (Hz)
        fs: sampling frequency (Hz)

    Returns:
        Mean power in band as float (0.0 for empty/short signals).
    """
    n = len(sig)
    if n < 2:
        return 0.0
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    fft_vals = np.fft.rfft(sig)
    power = (np.abs(fft_vals) ** 2) / n
    mask = (freqs >= lo) & (freqs < hi)
    if not np.any(mask):
        return 0.0
    return float(np.mean(power[mask]))


def compute_band_powers_from_buffer(
    eeg: np.ndarray,
    fs: float = _DEFAULT_FS,
) -> dict[str, float]:
    """Compute normalised band powers from a multi-channel EEG buffer.

    Args:
        eeg: 2-D array of shape (n_channels, n_samples)
        fs: sampling frequency (Hz)

    Returns:
        Dict mapping band name -> normalised fraction [0, 1].
        All zeros if buffer is too short.
    """
    if eeg.shape[1] < 2:
        return {k: 0.0 for k in _BANDS}

    # Average power across channels
    band_powers: dict[str, float] = {}
    for band, (lo, hi) in _BANDS.items():
        ch_powers = [bandpower(eeg[ch], lo, hi, fs) for ch in range(eeg.shape[0])]
        band_powers[band] = float(np.mean(ch_powers))

    total = sum(band_powers.values())
    if total <= 0:
        return {k: 0.0 for k in _BANDS}

    # Normalise to fractions that sum to 1.0
    return {k: v / total for k, v in band_powers.items()}


def make_buffers() -> dict[str, np.ndarray]:
    """Create empty ring buffers for EEG, PPG, and IMU.

    Returns:
        Dict with zero-filled numpy arrays for each modality:
        - eeg: (5, 1024)  — 5 channels x 4s @ 256 Hz
        - ppg: (1920,)    — 30s @ 64 Hz
        - accel: (624,)   — 4s @ 52 Hz x 3 axes
        - gyro: (624,)    — 4s @ 52 Hz x 3 axes
    """
    eeg_n = int(_DEFAULT_FS * _RING_SECONDS)
    ppg_n = int(_PPG_FS * _PPG_SECONDS)
    imu_n = int(_IMU_FS * _IMU_SECONDS) * 3  # 3 axes flat
    return {
        "eeg": np.zeros((_EEG_CHANNELS, eeg_n), dtype=np.float32),
        "ppg": np.zeros(ppg_n, dtype=np.float32),
        "accel": np.zeros(imu_n, dtype=np.float32),
        "gyro": np.zeros(imu_n, dtype=np.float32),
    }
