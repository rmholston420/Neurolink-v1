"""EEG band power computation.

Ported from Rigpa-v2 dsp/bandpower.py.
Uses scipy Welch PSD for band power estimation.

All functions are pure; no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal as sp_signal

# Standard EEG band definitions [lo, hi] Hz
_BANDS: dict[str, tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 50.0),
}

_EEG_FS: float = 256.0
_NPERSEG: int = 256

# Ring buffer sizes for make_buffers()
_N_EEG: int = int(_EEG_FS * 4.0)
_N_PPG: int = int(64.0 * 30.0)
_N_IMU: int = int(52.0 * 4.0 * 3)  # 3 axes * 4 seconds * 52 Hz


@dataclass
class BandPowers:
    """Normalised EEG band power fractions."""

    delta: float = 0.0
    theta: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0
    gamma: float = 0.0


def bandpower(sig: np.ndarray, lo: float, hi: float, fs: float = _EEG_FS) -> float:
    """Compute absolute band power in [lo, hi] Hz using Welch PSD.

    Args:
        sig: 1D signal array
        lo: Lower frequency bound (Hz)
        hi: Upper frequency bound (Hz)
        fs: Sampling rate (Hz)

    Returns:
        Absolute band power (sum of PSD in band), or 0.0 for empty/short input.
    """
    if sig is None or len(sig) < 2:
        return 0.0

    nperseg = min(_NPERSEG, len(sig))
    freqs, psd = sp_signal.welch(sig, fs=fs, nperseg=nperseg)
    freq_mask = (freqs >= lo) & (freqs <= hi)
    return float(np.sum(psd[freq_mask]))


def compute_band_powers(
    channels: list[list[float]] | np.ndarray,
    fs: float = _EEG_FS,
) -> BandPowers:
    """Compute normalised band power fractions from a list of channel arrays.

    Each channel is a 1D sequence of float samples. The powers are averaged
    across channels and then normalised so all bands sum to 1.  When the
    signal is all-zeros the result is a BandPowers with every field 0.0.

    Args:
        channels: Sequence of 1D channel arrays, shape (n_channels, n_samples),
                  or a 2D numpy array with the same layout.
        fs: Sampling rate (Hz).

    Returns:
        BandPowers dataclass with .delta, .theta, .alpha, .beta, .gamma.
    """
    zero = BandPowers()

    if channels is None:
        return zero

    arr = np.asarray(channels, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]  # treat as single channel

    n_channels, n_samples = arr.shape
    if n_samples < 2:
        return zero

    abs_powers: dict[str, float] = {}
    for band, (lo, hi) in _BANDS.items():
        ch_powers = [bandpower(arr[ch], lo, hi, fs) for ch in range(n_channels)]
        abs_powers[band] = float(np.mean(ch_powers))

    total = sum(abs_powers.values())
    if total <= 0:
        return zero

    normed = {band: abs_powers[band] / total for band in _BANDS}
    return BandPowers(
        delta=normed["delta"],
        theta=normed["theta"],
        alpha=normed["alpha"],
        beta=normed["beta"],
        gamma=normed["gamma"],
    )


def compute_band_powers_from_buffer(eeg: np.ndarray, fs: float = _EEG_FS) -> dict[str, float]:
    """Compute normalised band power fractions from a (5, N) EEG buffer.

    Averages power across all 5 channels, then normalises so all bands sum to 1.

    Args:
        eeg: EEG array of shape (5, N) or (N,) for single channel
        fs: Sampling rate (Hz)

    Returns:
        Dict mapping band name to normalised power fraction [0, 1].
        Returns all zeros if buffer is too short.
    """
    result: dict[str, float] = dict.fromkeys(_BANDS, 0.0)

    if eeg is None:
        return result

    if eeg.ndim == 1:
        eeg = eeg[np.newaxis, :]  # (1, N)

    n_channels, n_samples = eeg.shape
    if n_samples < 2:
        return result

    abs_powers: dict[str, float] = {}
    for band, (lo, hi) in _BANDS.items():
        ch_powers = [bandpower(eeg[ch], lo, hi, fs) for ch in range(n_channels)]
        abs_powers[band] = float(np.mean(ch_powers))

    total = sum(abs_powers.values())
    if total <= 0:
        return result

    return {band: abs_powers[band] / total for band in _BANDS}


def make_buffers() -> dict[str, np.ndarray]:
    """Return zero-filled ring buffer arrays sized for real-time use.

    Returns:
        Dict with keys 'eeg' (5, 1024), 'ppg' (1920,),
        'accel' (624,), 'gyro' (624,).
    """
    return {
        "eeg": np.zeros((5, _N_EEG), dtype=np.float32),
        "ppg": np.zeros(_N_PPG, dtype=np.float32),
        "accel": np.zeros(_N_IMU, dtype=np.float32),
        "gyro": np.zeros(_N_IMU, dtype=np.float32),
    }
