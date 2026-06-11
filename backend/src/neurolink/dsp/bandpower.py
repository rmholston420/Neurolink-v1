"""Band power computation and ring buffer management."""
from __future__ import annotations

import numpy as np

# Sampling rates
EEG_FS: float = 256.0
PPG_FS: float = 64.0
IMU_FS: float = 52.0

# Buffer durations (seconds)
EEG_BUF_SEC: float = 4.0
PPG_BUF_SEC: float = 30.0
IMU_BUF_SEC: float = 4.0

# Number of EEG channels: TP9, AF7, AF8, TP10, AUX
EEG_CHANNELS: int = 5


def bandpower(sig: np.ndarray, lo: float, hi: float, fs: float) -> float:
    """Compute mean power in [lo, hi] Hz band via rfft.

    Returns 0.0 for signals shorter than 2 samples.
    """
    n = len(sig)
    if n < 2:
        return 0.0
    fft_vals = np.abs(np.fft.rfft(sig)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mask = (freqs >= lo) & (freqs <= hi)
    if not mask.any():
        return 0.0
    return float(np.mean(fft_vals[mask]))


def make_buffers() -> dict[str, np.ndarray]:
    """Create zero-filled ring buffers for EEG, PPG, and IMU streams.

    Returns a dict with keys: eeg (5ch x N), ppg (N,), accel (3 x M), gyro (3 x M).
    """
    eeg_samples = int(EEG_BUF_SEC * EEG_FS)
    ppg_samples = int(PPG_BUF_SEC * PPG_FS)
    imu_samples = int(IMU_BUF_SEC * IMU_FS)
    return {
        "eeg": np.zeros((EEG_CHANNELS, eeg_samples), dtype=np.float32),
        "ppg": np.zeros(ppg_samples, dtype=np.float32),
        "accel": np.zeros((3, imu_samples), dtype=np.float32),
        "gyro": np.zeros((3, imu_samples), dtype=np.float32),
    }


def compute_band_powers_from_buffer(
    eeg_buf: np.ndarray,
    fs: float = EEG_FS,
) -> dict[str, float]:
    """Compute average band powers across all EEG channels.

    Returns dict with delta, theta, alpha, beta, gamma keys.
    Uses mean across channels then normalises to sum=1.
    """
    bands_raw: dict[str, float] = {}
    band_ranges = {
        "delta": (1.0, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "beta": (13.0, 30.0),
        "gamma": (30.0, 50.0),
    }
    n_ch = eeg_buf.shape[0] if eeg_buf.ndim == 2 else 1
    sig_list = [eeg_buf[i] for i in range(n_ch)] if eeg_buf.ndim == 2 else [eeg_buf]

    for band, (lo, hi) in band_ranges.items():
        powers = [bandpower(sig, lo, hi, fs) for sig in sig_list]
        bands_raw[band] = float(np.mean(powers))

    total = sum(bands_raw.values())
    if total <= 0:
        return {b: 0.0 for b in band_ranges}
    return {b: v / total for b, v in bands_raw.items()}
