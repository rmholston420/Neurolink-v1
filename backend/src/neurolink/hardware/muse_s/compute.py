"""Welch-based band power computation for Muse S EEG data.

Ported from Rigpa-v3 hardware/muse_s/compute.py.
"""
from __future__ import annotations

import numpy as np

from neurolink.models.eeg import BandPowers

EEG_FS: float = 256.0

_BAND_RANGES: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 50.0),
}


def compute_all_bands(
    channel_samples: dict[str, np.ndarray] | np.ndarray,
    fs: float = EEG_FS,
) -> BandPowers:
    """Compute band powers using Welch PSD, averaged across channels.

    Args:
        channel_samples: dict of {channel_name: 1D array} OR 2D array (ch x samples)
        fs: sampling frequency in Hz

    Returns:
        BandPowers with normalised power fractions (sum ~ 1.0)
    """
    try:
        import scipy.signal as signal  # lazy import

        # Normalise input to list of arrays
        if isinstance(channel_samples, np.ndarray):
            if channel_samples.ndim == 1:
                arrays = [channel_samples]
            else:
                arrays = [channel_samples[i] for i in range(channel_samples.shape[0])]
        else:
            arrays = list(channel_samples.values())

        if not arrays or all(len(a) < 2 for a in arrays):
            return BandPowers()

        band_powers: dict[str, list[float]] = {b: [] for b in _BAND_RANGES}

        for arr in arrays:
            if len(arr) < 2:
                continue
            nperseg = min(len(arr), int(fs * 2))  # 2-second window
            freqs, pxx = signal.welch(arr.astype(float), fs=fs, nperseg=nperseg)
            for band, (lo, hi) in _BAND_RANGES.items():
                mask = (freqs >= lo) & (freqs <= hi)
                if mask.any():
                    band_powers[band].append(float(np.mean(pxx[mask])))

        means: dict[str, float] = {}
        for band in _BAND_RANGES:
            vals = band_powers[band]
            means[band] = float(np.mean(vals)) if vals else 0.0

        total = sum(means.values())
        if total > 0:
            means = {b: v / total for b, v in means.items()}

        return BandPowers(
            delta=means.get("delta", 0.0),
            theta=means.get("theta", 0.0),
            alpha=means.get("alpha", 0.0),
            beta=means.get("beta", 0.0),
            gamma=means.get("gamma", 0.0),
        )
    except Exception:
        return BandPowers()
