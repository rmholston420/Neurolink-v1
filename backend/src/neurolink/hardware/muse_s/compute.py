"""Welch-based band power computation for Muse S Gen 1.

Ported from Rigpa-v3 hardware/muse_s/compute.py.
All functions are pure.
"""
from __future__ import annotations

import numpy as np

_EEG_FS: float = 256.0
_BANDS: dict[str, tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def compute_all_bands(
    channel_samples: dict[str, list[float]],
    fs: float = _EEG_FS,
) -> dict[str, float]:
    """Compute Welch-based normalised band powers from channel samples.

    Args:
        channel_samples: dict mapping channel name to list of float samples
        fs: sampling frequency (Hz)

    Returns:
        Dict mapping band name to normalised fraction [0, 1].
    """
    try:
        from scipy.signal import welch
    except ImportError:
        from neurolink.dsp.bandpower import compute_band_powers_from_buffer
        arrays = np.array(
            [channel_samples.get(ch, [0.0]) for ch in ["TP9", "AF7", "AF8", "TP10", "AUX"]],
            dtype=np.float32,
        )
        # Pad if needed
        max_len = max(arr.shape[0] for arr in arrays) if arrays.shape[0] else 2
        padded = np.zeros((arrays.shape[0], max_len), dtype=np.float32)
        for i, arr in enumerate(arrays):
            padded[i, :len(arr)] = arr
        return compute_band_powers_from_buffer(padded, fs=fs)

    band_powers: dict[str, float] = {k: 0.0 for k in _BANDS}
    n_channels = 0

    for ch_samples in channel_samples.values():
        if len(ch_samples) < 4:
            continue
        sig = np.array(ch_samples, dtype=np.float32)
        nperseg = min(len(sig), 256)
        freqs, psd = welch(sig, fs=fs, nperseg=nperseg)
        for band, (lo, hi) in _BANDS.items():
            mask = (freqs >= lo) & (freqs < hi)
            if np.any(mask):
                band_powers[band] += float(np.mean(psd[mask]))
        n_channels += 1

    if n_channels == 0:
        return {k: 0.0 for k in _BANDS}

    # Average across channels and normalise
    for k in band_powers:
        band_powers[k] /= n_channels
    total = sum(band_powers.values())
    if total <= 0:
        return {k: 0.0 for k in _BANDS}
    return {k: v / total for k, v in band_powers.items()}
