"""Athena-specific band power computation.

Ported from Rigpa-v3 hardware/muse_athena/compute.py.
Uses the same Welch approach as Muse S but may receive different channel layouts.
"""
from __future__ import annotations

import numpy as np

from neurolink.models.eeg import BandPowers
from neurolink.hardware.muse_s.compute import compute_all_bands, EEG_FS


def compute_athena_bands(
    channel_samples: dict[str, np.ndarray] | np.ndarray,
    fs: float = EEG_FS,
) -> BandPowers:
    """Compute band powers for Muse Athena EEG data.

    Delegates to the shared Welch-based compute_all_bands function.
    """
    return compute_all_bands(channel_samples, fs=fs)
