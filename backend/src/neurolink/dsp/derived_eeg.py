"""Derived EEG metrics: Frontal Alpha Asymmetry (FAA) and Frontal Midline Theta (FMt)."""
from __future__ import annotations

import math

import numpy as np

from neurolink.dsp.bandpower import bandpower, EEG_FS

# Channel indices in 5-channel EEG buffer: TP9=0, AF7=1, AF8=2, TP10=3, AUX=4
_CH_AF7 = 1
_CH_AF8 = 2
_CH_AUX = 4  # FPz proxy


def derived_eeg(
    eeg_bufs: np.ndarray,
    fs: float = EEG_FS,
) -> dict[str, float | None]:
    """Compute FAA and FMt from a 5-channel EEG buffer.

    Args:
        eeg_bufs: shape (5, N) array — channels: TP9, AF7, AF8, TP10, AUX
        fs: sampling frequency in Hz

    Returns:
        dict with keys 'faa' (float) and 'fmt' (float | None)
    """
    result: dict[str, float | None] = {"faa": None, "fmt": None}

    if eeg_bufs.ndim != 2 or eeg_bufs.shape[0] < 3:
        return result

    n = eeg_bufs.shape[1]
    if n < 2:
        return result

    # FAA = ln(alpha_AF8) - ln(alpha_AF7)
    # Positive FAA = approach motivation; negative = withdrawal
    alpha_af7 = bandpower(eeg_bufs[_CH_AF7], 8.0, 13.0, fs)
    alpha_af8 = bandpower(eeg_bufs[_CH_AF8], 8.0, 13.0, fs)

    if alpha_af7 > 0.0 and alpha_af8 > 0.0:
        result["faa"] = math.log(alpha_af8) - math.log(alpha_af7)
    else:
        result["faa"] = 0.0

    # FMt = frontal midline theta from AUX (FPz proxy)
    if eeg_bufs.shape[0] > _CH_AUX:
        result["fmt"] = bandpower(eeg_bufs[_CH_AUX], 4.0, 8.0, fs)
    else:
        result["fmt"] = None

    return result
