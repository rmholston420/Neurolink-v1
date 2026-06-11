"""Derived EEG metrics: FAA and FMt.

Frontal Alpha Asymmetry (FAA): AF8 alpha - AF7 alpha.
Frontal Midline Theta (FMt): Midline (AUX) theta.

Ported from Rigpa-v2 dsp/derived_eeg.py.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from neurolink.dsp.bandpower import bandpower

_EEG_FS: float = 256.0
# EEG channel indices (Muse 5-channel layout)
_CH_TP9: int = 0
_CH_AF7: int = 1
_CH_AF8: int = 2
_CH_TP10: int = 3
_CH_AUX: int = 4

_MIN_SAMPLES: int = 256  # ~1 second at 256 Hz


def derived_eeg(eeg: np.ndarray, fs: float = _EEG_FS) -> Dict[str, float | None]:
    """Compute FAA and FMt from a (5, N) EEG buffer.

    Args:
        eeg: EEG array of shape (5, N).
        fs: Sampling rate (Hz).

    Returns:
        Dict with keys 'faa' and 'fmt', both float or None if insufficient data.
    """
    result: Dict[str, float | None] = {"faa": None, "fmt": None}

    if eeg is None or eeg.ndim < 2:
        return result

    n_channels, n_samples = eeg.shape
    if n_samples < _MIN_SAMPLES or n_channels < 5:
        return result

    # FAA: log(AF8 alpha) - log(AF7 alpha)
    alpha_af8 = bandpower(eeg[_CH_AF8], lo=8.0, hi=13.0, fs=fs)
    alpha_af7 = bandpower(eeg[_CH_AF7], lo=8.0, hi=13.0, fs=fs)

    if alpha_af8 > 0 and alpha_af7 > 0:
        result["faa"] = float(np.log(alpha_af8) - np.log(alpha_af7))
    elif alpha_af8 > 0:
        result["faa"] = 1.0
    elif alpha_af7 > 0:
        result["faa"] = -1.0
    else:
        result["faa"] = 0.0

    # FMt: frontal midline theta (AUX channel)
    theta_aux = bandpower(eeg[_CH_AUX], lo=4.0, hi=8.0, fs=fs)
    result["fmt"] = float(theta_aux) if theta_aux > 0 else 0.0

    return result
