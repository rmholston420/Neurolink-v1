"""Derived EEG metrics: FAA (Frontal Alpha Asymmetry) and FMt (Frontal Midline Theta).

Ported from Rigpa-v2 muse_compute.py.
All functions are pure.
"""
from __future__ import annotations

import math

import numpy as np

from neurolink.dsp.bandpower import bandpower

_DEFAULT_FS: float = 256.0
_MIN_SAMPLES: int = 64  # at least 0.25s @ 256 Hz

# Channel indices in standard Muse 5-channel layout
_TP9 = 0
_AF7 = 1
_AF8 = 2
_TP10 = 3
_AUX = 4

# Alpha band
_ALPHA_LO: float = 8.0
_ALPHA_HI: float = 13.0
# Theta band
_THETA_LO: float = 4.0
_THETA_HI: float = 8.0


def derived_eeg(
    eeg: np.ndarray,
    fs: float = _DEFAULT_FS,
) -> dict[str, float | None]:
    """Compute FAA and FMt from a 5-channel EEG buffer.

    Args:
        eeg: 2-D array of shape (5, n_samples)
        fs: sampling frequency (Hz)

    Returns:
        Dict with keys:
        - "faa": Frontal Alpha Asymmetry (log(AF8_alpha) - log(AF7_alpha))
                 Positive = right alpha > left alpha (approach motivation)
        - "fmt": Frontal Midline Theta (AUX channel theta power)
        Both may be None if buffer is too short.
    """
    result: dict[str, float | None] = {"faa": None, "fmt": None}

    if eeg.shape[1] < _MIN_SAMPLES or eeg.shape[0] < 5:
        return result

    # FAA: ln(AF8 alpha) - ln(AF7 alpha)
    af7_alpha = bandpower(eeg[_AF7], _ALPHA_LO, _ALPHA_HI, fs)
    af8_alpha = bandpower(eeg[_AF8], _ALPHA_LO, _ALPHA_HI, fs)
    if af7_alpha > 0 and af8_alpha > 0:
        result["faa"] = math.log(af8_alpha) - math.log(af7_alpha)

    # FMt: AUX channel theta
    aux_theta = bandpower(eeg[_AUX], _THETA_LO, _THETA_HI, fs)
    if aux_theta > 0:
        result["fmt"] = aux_theta

    return result
