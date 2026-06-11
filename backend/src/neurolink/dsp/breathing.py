"""Breathing rate estimation from IBIs and accelerometer.

Ported from Rigpa-v3 dsp/breathing.py.

Two methods:
1. Respiratory Sinus Arrhythmia (RSA): FFT on IBI series
2. Accelerometer: FFT on accel-z axis

Fused result averages both methods when available.
"""
from __future__ import annotations

import numpy as np

from neurolink.models.eeg import BreathingPayload

_ACCEL_FS: float = 52.0
_IBI_FS_VIRTUAL: float = 4.0  # resample IBIs to 4 Hz virtual series
_RR_MIN_HZ: float = 0.1   # 6 bpm
_RR_MAX_HZ: float = 0.55  # 33 bpm
_MIN_IBIS: int = 10
_MIN_ACCEL_SAMPLES: int = int(_ACCEL_FS * 10)  # 10 seconds


def compute_breathing(
    ibi_ms: list[float],
    accel_z: np.ndarray | None = None,
    accel_fs: float = _ACCEL_FS,
) -> BreathingPayload:
    """Estimate breathing rate from IBIs and/or accelerometer.

    Args:
        ibi_ms: List of IBI values in milliseconds.
        accel_z: 1D accelerometer z-axis array (optional).
        accel_fs: Accelerometer sampling rate (Hz).

    Returns:
        BreathingPayload with rr_bpm (fused), rr_ppg, rr_accel.
    """
    rr_ppg = _rr_from_ibis(ibi_ms) if len(ibi_ms) >= _MIN_IBIS else None
    rr_accel = (
        _rr_from_accel(accel_z, accel_fs)
        if accel_z is not None and len(accel_z) >= _MIN_ACCEL_SAMPLES
        else None
    )

    if rr_ppg is not None and rr_accel is not None:
        rr_bpm = (rr_ppg + rr_accel) / 2.0
    elif rr_ppg is not None:
        rr_bpm = rr_ppg
    elif rr_accel is not None:
        rr_bpm = rr_accel
    else:
        rr_bpm = None

    return BreathingPayload(rr_bpm=rr_bpm, rr_ppg=rr_ppg, rr_accel=rr_accel)


def _rr_from_ibis(ibi_ms: list[float]) -> float | None:
    """Estimate respiratory rate from IBI series via FFT."""
    arr = np.array(ibi_ms, dtype=np.float32)
    if len(arr) < _MIN_IBIS:
        return None

    # Detrend and window
    arr -= arr.mean()
    arr *= np.hanning(len(arr))

    # Zero-pad for frequency resolution
    n_fft = max(len(arr), 512)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / _IBI_FS_VIRTUAL)
    psd = np.abs(np.fft.rfft(arr, n=n_fft)) ** 2

    mask = (freqs >= _RR_MIN_HZ) & (freqs <= _RR_MAX_HZ)
    if not mask.any():
        return None

    peak_freq = freqs[mask][np.argmax(psd[mask])]
    return float(peak_freq * 60.0)


def _rr_from_accel(accel_z: np.ndarray, fs: float) -> float | None:
    """Estimate respiratory rate from accelerometer z-axis via FFT."""
    if len(accel_z) < _MIN_ACCEL_SAMPLES:
        return None

    arr = accel_z - accel_z.mean()
    arr *= np.hanning(len(arr))

    freqs = np.fft.rfftfreq(len(arr), d=1.0 / fs)
    psd = np.abs(np.fft.rfft(arr)) ** 2

    mask = (freqs >= _RR_MIN_HZ) & (freqs <= _RR_MAX_HZ)
    if not mask.any():
        return None

    peak_freq = freqs[mask][np.argmax(psd[mask])]
    return float(peak_freq * 60.0)
