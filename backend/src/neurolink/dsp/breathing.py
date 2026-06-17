"""Breathing rate estimation from IBIs and accelerometer.

Two methods:
1. Respiratory Sinus Arrhythmia (RSA): FFT on IBI series
2. Accelerometer: FFT on accel-z axis

Fused result averages both when available.
"""

from __future__ import annotations

import numpy as np

from neurolink.models.eeg import BreathingPayload

_ACCEL_FS: float = 52.0
_IBI_FS_VIRTUAL: float = 4.0
_RR_MIN_HZ: float = 0.1   # 6 bpm
_RR_MAX_HZ: float = 0.55  # 33 bpm
_MIN_IBIS: int = 10
_MIN_ACCEL_SAMPLES: int = int(_ACCEL_FS * 10)  # 10 seconds

# If the raw signal is constant (std == 0 after mean-subtract) there is no
# oscillatory content and we return None.  We use a std threshold rather than
# a PSD threshold so that real IBI sequences with genuinely low but non-zero
# variance (e.g. [800]*30 which is a perfectly valid but degenerate input)
# are correctly identified as non-oscillatory without suppressing real signals.
_STD_NOISE_FLOOR: float = 1e-9


def estimate_rr(
    signal: list[float] | np.ndarray,
    fs: float = _ACCEL_FS,
) -> float | None:
    """Estimate respiratory rate (bpm) from any 1D biosignal via FFT.

    Returns:
        Estimated respiratory rate in bpm, or None if the signal is too
        short, constant, or has no peak in the physiological range.
    """
    if signal is None or len(signal) < 2:
        return None

    arr = np.asarray(signal, dtype=np.float64)
    arr = arr - arr.mean()

    if arr.std() < _STD_NOISE_FLOOR:
        return None

    arr *= np.hanning(len(arr))

    n_fft = max(len(arr), 512)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    psd = np.abs(np.fft.rfft(arr, n=n_fft)) ** 2

    mask = (freqs >= _RR_MIN_HZ) & (freqs <= _RR_MAX_HZ)
    if not mask.any():
        return None

    peak_freq = freqs[mask][np.argmax(psd[mask])]
    return float(peak_freq * 60.0)


def compute_breathing(
    ibi_ms: list[float],
    accel_z: np.ndarray | None = None,
    accel_fs: float = _ACCEL_FS,
) -> BreathingPayload:
    """Estimate breathing rate from IBIs and/or accelerometer."""
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
    arr = np.array(ibi_ms, dtype=np.float64)
    if len(arr) < _MIN_IBIS:
        return None

    arr = arr - arr.mean()
    if arr.std() < _STD_NOISE_FLOOR:
        return None

    arr *= np.hanning(len(arr))

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

    arr = accel_z.astype(np.float64) - accel_z.mean()
    if arr.std() < _STD_NOISE_FLOOR:
        return None

    arr *= np.hanning(len(arr))

    freqs = np.fft.rfftfreq(len(arr), d=1.0 / fs)
    psd = np.abs(np.fft.rfft(arr)) ** 2

    mask = (freqs >= _RR_MIN_HZ) & (freqs <= _RR_MAX_HZ)
    if not mask.any():
        return None

    peak_freq = freqs[mask][np.argmax(psd[mask])]
    return float(peak_freq * 60.0)
