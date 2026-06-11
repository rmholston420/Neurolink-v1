"""Fused respiratory rate from PPG and IMU."""
from __future__ import annotations

import numpy as np

from neurolink.models.eeg import BreathingPayload

_MIN_IBI_FOR_FM: int = 6  # minimum IBI samples for PPG-FM method
_MIN_ACCEL_SAMPLES: int = 52  # minimum accel samples (1 second at 52 Hz)


def _ppg_fm_rr(ibi_ms: list[float]) -> float | None:
    """Estimate respiratory rate from IBI via frequency modulation (PPG-FM).

    Looks for dominant frequency in 0.1-0.5 Hz range (6-30 bpm) in the
    IBI time series.
    """
    if len(ibi_ms) < _MIN_IBI_FOR_FM:
        return None
    try:
        import scipy.signal as signal  # lazy import

        ibi = np.array(ibi_ms, dtype=float)
        # Resample at 4 Hz (IBI is unevenly spaced, approximate as evenly spaced)
        fs_ibi = 4.0
        freqs, pxx = signal.welch(ibi, fs=fs_ibi, nperseg=min(len(ibi), 32))
        mask = (freqs >= 0.1) & (freqs <= 0.5)
        if not mask.any():
            return None
        peak_freq = freqs[mask][np.argmax(pxx[mask])]
        rr_bpm = float(peak_freq * 60.0)
        return rr_bpm if 4.0 < rr_bpm < 35.0 else None
    except Exception:
        return None


def _accel_rr(accel_z: np.ndarray, fs: float = 52.0) -> float | None:
    """Estimate respiratory rate from accelerometer Z-axis."""
    if len(accel_z) < _MIN_ACCEL_SAMPLES:
        return None
    try:
        import scipy.signal as signal  # lazy import

        freqs, pxx = signal.welch(accel_z, fs=fs, nperseg=min(len(accel_z), 128))
        mask = (freqs >= 0.1) & (freqs <= 0.5)
        if not mask.any():
            return None
        peak_freq = freqs[mask][np.argmax(pxx[mask])]
        rr_bpm = float(peak_freq * 60.0)
        return rr_bpm if 4.0 < rr_bpm < 35.0 else None
    except Exception:
        return None


def compute_breathing(
    ibis_ms: list[float],
    accel_z: np.ndarray | None = None,
) -> BreathingPayload:
    """Compute fused respiratory rate from PPG-FM and IMU accel-z.

    Fuses the two estimates by averaging when both are available.
    Falls back to whichever is available. Returns empty payload if neither.

    Args:
        ibis_ms: IBI sequence in milliseconds
        accel_z: 1D accelerometer Z-axis signal at IMU_FS Hz

    Returns:
        BreathingPayload with rr_bpm, rr_ppg, rr_accel.
    """
    rr_ppg = _ppg_fm_rr(ibis_ms)
    rr_accel = _accel_rr(accel_z) if accel_z is not None else None

    if rr_ppg is not None and rr_accel is not None:
        rr_fused = (rr_ppg + rr_accel) / 2.0
    elif rr_ppg is not None:
        rr_fused = rr_ppg
    elif rr_accel is not None:
        rr_fused = rr_accel
    else:
        rr_fused = None

    return BreathingPayload(
        rr_bpm=rr_fused,
        rr_ppg=rr_ppg,
        rr_accel=rr_accel,
    )
