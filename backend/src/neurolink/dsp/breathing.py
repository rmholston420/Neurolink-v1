"""Respiratory rate computation: PPG-FM + IMU accel-z fused estimate.

Ported from Rigpa-v3 dsp/breathing.py.
Requires scipy. Degrades gracefully if not installed.
All functions are pure.
"""
from __future__ import annotations

import numpy as np

from neurolink.models.eeg import BreathingPayload

_MIN_IBIS: int = 10      # minimum IBIs for PPG-FM
_MIN_ACCEL: int = 52 * 5  # minimum accel samples for FFT (5s @ 52 Hz)
_ACCEL_FS: float = 52.0
_IBI_FS_INTERP: float = 4.0  # resample IBI to 4 Hz for spectral analysis
_RR_BAND: tuple[float, float] = (0.1, 0.5)  # 6-30 bpm respiratory band


def compute_breathing(
    ibis_ms: list[float],
    accel_z: np.ndarray | None = None,
) -> BreathingPayload:
    """Compute fused respiratory rate from PPG IBIs and/or IMU accel-z.

    Args:
        ibis_ms: list of IBI values in milliseconds
        accel_z: 1-D accel Z-axis array at _ACCEL_FS Hz (optional)

    Returns:
        BreathingPayload with rr_bpm (fused), rr_ppg, rr_accel.
        Fields are None if insufficient data or scipy unavailable.
    """
    rr_ppg: float | None = None
    rr_accel: float | None = None

    # PPG-FM estimate
    if len(ibis_ms) >= _MIN_IBIS:
        rr_ppg = _rr_from_ibis(ibis_ms)

    # Accel-z estimate
    if accel_z is not None and len(accel_z) >= _MIN_ACCEL:
        rr_accel = _rr_from_accel(accel_z)

    # Fusion: average available estimates
    estimates = [e for e in (rr_ppg, rr_accel) if e is not None]
    rr_bpm = float(np.mean(estimates)) if estimates else None

    return BreathingPayload(rr_bpm=rr_bpm, rr_ppg=rr_ppg, rr_accel=rr_accel)


def _rr_from_ibis(ibis_ms: list[float]) -> float | None:
    """Estimate RR from IBIs via FFT peak detection."""
    try:
        from scipy.signal import welch
        from scipy.interpolate import interp1d

        ibi = np.array(ibis_ms) / 1000.0  # to seconds
        cumulative = np.cumsum(ibi)
        # Resample to uniform grid
        t_end = cumulative[-1]
        t_uniform = np.arange(0, t_end, 1.0 / _IBI_FS_INTERP)
        if len(t_uniform) < 8:
            return None
        interp_fn = interp1d(cumulative, ibi, kind="linear", fill_value="extrapolate")
        ibi_interp = interp_fn(t_uniform)
        # Welch PSD
        freqs, psd = welch(ibi_interp, fs=_IBI_FS_INTERP, nperseg=min(len(ibi_interp), 64))
        mask = (freqs >= _RR_BAND[0]) & (freqs <= _RR_BAND[1])
        if not np.any(mask):
            return None
        peak_freq = freqs[mask][np.argmax(psd[mask])]
        return float(peak_freq * 60.0)
    except Exception:
        return None


def _rr_from_accel(accel_z: np.ndarray) -> float | None:
    """Estimate RR from accel-z via Welch PSD."""
    try:
        from scipy.signal import welch

        freqs, psd = welch(accel_z, fs=_ACCEL_FS, nperseg=min(len(accel_z), 256))
        mask = (freqs >= _RR_BAND[0]) & (freqs <= _RR_BAND[1])
        if not np.any(mask):
            return None
        peak_freq = freqs[mask][np.argmax(psd[mask])]
        return float(peak_freq * 60.0)
    except Exception:
        return None
