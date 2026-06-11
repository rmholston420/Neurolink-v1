"""PPG processing: HR, IBI, HRV, Poincaré indices.

Ported from Rigpa-v2 ppg.py + Rigpa-v3 dsp/ppg.py.
Requires neurokit2 and scipy. Degrades gracefully if not installed.
All functions are pure.
"""
from __future__ import annotations

import math

import numpy as np

from neurolink.models.eeg import PoincareIndices, PPGPayload

_MIN_SAMPLES_FOR_PPG: int = 320  # 5s @ 64 Hz
_PPG_FS_DEFAULT: float = 64.0


def compute_ppg(ppg_ir: np.ndarray, fs: float = _PPG_FS_DEFAULT) -> PPGPayload:
    """Compute HR, IBI, HRV from a PPG IR waveform.

    Args:
        ppg_ir: 1-D PPG IR waveform array
        fs: sampling frequency (Hz)

    Returns:
        PPGPayload with hr_bpm, IBI list, RMSSD, SDNN, pNN50, and Poincaré.
        Returns empty PPGPayload if buffer is too short or neurokit2 unavailable.
    """
    if len(ppg_ir) < _MIN_SAMPLES_FOR_PPG:
        return PPGPayload()

    try:
        import neurokit2 as nk  # lazy import

        signals, info = nk.ppg_process(ppg_ir, sampling_rate=int(fs))
        peaks = info.get("PPG_Peaks", [])
        if len(peaks) < 2:
            return PPGPayload()

        # Compute IBIs (ms)
        peak_times = np.array(peaks) / fs
        ibis_s = np.diff(peak_times)
        ibis_ms = (ibis_s * 1000.0).tolist()

        # HR
        hr_bpm = float(60.0 / np.mean(ibis_s)) if ibis_s.size > 0 else 0.0

        # HRV
        rmssd = _compute_rmssd(ibis_ms)
        sdnn = _compute_sdnn(ibis_ms)
        pnn50 = _compute_pnn50(ibis_ms)
        poincare = _poincare(ibis_ms)

        return PPGPayload(
            hr_bpm=hr_bpm,
            ibi_ms=ibis_ms,
            hrv_rmssd=rmssd,
            hrv_sdnn=sdnn,
            hrv_pnn50=pnn50,
            poincare=poincare,
        )

    except Exception:
        return PPGPayload()


def _compute_rmssd(ibis_ms: list[float]) -> float:
    """Root mean square of successive differences."""
    if len(ibis_ms) < 2:
        return 0.0
    diffs = np.diff(np.array(ibis_ms))
    return float(math.sqrt(float(np.mean(diffs ** 2))))


def _compute_sdnn(ibis_ms: list[float]) -> float:
    """Standard deviation of NN intervals."""
    if len(ibis_ms) < 2:
        return 0.0
    return float(np.std(np.array(ibis_ms)))


def _compute_pnn50(ibis_ms: list[float]) -> float:
    """Percentage of successive differences > 50 ms."""
    if len(ibis_ms) < 2:
        return 0.0
    diffs = np.abs(np.diff(np.array(ibis_ms)))
    return float(np.mean(diffs > 50.0) * 100.0)


def _poincare(ibis_ms: list[float]) -> PoincareIndices:
    """Compute Poincaré plot indices (SD1, SD2) from IBI list.

    Args:
        ibis_ms: list of IBI values in milliseconds

    Returns:
        PoincareIndices with sd1, sd2, sd1/sd2 ratio, and ellipse area.
    """
    if len(ibis_ms) < 2:
        return PoincareIndices()

    ibi = np.array(ibis_ms)
    x = ibi[:-1]
    y = ibi[1:]

    # SD1 = std of (y - x) / sqrt(2)
    diff_series = (y - x) / math.sqrt(2)
    sd1 = float(np.std(diff_series))

    # SD2 = std of (y + x) / sqrt(2)
    sum_series = (y + x) / math.sqrt(2)
    sd2 = float(np.std(sum_series))

    ratio = sd1 / sd2 if sd2 > 0 else 0.0
    area = math.pi * sd1 * sd2

    return PoincareIndices(sd1=sd1, sd2=sd2, sd1_sd2_ratio=ratio, ellipse_area=area)
