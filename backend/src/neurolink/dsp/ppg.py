"""PPG / HRV computation.

Ported from Rigpa-v2 dsp/ppg.py + Rigpa-v3 dsp/ppg.py.
Uses neurokit2 for R-peak detection and HRV metrics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import structlog

from neurolink.models.eeg import PPGPayload

log = structlog.get_logger(__name__)

_PPG_FS: float = 64.0
_MIN_SAMPLES: int = int(_PPG_FS * 10)  # 10 seconds minimum
_HR_VALID_MIN: float = 30.0
_HR_VALID_MAX: float = 200.0


@dataclass
class PoincareMetrics:
    """Poincare plot HRV metrics."""

    sd1: float = 0.0
    sd2: float = 0.0
    ellipse_area: float = 0.0


def compute_ppg(ppg_arr: np.ndarray, fs: float = _PPG_FS) -> PPGPayload:
    """Compute PPG-derived HR and HRV from a PPG buffer.

    Args:
        ppg_arr: 1D PPG signal array.
        fs: Sampling rate (Hz).

    Returns:
        PPGPayload with hr_bpm, hrv_rmssd, ibi_ms, and Poincare metrics.
    """
    empty = PPGPayload(hr_bpm=0.0, hrv_rmssd=0.0, ibi_ms=[])

    if ppg_arr is None or len(ppg_arr) < _MIN_SAMPLES:
        return empty

    try:
        import neurokit2 as nk

        _processed, info = nk.ppg_process(ppg_arr, sampling_rate=int(fs))
        peaks = info.get("PPG_Peaks", [])

        if len(peaks) < 3:
            return empty

        # Compute IBI in ms
        ibi_ms = list((np.diff(peaks) / fs * 1000).astype(float))

        # Filter physiologically valid IBIs
        ibi_ms = [ibi for ibi in ibi_ms if 300 <= ibi <= 2000]

        if not ibi_ms:
            return empty

        hr_bpm = 60000.0 / float(np.mean(ibi_ms))
        if not (_HR_VALID_MIN <= hr_bpm <= _HR_VALID_MAX):
            return empty

        # RMSSD
        if len(ibi_ms) >= 2:
            diffs = np.diff(ibi_ms)
            hrv_rmssd = float(np.sqrt(np.mean(diffs**2)))
        else:
            hrv_rmssd = 0.0

        poincare = _poincare(ibi_ms)

        return PPGPayload(
            hr_bpm=hr_bpm,
            hrv_rmssd=hrv_rmssd,
            ibi_ms=ibi_ms,
            sd1=poincare.sd1,
            sd2=poincare.sd2,
            ellipse_area=poincare.ellipse_area,
        )

    except Exception as exc:
        log.warning("ppg_compute_error", error=str(exc))
        return empty


def _poincare(ibi_ms: list[float]) -> PoincareMetrics:
    """Compute Poincare plot HRV metrics (SD1, SD2, ellipse area).

    Args:
        ibi_ms: List of IBI values in milliseconds.

    Returns:
        PoincareMetrics with sd1, sd2, ellipse_area.
    """
    if len(ibi_ms) < 2:
        return PoincareMetrics()

    arr = np.array(ibi_ms)
    rr_n = arr[:-1]
    rr_n1 = arr[1:]

    sd1 = float(np.std((rr_n1 - rr_n) / math.sqrt(2)))
    sd2 = float(np.std((rr_n1 + rr_n) / math.sqrt(2)))
    ellipse_area = math.pi * sd1 * sd2

    return PoincareMetrics(sd1=sd1, sd2=sd2, ellipse_area=ellipse_area)
