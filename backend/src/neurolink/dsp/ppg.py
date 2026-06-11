"""PPG processing: HR, IBI, HRV, Poincare indices."""
from __future__ import annotations

import math

import numpy as np

from neurolink.models.eeg import PoincareIndices, PPGPayload

PPG_FS: float = 64.0
_MIN_SAMPLES_FOR_PPG: int = int(PPG_FS * 5)  # 5 seconds minimum


def _poincare(ibi_ms: list[float]) -> PoincareIndices:
    """Compute Poincare plot indices from IBI sequence (ms).

    SD1 = short-term HRV (beat-to-beat)
    SD2 = long-term HRV (trend)
    """
    if len(ibi_ms) < 2:
        return PoincareIndices()
    ibi = np.array(ibi_ms, dtype=float)
    diff = np.diff(ibi)
    sd1 = float(np.std(diff) / math.sqrt(2))
    sd2 = float(math.sqrt(max(2.0 * np.var(ibi) - np.var(diff) / 2.0, 0.0)))
    ratio = sd1 / sd2 if sd2 > 0 else 0.0
    area = math.pi * sd1 * sd2
    return PoincareIndices(
        sd1=sd1,
        sd2=sd2,
        sd1_sd2_ratio=ratio,
        ellipse_area=area,
    )


def compute_ppg(
    ppg_ir: np.ndarray,
    fs: float = PPG_FS,
) -> PPGPayload:
    """Compute HR, IBI, and HRV metrics from a PPG IR buffer.

    Requires neurokit2. Degrades gracefully to empty payload if unavailable
    or if buffer is too short.

    Args:
        ppg_ir: 1D numpy array of PPG IR signal
        fs: sampling rate in Hz

    Returns:
        PPGPayload with HR, IBI, RMSSD, SDNN, pNN50, and Poincare.
    """
    if len(ppg_ir) < _MIN_SAMPLES_FOR_PPG:
        return PPGPayload()

    try:
        import neurokit2 as nk  # lazy import

        signals, info = nk.ppg_process(ppg_ir, sampling_rate=int(fs))
        peaks = info.get("PPG_Peaks", [])
        if peaks is None or len(peaks) < 2:
            return PPGPayload()

        peak_times_s = np.array(peaks) / fs
        ibi_s = np.diff(peak_times_s)
        ibi_ms = list(ibi_s * 1000.0)

        hr_bpm = float(60.0 / np.mean(ibi_s)) if len(ibi_s) > 0 else 0.0
        # Clamp HR to physiologically plausible range
        hr_bpm = max(30.0, min(200.0, hr_bpm))

        rmssd = float(np.sqrt(np.mean(np.diff(ibi_ms) ** 2))) if len(ibi_ms) > 1 else 0.0
        sdnn = float(np.std(ibi_ms)) if len(ibi_ms) > 1 else 0.0

        if len(ibi_ms) > 1:
            nn_diffs = np.abs(np.diff(ibi_ms))
            pnn50 = float(np.sum(nn_diffs > 50.0) / len(nn_diffs))
        else:
            pnn50 = 0.0

        poincare = _poincare(ibi_ms)

        return PPGPayload(
            hr_bpm=hr_bpm,
            ibi_ms=ibi_ms,
            hrv_rmssd=rmssd,
            hrv_sdnn=sdnn,
            hrv_pnn50=pnn50,
            poincare=poincare,
        )
    except Exception:
        return PPGPayload()
