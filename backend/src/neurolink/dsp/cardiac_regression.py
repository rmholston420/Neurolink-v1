"""Stage 6 -- PPG-referenced cardiac artifact regression (AAS method).

Method: Average Artifact Subtraction
--------------------------------------
The cardiac (ballistocardiographic / pulse) artifact in EEG is near-periodic
and phase-locked to the heartbeat.  When a PPG channel is co-recorded (always
true on Muse hardware) we detect R-peak timing from PPG inter-beat intervals
(IBIs), extract EEG epochs around each beat, build a trimmed-mean template,
and subtract it on every beat.

Why AAS over ICA for cardiac removal
--------------------------------------
ICA requires >=64 channels for reliable source separation.  Muse hardware
has 4 EEG channels -- at that density ICA frequently confuses cardiac and
neural components.  AAS is validated at low channel counts, requires no
matrix decomposition, and runs in O(n_channels) per frame.

Reference
---------
Allen PJ, Polizzi G, Krakow K, Fish DR, Lemieux L (1998).
"Identification of EEG events in the MR scanner: the problem of pulse
artifact and a method for its subtraction". NeuroImage 8(3):229-239.

Graceful degradation
--------------------
When PPG IBI data is absent, insufficient, or out of physiological range
the corrector returns the EEG array unchanged.  No exception is raised.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import structlog

log = structlog.get_logger(__name__)


@dataclass
class CardiacRegressionConfig:
    """Tunable parameters for the cardiac artifact corrector.

    Attributes
    ----------
    enable:
        Master switch.  False -> corrector is a no-op.
    eeg_channels:
        Channel indices to correct.  Default [0,1,2,3] = TP9/AF7/AF8/TP10.
    half_win_ms:
        Half-width of the cardiac template epoch (ms).  400 ms spans the
        full cardiac waveform even at 40 bpm.
    template_beats:
        Number of successive beats averaged to build the template.
        8 beats ~= ~6 s at 75 bpm -- enough for a stable trimmed mean.
    recalib_beats:
        Refresh the template after this many beats.  Default 8 so the
        template is always built from the most recent window.
    trim_fraction:
        Proportion trimmed from each end.  Default 0.05 (5 %).
    min_ibi_ms:
        Minimum plausible IBI (ms).  Rejects spurious PPG detections.
        Default 400 ms (150 bpm upper bound).
    max_ibi_ms:
        Maximum plausible IBI (ms).  Rejects missed-beat errors.
        Default 2000 ms (30 bpm lower bound).
    """

    enable: bool = True
    eeg_channels: list[int] = field(default_factory=lambda: [0, 1, 2, 3])
    half_win_ms: float = 400.0
    template_beats: int = 8
    recalib_beats: int = 8
    trim_fraction: float = 0.05
    min_ibi_ms: float = 400.0
    max_ibi_ms: float = 2000.0


class CardiacRegressor:
    """PPG-referenced cardiac artifact corrector (AAS method).

    Usage (EEGPump Stage 6)
    -----------------------
    corrector = CardiacRegressor()

    # After Stage 5 (ocular regression), when plan.apply_cardiac_regression:
    if not artifact_rejected and toggles.stage6_cardiac and _plan_apply_cardiac:
        if ppg_payload is not None and ppg_payload.ibi_ms:
            eeg_arr = corrector.apply(eeg_arr, ppg_payload.ibi_ms, fs=_EEG_FS)

    Call reset() at session start / reconnect to clear accumulated state.
    """

    def __init__(self, config: CardiacRegressionConfig | None = None) -> None:
        self._lock = threading.Lock()
        self._cfg: CardiacRegressionConfig = config or CardiacRegressionConfig()
        self._template: np.ndarray | None = None  # (n_ch, win_samples)
        self._epoch_buffer: list[np.ndarray] = []
        self._beats_since_recalib: int = 0
        # Rolling ring of EEG sample columns (~2 s at 256 Hz)
        self._eeg_ring: deque[np.ndarray] = deque(maxlen=512)

    # Public API

    def apply(
        self,
        eeg: np.ndarray,
        ibi_ms: list[float],
        fs: float = 256.0,
    ) -> np.ndarray:
        """Apply cardiac template subtraction to one EEG frame.

        Parameters
        ----------
        eeg:    (n_channels, n_samples) float array, post Stages 1-5.
        ibi_ms: Inter-beat intervals (ms) from the PPG module.
        fs:     EEG sampling rate (Hz).

        Returns
        -------
        Corrected float32 array of the same shape as ``eeg``.
        Returns the original array unchanged if correction cannot proceed.
        """
        with self._lock:
            cfg = self._cfg

        if not cfg.enable or eeg is None or eeg.ndim != 2:
            return eeg
        if not ibi_ms:
            return eeg

        valid_ibis = [ibi for ibi in ibi_ms if cfg.min_ibi_ms <= ibi <= cfg.max_ibi_ms]
        if not valid_ibis:
            return eeg

        fs = float(fs)
        half_win = round(cfg.half_win_ms * fs / 1000.0)
        n_samples = eeg.shape[1]
        n_ch = eeg.shape[0]

        with self._lock:
            for s in range(n_samples):
                self._eeg_ring.append(eeg[:, s].copy())
            ring_len = len(self._eeg_ring)

        if ring_len < 2 * half_win + 1:
            return eeg

        last_ibi_samples = round(valid_ibis[-1] * fs / 1000.0)
        beat_pos = ring_len - 1 - max(0, last_ibi_samples // 2)
        start = beat_pos - half_win
        end = beat_pos + half_win + 1

        if start < 0 or end > ring_len:
            return eeg

        with self._lock:
            ring_snap = list(self._eeg_ring)

        ring_arr = np.array(ring_snap, dtype=np.float32).T  # (n_ch, ring_len)
        epoch = ring_arr[:, start:end]  # (n_ch, win_len)

        with self._lock:
            self._epoch_buffer.append(epoch)
            self._beats_since_recalib += 1

            if (
                len(self._epoch_buffer) >= cfg.template_beats
                and self._beats_since_recalib >= cfg.recalib_beats
            ):
                self._template = self._build_template(
                    self._epoch_buffer[-cfg.template_beats :],
                    cfg.trim_fraction,
                )
                self._beats_since_recalib = 0
                log.debug(
                    "cardiac_template_updated",
                    half_win=half_win,
                    last_ibi_ms=round(valid_ibis[-1], 1),
                    n_buffered=len(self._epoch_buffer),
                )

            template = self._template

        if template is None:
            return eeg

        eeg_out = eeg.astype(np.float32, copy=True)
        ch_idx = [i for i in cfg.eeg_channels if i < n_ch]

        frame_start_in_ring = ring_len - n_samples
        f_start = max(start, frame_start_in_ring) - frame_start_in_ring
        f_end = min(end, ring_len) - frame_start_in_ring
        f_start = max(0, f_start)
        f_end = min(n_samples, f_end)

        if f_start >= f_end:
            return eeg_out

        t_start = f_start + (frame_start_in_ring - start)
        t_start = max(0, t_start)
        t_len = f_end - f_start
        t_end = t_start + t_len

        if t_end > template.shape[1]:
            return eeg_out

        for ch in ch_idx:
            eeg_out[ch, f_start:f_end] -= template[ch, t_start:t_end]

        return eeg_out

    def reset(self) -> None:
        """Reset all accumulated state.  Call at session start / reconnect."""
        with self._lock:
            self._template = None
            self._epoch_buffer.clear()
            self._beats_since_recalib = 0
            self._eeg_ring.clear()
        log.info("cardiac_regressor_reset")

    def get_config(self) -> CardiacRegressionConfig:
        with self._lock:
            import copy

            return copy.copy(self._cfg)

    def set_config(self, config: CardiacRegressionConfig) -> None:
        with self._lock:
            self._cfg = config
        log.info("cardiac_regressor_config_updated")

    # Internal

    @staticmethod
    def _build_template(
        epochs: list[np.ndarray],
        trim_fraction: float,
    ) -> np.ndarray:
        """Trimmed-mean template across a list of (n_ch, win) beat epochs."""
        from scipy.stats import trim_mean

        stack = np.stack(epochs, axis=0)  # (n_beats, n_ch, win)
        n_ch, win = stack.shape[1], stack.shape[2]
        template = np.zeros((n_ch, win), dtype=np.float32)
        for ch in range(n_ch):
            for s in range(win):
                template[ch, s] = float(
                    trim_mean(stack[:, ch, s].astype(np.float64), trim_fraction)
                )
        return template
