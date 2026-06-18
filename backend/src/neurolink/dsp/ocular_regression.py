"""Stage 5 — Gratton-Coles ocular artifact regression.

When an EOG / AUX reference channel is available (e.g. the Muse AUX
jack positioned near the eye, or a dedicated VEOG channel on research
headsets) this stage subtracts the eye-blink and eye-movement component
from every EEG channel using per-channel OLS slope coefficients.

Why regression over ICA for low-channel EEG
--------------------------------------------
ICA requires a minimum of ~64 channels for reliable source separation.
Muse-class hardware has 4 EEG channels; at that density ICA frequently
misidentifies neural components as artifacts and vice-versa.  The
Gratton-Coles regression is:
  - Computationally trivial (one dot-product per channel per frame)
  - Deterministic and interpretable
  - Validated at low channel counts
  - Applicable in real-time without a calibration window

Reference
---------
Gratton G, Coles MG, Donchin E (1983). "A new method for off-line removal
of ocular artifact". Electroenceph Clin Neurophysiol 55(4):468-484.
Velisar A et al. (2019) review of EOG artifact removal in wearable EEG.

Algorithm
---------
1. Maintain a rolling window of (EEG, EOG) sample pairs.
2. Adaptive recalibration (replaces fixed recalib_frames interval):
     - Every frame, the last 30 EOG samples' variance is compared to a
       rolling 30-frame mean.  When the ratio exceeds 2.0 or falls below
       0.5, early recalibration fires immediately to track electrode
       shifts, sweat-film changes, or a new eye-movement baseline.
     - recalib_frames acts as a fixed fallback floor (max interval).
3. Each frame: EEG_corrected_i = EEG_i - b_i * EOG
   where b_i = cov(EEG_i, EOG) / var(EOG)

Graceful degradation
--------------------
If no EOG channel is present (``eog_channel_idx >= n_channels``) the
corrector returns the array unchanged without error.  This means
existing hardware adapters that lack an AUX channel require no changes.

Public API
----------
  OcularRegressionConfig  — tunable parameters
  OcularRegressor         — stateful corrector; call apply() each tick
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass

import numpy as np
import structlog

log = structlog.get_logger(__name__)


@dataclass
class OcularRegressionConfig:
    """Configuration for the Gratton-Coles ocular regressor.

    Attributes
    ----------
    enable:
        Master switch.  When False the corrector is a no-op.
    eog_channel_idx:
        Index of the EOG / AUX channel in the EEG frame array.
        Default 4 (Muse AUX jack).  Set to -1 or a value >=
        n_channels to disable implicitly.
    eeg_channels:
        Indices of EEG channels to correct.  Default [0,1,2,3].
    calib_window_samples:
        Number of samples to accumulate before the first coefficient
        fit (and to maintain in the rolling window).  Default 1024
        samples ≈ 4 s at 256 Hz — long enough to capture several
        blinks for stable OLS.
    recalib_frames:
        Maximum interval (in EEGPump frames) before a forced
        recalibration.  Default 512 ≈ 2 min at 4 Hz.  Acts as a
        fallback floor; adaptive variance detection fires earlier.
    min_eog_variance:
        Minimum variance of the EOG signal required to attempt
        regression.  Prevents division near zero when the subject is
        motionless and the AUX channel is flat.  Default 0.1 µV².
    """

    enable: bool = True
    eog_channel_idx: int = 4  # Muse AUX jack / VEOG channel
    eeg_channels: list[int] | None = None  # None → [0, 1, 2, 3]
    calib_window_samples: int = 1024
    recalib_frames: int = 512
    min_eog_variance: float = 0.1

    def __post_init__(self) -> None:
        if self.eeg_channels is None:
            object.__setattr__(self, "eeg_channels", [0, 1, 2, 3])


class OcularRegressor:
    """Stateful Gratton-Coles ocular artifact regressor.

    Thread-safety
    -------------
    The coefficient vector and rolling buffer are protected by ``_lock``.
    ``apply()`` is safe to call from the EEGPump asyncio task while a
    REST handler updates the config.

    Usage
    -----
    regressor = OcularRegressor()
    # In EEGPump._build_payload(), after Stage 4 (ASR):
    if not artifact_rejected:
        eeg_arr = regressor.apply(eeg_arr)
    """

    def __init__(self, config: OcularRegressionConfig | None = None) -> None:
        self._lock = threading.Lock()
        self._cfg: OcularRegressionConfig = config or OcularRegressionConfig()
        # Rolling calibration buffer: deques of 1-D arrays
        cfg = self._cfg
        maxlen = cfg.calib_window_samples
        self._eeg_buf: deque[np.ndarray] = deque(maxlen=maxlen)  # each (n_eeg_ch,)
        self._eog_buf: deque[float] = deque(maxlen=maxlen)
        self._slopes: np.ndarray | None = None  # (n_eeg_ch,) coefficients b_i
        self._frames_since_recalib: int = 0
        self._frames_applied: int = 0
        self._frames_passed_through: int = 0
        self._total_frames: int = 0
        self._last_recalib_frame: int = 0
        # Rolling EOG variance history for adaptive recalibration.
        # Stores variance of the last 30 EOG samples, one entry per frame.
        # When current_var / rolling_mean > 2.0 or < 0.5, early recalib fires.
        self._var_history: deque[float] = deque(maxlen=30)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def apply(self, eeg: np.ndarray) -> np.ndarray:
        """Apply ocular regression correction to one EEG frame.

        Args:
            eeg: ndarray (n_channels, n_samples) float32/64.

        Returns:
            Corrected ndarray, same shape and dtype.  Returned unchanged
            if no EOG channel is present or if regression is disabled.
        """
        with self._lock:
            cfg = self._cfg

        if not cfg.enable:
            return eeg

        if eeg.ndim != 2:
            return eeg

        n_ch, n_samples = eeg.shape
        eog_idx = cfg.eog_channel_idx

        # Graceful degradation: no EOG channel available
        if eog_idx < 0 or eog_idx >= n_ch:
            with self._lock:
                self._frames_passed_through += 1
            return eeg

        eeg_idx = [i for i in (cfg.eeg_channels or []) if i < n_ch]
        if not eeg_idx:
            return eeg

        eog_signal = eeg[eog_idx].astype(np.float64)  # (n_samples,)
        eeg_sub = eeg[eeg_idx].astype(np.float64)  # (n_eeg_ch, n_samples)

        # Accumulate per-sample pairs into the rolling buffer
        with self._lock:
            for s in range(n_samples):
                self._eeg_buf.append(eeg_sub[:, s])  # (n_eeg_ch,)
                self._eog_buf.append(float(eog_signal[s]))

            self._frames_since_recalib += 1
            self._total_frames += 1
            # Adaptive variance check: snapshot last 30 EOG samples
            _eog_snap = list(self._eog_buf)[-30:] if len(self._eog_buf) >= 30 else None
            should_recalib = (
                self._variance_triggered(
                    np.array(_eog_snap, dtype=np.float64) if _eog_snap else None,
                    cfg,
                )
                and len(self._eog_buf) >= cfg.calib_window_samples
            )

        if should_recalib:
            self._fit_slopes(cfg)

        with self._lock:
            slopes = self._slopes

        if slopes is None:
            # Not yet enough data for first fit
            with self._lock:
                self._frames_passed_through += 1
            return eeg

        # Apply correction: EEG_i -= slope_i * EOG
        out = eeg.copy().astype(np.float64)
        for i, ch in enumerate(eeg_idx):
            out[ch] -= slopes[i] * eog_signal

        with self._lock:
            self._frames_applied += 1

        return out.astype(eeg.dtype)

    def reset(self) -> None:
        """Clear calibration buffers and coefficients."""
        with self._lock:
            self._eeg_buf.clear()
            self._eog_buf.clear()
            self._slopes = None
            self._frames_since_recalib = 0
            self._frames_applied = 0
            self._frames_passed_through = 0
            self._total_frames = 0
            self._last_recalib_frame = 0
            self._var_history.clear()
        log.info("stage5_ocular_regression_reset")

    def get_stats(self) -> dict:
        with self._lock:
            slopes = self._slopes.tolist() if self._slopes is not None else None
            return {
                "slopes": slopes,
                "calib_buffer_fill": len(self._eog_buf),
                "frames_applied": self._frames_applied,
                "frames_passed_through": self._frames_passed_through,
                "frames_since_recalib": self._frames_since_recalib,
                "total_frames": self._total_frames,
                "last_recalib_frame": self._last_recalib_frame,
                "var_history_len": len(self._var_history),
            }

    def get_config(self) -> OcularRegressionConfig:
        with self._lock:
            import copy

            return copy.copy(self._cfg)

    def set_config(self, config: OcularRegressionConfig) -> None:
        """Replace config and reset calibration buffers."""
        with self._lock:
            self._cfg = config
        self.reset()
        log.info("stage5_ocular_regression_config_updated", config=config)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _variance_triggered(
        self,
        eog_recent: np.ndarray | None,
        cfg: OcularRegressionConfig,
    ) -> bool:
        """Return True if recalibration should fire this frame.

        Trigger conditions (first match wins):
        1. First-ever calibration (no slopes yet).
        2. Adaptive: EOG variance has shifted by >2x or <0.5x rolling mean.
           Indicates electrode shift, sweat film change, or new eye baseline.
        3. Fixed-interval fallback: frames since last recalib >= recalib_frames.
        """
        if self._slopes is None:
            return True

        if eog_recent is not None and len(eog_recent) >= 10:
            current_var = float(np.var(eog_recent))
            self._var_history.append(current_var)
            if len(self._var_history) >= 5:
                rolling_mean = float(np.mean(self._var_history))
                if rolling_mean > 0:
                    ratio = current_var / rolling_mean
                    if ratio > 2.0 or ratio < 0.5:
                        log.debug(
                            "stage5_variance_triggered_recalib",
                            current_var=round(current_var, 3),
                            rolling_mean=round(rolling_mean, 3),
                            ratio=round(ratio, 3),
                            frames_since_last=self._frames_since_recalib,
                        )
                        return True

        return self._frames_since_recalib >= cfg.recalib_frames

    def _fit_slopes(self, cfg: OcularRegressionConfig) -> None:
        """Refit OLS regression coefficients from the rolling buffer."""
        try:
            with self._lock:
                eog_arr = np.array(self._eog_buf, dtype=np.float64)  # (N,)
                eeg_arr = np.array(self._eeg_buf, dtype=np.float64)  # (N, n_eeg_ch)

            eog_var = float(np.var(eog_arr))
            if eog_var < cfg.min_eog_variance:
                log.debug(
                    "stage5_ocular_regression_skip_low_var",
                    eog_var=eog_var,
                    threshold=cfg.min_eog_variance,
                )
                return

            # b_i = cov(EEG_i, EOG) / var(EOG)
            eog_centred = eog_arr - eog_arr.mean()
            eeg_centred = eeg_arr - eeg_arr.mean(axis=0, keepdims=True)
            slopes = (eog_centred @ eeg_centred) / (eog_var * len(eog_arr))  # (n_eeg_ch,)

            with self._lock:
                self._slopes = slopes
                self._frames_since_recalib = 0
                self._last_recalib_frame = self._total_frames

            log.info(
                "stage5_ocular_regression_recalibrated",
                slopes=slopes.tolist(),
                n_samples=len(eog_arr),
                eog_var=round(eog_var, 4),
            )
        except Exception as exc:
            log.error(
                "stage5_ocular_regression_fit_error",
                error=str(exc),
                exc_info=True,
            )
