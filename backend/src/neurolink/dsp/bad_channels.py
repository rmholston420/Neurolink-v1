"""Stage 2 — Bad channel detection.

Detects two classes of bad channels on a running basis:

1. Flat-line  — variance of the channel buffer drops below a threshold
                 (default 0.01 µV²).  Caused by electrode lift-off,
                 broken lead, or fully dried gel.

2. Noisy      — mean broadband PSD of the channel exceeds
                 psd_ratio_threshold × median across all channels
                 (default 5×).  Caused by EMG, cable artefact, or
                 poor skin-electrode contact producing high impedance.

Both detectors use an exponentially-weighted running estimate updated
on every EEGPump tick so the decision adapts within ~20 frames
(~5 s at 4 Hz).

A manual override list lets the REST layer flag / un-flag channels
independently of the automatic decision.  Manual flags take priority.

Channel order follows EEGSample.channels:
    index 0 → TP9
    index 1 → AF7
    index 2 → AF8
    index 3 → TP10
    index 4 → AUX   (non-EEG; excluded from PSD comparison)
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np
import structlog
from scipy import signal as sp_signal

log = structlog.get_logger(__name__)

# Channel names matching EEGSample.channels index order
CHANNEL_NAMES: list[str] = ["TP9", "AF7", "AF8", "TP10", "AUX"]
# EEG-only channels (AUX excluded from PSD ratio comparison)
_EEG_IDX: list[int] = [0, 1, 2, 3]


@dataclass
class DetectorConfig:
    """Tunable thresholds for BadChannelDetector."""

    var_threshold: float = 0.01  # µV²  — below this → flat-line bad
    psd_ratio_threshold: float = 5.0  # ×median PSD → noisy bad
    ema_alpha: float = 0.1  # EMA smoothing (0.1 →20-frame half-life)
    fs: float = 256.0
    nperseg: int = 128


@dataclass
class ChannelStats:
    """Running stats for a single channel."""

    name: str
    ema_variance: float = 0.0
    ema_mean_psd: float = 0.0
    flat_line: bool = False
    noisy: bool = False
    manual_bad: bool = False

    @property
    def is_bad(self) -> bool:
        return self.manual_bad or self.flat_line or self.noisy

    def reason(self) -> str:
        reasons: list[str] = []
        if self.manual_bad:
            reasons.append("manual")
        if self.flat_line:
            reasons.append("flat_line")
        if self.noisy:
            reasons.append("noisy")
        return ",".join(reasons) if reasons else "ok"


class BadChannelDetector:
    """Thread-safe per-session bad channel detector.

    Usage
    -----
    detector.update(eeg_arr)          # called by EEGPump each tick
    bad = detector.get_bad_channels() # returns list[str] of bad names
    eeg_clean = interp(eeg_arr, bad)  # passed to spherical_spline
    """

    def __init__(self, config: DetectorConfig | None = None) -> None:
        self._cfg = config or DetectorConfig()
        self._lock = threading.Lock()
        self._stats: list[ChannelStats] = [ChannelStats(name=n) for n in CHANNEL_NAMES]

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def update(self, eeg: np.ndarray) -> None:
        """Update running statistics from a (n_channels, n_samples) array.

        Skips channels that are manually flagged (still updates EMA so
        that the channel can be auto-cleared if manual flag is removed).

        Args:
            eeg: float32 array of shape (n_channels, n_samples).
                 n_channels may be < 5 if AUX is absent.
        """
        if eeg is None or eeg.ndim != 2 or eeg.shape[1] < 2:
            return

        cfg = self._cfg
        n_ch = min(eeg.shape[0], len(CHANNEL_NAMES))
        alpha = cfg.ema_alpha

        # Compute per-channel variance and mean PSD
        variances: list[float] = []
        mean_psds: list[float] = []
        for ch in range(n_ch):
            variances.append(float(np.var(eeg[ch])))
            _, psd = sp_signal.welch(
                eeg[ch].astype(np.float64),
                fs=cfg.fs,
                nperseg=min(cfg.nperseg, eeg.shape[1]),
            )
            mean_psds.append(float(np.mean(psd)))

        # Median PSD over EEG-only channels present in this buffer
        eeg_psds = [mean_psds[i] for i in _EEG_IDX if i < n_ch]
        median_psd = float(np.median(eeg_psds)) if eeg_psds else 1.0
        if median_psd <= 0:
            median_psd = 1.0

        with self._lock:
            for ch in range(n_ch):
                s = self._stats[ch]
                # EMA update
                s.ema_variance = alpha * variances[ch] + (1 - alpha) * s.ema_variance
                s.ema_mean_psd = alpha * mean_psds[ch] + (1 - alpha) * s.ema_mean_psd
                # Threshold decisions (AUX skipped for PSD ratio check)
                s.flat_line = s.ema_variance < cfg.var_threshold
                if ch in _EEG_IDX:
                    s.noisy = s.ema_mean_psd > cfg.psd_ratio_threshold * median_psd
                else:
                    s.noisy = False  # AUX: only flat-line check

    def get_bad_channels(self) -> list[str]:
        """Return names of all currently bad channels."""
        with self._lock:
            return [s.name for s in self._stats if s.is_bad]

    def get_stats(self) -> list[ChannelStats]:
        """Return a snapshot of all per-channel stats (copy)."""
        with self._lock:
            import copy

            return copy.deepcopy(self._stats)

    def set_manual_bad(self, channel: str, bad: bool) -> None:
        """Manually flag or un-flag a channel by name."""
        with self._lock:
            for s in self._stats:
                if s.name.upper() == channel.upper():
                    s.manual_bad = bad
                    log.info(
                        "stage2_manual_flag",
                        channel=channel,
                        bad=bad,
                    )
                    return
        raise ValueError(f"Unknown channel: {channel!r}")

    def get_config(self) -> DetectorConfig:
        with self._lock:
            import copy

            return copy.copy(self._cfg)

    def set_config(self, config: DetectorConfig) -> None:
        with self._lock:
            self._cfg = config
            log.info("stage2_config_updated", config=config)

    def reset(self) -> None:
        """Reset all running stats (call at session start)."""
        with self._lock:
            for s in self._stats:
                s.ema_variance = 0.0
                s.ema_mean_psd = 0.0
                s.flat_line = False
                s.noisy = False
                s.manual_bad = False
        log.info("stage2_detector_reset")
