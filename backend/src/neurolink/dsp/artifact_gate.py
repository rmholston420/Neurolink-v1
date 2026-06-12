"""Stage 3 — Epoch-level artifact gate.

Sits in the pipeline after Stage 2 (bad channel detection + spherical
spline interpolation) and before band-power extraction.

Three independent detection passes, each configurable:

1. Amplitude threshold
   Any channel whose peak-to-peak amplitude exceeds ``pk2pk_uv``
   (default 100 µV) flags the entire frame as contaminated.
   Rationale: Lindsley 1944 / EEGLAB reject_threshold convention;
   validated for Muse-class dry-electrode wearables.

2. IMU motion gate
   When an accelerometer RMS (across all axes, in *g*) exceeds
   ``accel_rms_g`` (default 0.15 g) the frame is flagged as motion-
   contaminated.  IMU-gated rejection is the recommended strategy for
   wearable EEG (Lopes da Silva 2024; Blum et al. 2019).

3. Kurtosis burst detection
   Excess kurtosis > ``kurtosis_threshold`` (default 5) across any EEG
   channel flags high-kurtosis bursts (muscle EMG / electrode pop).
   Kurtosis is computed on the raw float64 channel vector; the
   scipy.stats excess-kurtosis convention (Fisher, default) is used.

The gate is *non-destructive*: it never modifies the EEG array.
Instead it returns an ``ArtifactDecision`` that callers use to decide
whether to forward the frame to downstream DSP.

Threshold defaults
------------------
All numeric defaults are sourced from
``neurolink.dsp.artifact_config`` so every module in the pipeline
shares the same authoritative values.  Runtime overrides are applied
via ``set_config()`` / ``get_config()`` without restarting the pump,
enabling per-session adaptive tightening.

Thread-safety
-------------
All public methods take the config lock before reading ``_cfg``.
``ArtifactGate`` is safe to call from the EEGPump asyncio task while
a REST handler mutates the config.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import structlog
from scipy import stats as sp_stats

from neurolink.dsp.artifact_config import (
    ARTIFACT_ACCEL_RMS_G,
    ARTIFACT_KURTOSIS_THRESHOLD,
    ARTIFACT_PK2PK_UV,
)

log = structlog.get_logger(__name__)

# EEG-only channel indices (AUX excluded)
_EEG_IDX: list[int] = [0, 1, 2, 3]


@dataclass
class GateConfig:
    """Tunable thresholds for ArtifactGate.

    Defaults are sourced from ``neurolink.dsp.artifact_config`` so all
    pipeline stages share the same authoritative baseline values.
    """

    pk2pk_uv: float = ARTIFACT_PK2PK_UV          # µV  — amplitude threshold
    accel_rms_g: float = ARTIFACT_ACCEL_RMS_G    # g   — IMU motion threshold
    kurtosis_threshold: float = ARTIFACT_KURTOSIS_THRESHOLD  # burst threshold
    enable_amplitude: bool = True
    enable_imu: bool = True
    enable_kurtosis: bool = True


@dataclass
class ArtifactDecision:
    """Result of one gate evaluation."""

    reject: bool = False
    reasons: list[str] = field(default_factory=list)

    def add_reason(self, reason: str) -> None:
        self.reasons.append(reason)
        self.reject = True

    @property
    def clean(self) -> bool:
        return not self.reject


class ArtifactGate:
    """Stateless per-frame artifact gate.

    Usage
    -----
    gate = ArtifactGate()
    decision = gate.evaluate(eeg_arr, accel_arr)  # call each pump tick
    if decision.clean:
        bands = compute_band_powers_from_buffer(eeg_arr)
    """

    def __init__(self, config: GateConfig | None = None) -> None:
        self._lock = threading.Lock()
        self._cfg: GateConfig = config or GateConfig()
        self._total_frames: int = 0
        self._rejected_frames: int = 0

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def evaluate(
        self,
        eeg: np.ndarray,
        accel: np.ndarray | None = None,
    ) -> ArtifactDecision:
        """Evaluate one frame for artifacts.

        Args:
            eeg:   (n_channels, n_samples) float array.  Only the first
                   4 channels (TP9, AF7, AF8, TP10) are evaluated;
                   AUX (index 4) is ignored.
            accel: (3, n_accel_samples) or (n_accel_samples,) float
                   array of accelerometer readings in *g*.  Pass None
                   to skip IMU gate.

        Returns:
            ArtifactDecision with reject flag and list of reasons.
        """
        with self._lock:
            cfg = self._cfg

        decision = ArtifactDecision()

        if eeg is None or eeg.ndim != 2 or eeg.shape[1] < 2:
            return decision

        n_ch = eeg.shape[0]
        eeg_idx = [i for i in _EEG_IDX if i < n_ch]

        # 1. Amplitude threshold
        if cfg.enable_amplitude and eeg_idx:
            eeg_f64 = eeg[eeg_idx].astype(np.float64)
            pk2pk = np.ptp(eeg_f64, axis=1)  # per-channel range
            bad_mask = pk2pk > cfg.pk2pk_uv
            if bad_mask.any():
                bad_names = [
                    ["TP9", "AF7", "AF8", "TP10"][i] for i in np.where(bad_mask)[0]
                ]
                decision.add_reason(f"amplitude>{cfg.pk2pk_uv}uV ch={bad_names}")
                log.debug(
                    "stage3_amplitude_reject",
                    channels=bad_names,
                    pk2pk=pk2pk[bad_mask].tolist(),
                )

        # 2. IMU motion gate
        if cfg.enable_imu and accel is not None:
            accel_arr = np.asarray(accel, dtype=np.float64)
            if accel_arr.ndim == 1:
                accel_arr = accel_arr[np.newaxis, :]
            rms = float(np.sqrt(np.mean(accel_arr ** 2)))
            if rms > cfg.accel_rms_g:
                decision.add_reason(f"imu_rms={rms:.3f}g>{cfg.accel_rms_g}g")
                log.debug("stage3_imu_reject", rms_g=rms)

        # 3. Kurtosis burst detection
        if cfg.enable_kurtosis and eeg_idx:
            eeg_f64 = eeg[eeg_idx].astype(np.float64)
            for i, ch_idx in enumerate(eeg_idx):
                kurt = float(sp_stats.kurtosis(eeg_f64[i], fisher=True))
                if kurt > cfg.kurtosis_threshold:
                    ch_name = ["TP9", "AF7", "AF8", "TP10"][ch_idx]
                    decision.add_reason(
                        f"kurtosis={kurt:.1f}>{cfg.kurtosis_threshold} ch={ch_name}"
                    )
                    log.debug(
                        "stage3_kurtosis_reject",
                        channel=ch_name,
                        kurtosis=kurt,
                    )

        with self._lock:
            self._total_frames += 1
            if decision.reject:
                self._rejected_frames += 1

        return decision

    def get_stats(self) -> dict:
        """Return running frame counters and rejection rate."""
        with self._lock:
            total = self._total_frames
            rejected = self._rejected_frames
        rate = rejected / total if total else 0.0
        return {
            "total_frames": total,
            "rejected_frames": rejected,
            "rejection_rate": round(rate, 4),
        }

    def reset_stats(self) -> None:
        """Reset frame counters (call at session start)."""
        with self._lock:
            self._total_frames = 0
            self._rejected_frames = 0
        log.info("stage3_stats_reset")

    def get_config(self) -> GateConfig:
        with self._lock:
            import copy
            return copy.copy(self._cfg)

    def set_config(self, config: GateConfig) -> None:
        with self._lock:
            self._cfg = config
        log.info("stage3_config_updated", config=config)
