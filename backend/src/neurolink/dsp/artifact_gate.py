"""Stage 3 — Epoch-level artifact gate.

Sits in the pipeline after Stage 2 (bad channel detection + spherical
spline interpolation) and before band-power extraction.

Three independent detection passes, each configurable:

1. Amplitude threshold (device-aware)
   Any channel whose peak-to-peak amplitude exceeds ``pk2pk_uv``
   flags the entire frame as contaminated.  When ``electrode_type`` is
   not set explicitly in GateConfig, it is resolved at construction time
   from the active adapter via adapter_factory so dry-electrode devices
   (Muse) automatically receive a tighter threshold (75 µV) than
   wet-electrode systems (100 µV).

2. IMU motion gate
   Accelerometer AC-RMS (after subtracting per-axis mean to remove
   gravity) > ``accel_rms_g`` → frame flagged as motion-contaminated.
   Using AC-RMS prevents the ~1 g steady-state gravity on the Z axis
   of a stationary device from triggering false rejections.

3. Kurtosis burst detection
   Excess kurtosis > ``kurtosis_threshold`` (default 5) → EMG burst
   or electrode-pop contamination.

Threshold defaults
------------------
All numeric defaults are sourced from
``neurolink.dsp.artifact_config`` so every module in the pipeline
shares the same authoritative values.  Runtime overrides are applied
via ``set_config()`` / ``get_config()`` without restarting the pump.

Thread-safety
-------------
All public methods take the config lock before reading ``_cfg``.
``ArtifactGate`` is safe to call from the EEGPump asyncio task while
a REST handler mutates the config.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import structlog
from scipy import stats as sp_stats

from neurolink.dsp.artifact_config import (
    ARTIFACT_ACCEL_RMS_G,
    ARTIFACT_KURTOSIS_THRESHOLD,
    ARTIFACT_PK2PK_UV,
)

log = structlog.get_logger(__name__)


_EEG_IDX: list[int] = [0, 1, 2, 3]
_CH_NAMES = ["TP9", "AF7", "AF8", "TP10"]

ElectrodeType = Literal["dry", "wet", "semi"]

# Per-electrode default amplitude thresholds (µV peak-to-peak).
# Dry electrodes have higher contact impedance and produce more mechanical
# noise, so a tighter gate reduces false-clean frames during sweat transients.
_ELECTRODE_PK2PK_DEFAULTS: dict[str, float] = {
    "dry": 75.0,  # Muse-class wearables — dry gel or foam contact
    "semi": 90.0,  # hybrid (saline-tip); intermediate noise floor
    "wet": ARTIFACT_PK2PK_UV,  # 100 µV — traditional gel; EEGLAB default
}


def _default_pk2pk_for_electrode_type(electrode_type: ElectrodeType) -> float:
    return _ELECTRODE_PK2PK_DEFAULTS.get(electrode_type, ARTIFACT_PK2PK_UV)


def _detect_electrode_type() -> ElectrodeType:
    """Infer electrode type from the active adapter via adapter_factory.
    Falls back to 'dry' — safe default for Muse-class hardware.
    """
    try:
        from neurolink.adapter_factory import create_adapter
        from neurolink.config import get_settings

        settings = get_settings()
        adapter = create_adapter(
            adapter_type=settings.adapter_type,
            device_model=settings.device_model,
        )
        electrode_type: ElectrodeType = getattr(adapter, "electrode_type", "dry")
        if electrode_type not in _ELECTRODE_PK2PK_DEFAULTS:
            log.warning("unknown_electrode_type_fallback", electrode_type=electrode_type)
            return "dry"
        return electrode_type
    except Exception as exc:
        log.warning("electrode_type_detection_failed", error=str(exc))
        return "dry"


@dataclass
class GateConfig:
    """Tunable thresholds for ArtifactGate.

    Parameters
    ----------
    electrode_type:
        'dry' | 'wet' | 'semi'.  When None (default), resolved from
        adapter_factory at construction time so bare ``GateConfig()``
        automatically applies the correct device-specific pk2pk threshold.
        Muse S / Muse S Athena → 'dry' → 75 µV.
    pk2pk_uv:
        Peak-to-peak amplitude limit (µV).  When None (default), derived
        from ``electrode_type``.  Pass explicit value to override at runtime.
    accel_rms_g:
        IMU AC-RMS gate (g).  Applied to the AC component of accelerometer
        data (after per-axis mean subtraction) so gravity does not trigger
        false rejections.
    kurtosis_threshold:
        Excess-kurtosis burst detection threshold (Fisher convention).
    """

    electrode_type: ElectrodeType | None = None
    pk2pk_uv: float | None = None
    accel_rms_g: float = ARTIFACT_ACCEL_RMS_G
    kurtosis_threshold: float = ARTIFACT_KURTOSIS_THRESHOLD
    enable_amplitude: bool = True
    enable_imu: bool = True
    enable_kurtosis: bool = True

    def __post_init__(self) -> None:
        if self.electrode_type is None:
            self.electrode_type = _detect_electrode_type()
        if self.pk2pk_uv is None:
            self.pk2pk_uv = _default_pk2pk_for_electrode_type(
                self.electrode_type
            )
        log.debug(
            "gate_config_resolved",
            electrode_type=self.electrode_type,
            pk2pk_uv=self.pk2pk_uv,
        )


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
    decision = gate.evaluate(eeg_arr, accel_arr)
    if decision.clean:
        bands = compute_band_powers_from_buffer(eeg_arr)
    """

    def __init__(self, config: GateConfig | None = None) -> None:
        self._lock = threading.Lock()
        self._cfg: GateConfig = config or GateConfig()
        self._total_frames: int = 0
        self._rejected_frames: int = 0

    # ── Public API ────────────────────────────────────────────────────────────────────────────

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
            accel: (3, n_accel_samples) or (n_accel_samples,) array in g.
                   Pass None to skip IMU gate.

        Returns:
            ArtifactDecision with reject flag and list of reasons.
        """
        # Snapshot config atomically — a concurrent set_config() on another
        # thread must not split a pk2pk_uv read across two configs.
        with self._lock:
            cfg = self._cfg

        decision = ArtifactDecision()

        if eeg is None or eeg.ndim != 2 or eeg.shape[1] < 2:
            return decision

        n_ch = eeg.shape[0]
        eeg_idx = [i for i in _EEG_IDX if i < n_ch]

        # 1. Amplitude threshold — device-aware via cfg.pk2pk_uv
        if cfg.enable_amplitude and eeg_idx:
            eeg_f64 = eeg[eeg_idx].astype(np.float64)
            pk2pk = np.ptp(eeg_f64, axis=1)  # per-channel range
            bad_mask = pk2pk > cfg.pk2pk_uv
            if bad_mask.any():
                bad_names = [_CH_NAMES[i] for i in np.where(bad_mask)[0]]
                decision.add_reason(
                    f"amplitude>{cfg.pk2pk_uv:.0f}uV ch={bad_names} electrode={cfg.electrode_type}"
                )
                log.debug(
                    "stage3_amplitude_reject",
                    channels=bad_names,
                    pk2pk=pk2pk[bad_mask].tolist(),
                    threshold_uv=cfg.pk2pk_uv,
                    electrode_type=cfg.electrode_type,
                )

        # 2. IMU motion gate — use AC-RMS (subtract per-axis mean) so that
        #    steady-state gravity (~1 g on Z axis) does not trigger rejection.
        #    Only genuine dynamic motion (shaking, nodding) exceeds the threshold.
        if cfg.enable_imu and accel is not None:
            accel_arr = np.asarray(accel, dtype=np.float64)
            if accel_arr.ndim == 1:
                accel_arr = accel_arr[np.newaxis, :]
            # Subtract per-axis mean to remove DC / gravity component.
            accel_ac = accel_arr - accel_arr.mean(axis=1, keepdims=True)
            rms = float(np.sqrt(np.mean(accel_ac**2)))
            if rms > cfg.accel_rms_g:
                decision.add_reason(f"imu_rms_ac={rms:.3f}g>{cfg.accel_rms_g}g")
                log.debug("stage3_imu_reject", rms_ac_g=rms)

        # 3. Kurtosis burst detection
        if cfg.enable_kurtosis and eeg_idx:
            eeg_f64 = eeg[eeg_idx].astype(np.float64)
            for i, ch_idx in enumerate(eeg_idx):
                kurt = float(sp_stats.kurtosis(eeg_f64[i], fisher=True))
                if kurt > cfg.kurtosis_threshold:
                    ch_name = _CH_NAMES[ch_idx]
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
        log.info(
            "stage3_config_updated",
            electrode_type=config.electrode_type,
            pk2pk_uv=config.pk2pk_uv,
        )
