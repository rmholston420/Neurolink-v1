"""Stage 4 — Artifact Subspace Reconstruction (ASR).

Reference
---------
Chang C-Y et al. (2020) "Evaluation of Artifact Subspace Reconstruction
for Automatic EEG Artifact Removal", Front. Hum. Neurosci. 14:578482.

Public API
----------
  ASRConfig         — dataclass of tunable parameters
  ArtifactSubspaceReconstructor — stateful corrector; call apply() each tick
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
import structlog

from neurolink.dsp.artifact_config import ASR_BURST_SD, ASR_CALIB_SEC

log = structlog.get_logger(__name__)


class ASRState(Enum):
    CALIBRATING = auto()
    READY = auto()
    DISABLED = auto()


@dataclass
class ASRConfig:
    enable: bool = True
    fs: float = 256.0
    calib_sec: float = ASR_CALIB_SEC
    burst_sd: float = ASR_BURST_SD
    eeg_channels: list[int] = None

    def __post_init__(self) -> None:
        if self.eeg_channels is None:
            object.__setattr__(self, "eeg_channels", [0, 1, 2, 3])


class ArtifactSubspaceReconstructor:
    """Stateful ASR corrector for streaming EEG."""

    def __init__(self, config: ASRConfig | None = None) -> None:
        self._lock = threading.Lock()
        self._cfg: ASRConfig = config or ASRConfig()
        self._state: ASRState = (
            ASRState.DISABLED if not self._cfg.enable else ASRState.CALIBRATING
        )
        self._calib_frames: list[np.ndarray] = []
        self._calib_samples_needed: int = int(
            self._cfg.calib_sec * self._cfg.fs
        )
        self._calib_samples_collected: int = 0
        self._M: np.ndarray | None = None
        self._M_inv: np.ndarray | None = None
        self._calib_rms: float = 1.0
        self._frames_processed: int = 0
        self._frames_corrected: int = 0
        self._samples_reconstructed: int = 0

    def apply(self, eeg: np.ndarray) -> np.ndarray:
        with self._lock:
            cfg = self._cfg
            state = self._state

        if state == ASRState.DISABLED:
            return eeg

        if eeg.ndim != 2 or eeg.shape[1] < 2:
            return eeg

        eeg_idx = [i for i in cfg.eeg_channels if i < eeg.shape[0]]
        if not eeg_idx:
            return eeg

        if state == ASRState.CALIBRATING:
            return self._accumulate_calibration(eeg, eeg_idx, cfg)

        return self._reconstruct(eeg, eeg_idx, cfg)

    def reset(self) -> None:
        with self._lock:
            self._state = (
                ASRState.DISABLED if not self._cfg.enable else ASRState.CALIBRATING
            )
            self._calib_frames = []
            self._calib_samples_collected = 0
            self._M = None
            self._M_inv = None
            self._calib_rms = 1.0
            self._frames_processed = 0
            self._frames_corrected = 0
            self._samples_reconstructed = 0
        log.info("stage4_asr_reset")

    def get_state(self) -> str:
        with self._lock:
            return self._state.name

    def get_stats(self) -> dict:
        with self._lock:
            return {
                "state": self._state.name,
                "calib_samples_collected": self._calib_samples_collected,
                "calib_samples_needed": self._calib_samples_needed,
                "frames_processed": self._frames_processed,
                "frames_corrected": self._frames_corrected,
                "samples_reconstructed": self._samples_reconstructed,
                "calib_rms": round(self._calib_rms, 6),
            }

    def get_config(self) -> ASRConfig:
        with self._lock:
            import copy
            return copy.copy(self._cfg)

    def set_config(self, config: ASRConfig) -> None:
        with self._lock:
            self._cfg = config
        self.reset()
        log.info("stage4_asr_config_updated", config=config)

    def _accumulate_calibration(
        self,
        eeg: np.ndarray,
        eeg_idx: list[int],
        cfg: ASRConfig,
    ) -> np.ndarray:
        sub = eeg[eeg_idx].astype(np.float64)
        with self._lock:
            self._calib_frames.append(sub)
            self._calib_samples_collected += sub.shape[1]
            if self._calib_samples_collected >= self._calib_samples_needed:
                self._fit_model(cfg)
        return eeg

    def _fit_model(self, cfg: ASRConfig) -> None:
        """Fit whitening matrix from the calibration buffer.

        Must be called while ``_lock`` is held.
        Resets _frames_processed so the counter reflects only post-calibration
        reconstruction calls (not the calibration accumulation frames).
        """
        try:
            data = np.concatenate(self._calib_frames, axis=1)
            data -= data.mean(axis=1, keepdims=True)
            C = np.cov(data)
            if C.ndim == 0:
                C = np.array([[float(C)]])
            try:
                M = np.linalg.cholesky(C + np.eye(C.shape[0]) * 1e-8)
            except np.linalg.LinAlgError:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-8)
                M = eigvecs @ np.diag(np.sqrt(eigvals))
            M_inv = np.linalg.pinv(M)
            whitened = M_inv @ data
            sample_rms = np.sqrt(np.mean(whitened ** 2, axis=0))
            calib_rms = float(np.median(sample_rms))
            self._M = M
            self._M_inv = M_inv
            self._calib_rms = max(calib_rms, 1e-9)
            self._state = ASRState.READY
            self._calib_frames = []
            # Reset post-calibration counters so tests get a clean baseline
            self._frames_processed = 0
            self._frames_corrected = 0
            self._samples_reconstructed = 0
            log.info(
                "stage4_asr_calibrated",
                n_samples=data.shape[1],
                calib_rms=round(calib_rms, 6),
                n_channels=data.shape[0],
            )
        except Exception as exc:
            log.error("stage4_asr_calibration_failed", error=str(exc), exc_info=True)
            self._calib_frames = []
            self._calib_samples_collected = 0

    def _reconstruct(
        self,
        eeg: np.ndarray,
        eeg_idx: list[int],
        cfg: ASRConfig,
    ) -> np.ndarray:
        with self._lock:
            M = self._M
            M_inv = self._M_inv
            calib_rms = self._calib_rms

        if M is None or M_inv is None:
            return eeg

        out = eeg.copy().astype(np.float64)
        sub = out[eeg_idx]
        sub -= sub.mean(axis=1, keepdims=True)

        whitened = M_inv @ sub
        sample_rms = np.sqrt(np.mean(whitened ** 2, axis=0))
        burst_mask = sample_rms > (cfg.burst_sd * calib_rms)
        n_burst = int(burst_mask.sum())

        if n_burst > 0:
            clean_proj = M @ whitened
            sub[:, burst_mask] = clean_proj[:, burst_mask]
            out[eeg_idx] = sub

        with self._lock:
            self._frames_processed += 1
            if n_burst > 0:
                self._frames_corrected += 1
                self._samples_reconstructed += n_burst

        return out.astype(eeg.dtype)
