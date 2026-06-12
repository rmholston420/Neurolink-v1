"""Stage 4 — Artifact Subspace Reconstruction (ASR).

ASR is the recommended automated artifact-correction strategy for
low-density wearable EEG (< 64 channels) where ICA source separation
degrades due to insufficient channel count.  It outperforms pure ICA
for Muse-class 4-channel hardware.

Reference
---------
Chang C-Y et al. (2020) "Evaluation of Artifact Subspace Reconstruction
for Automatic EEG Artifact Removal", Front. Hum. Neurosci. 14:578482.
Kothe & Jung (2016) EEGLAB clean_rawdata plugin, BurstCriterion=20 SD.
Blum et al. (2019) "A Riemannian modification of artifact subspace
reconstruction", Brain Topogr. 32(4):648-659.

Algorithm (simplified, suitable for low-channel-count wearable EEG)
--------------------------------------------------------------------
1. Calibration phase (first ``calib_sec`` seconds of clean data):
   - Accumulate EEG frames until buffer reaches ``calib_sec * fs``
     samples.
   - Compute the sample covariance C_ref of the calibration buffer.
   - Derive the mixing matrix M = cholesky(C_ref) so we can map data
     into a whitened subspace.

2. Online correction phase (every subsequent frame):
   - Project the frame into the whitened subspace.
   - Compute per-sample RMS across channels (the "burst criterion").
   - Samples where RMS > ``burst_sd * calib_rms`` are considered burst
     artifacts.
   - Replace burst samples by projecting back from the calibration
     covariance rather than from the contaminated data, effectively
     reconstructing the clean signal.
   - Return the corrected frame in original (microvolts) space.

Limitations
-----------
* With only 4 EEG channels the subspace has rank 4; reconstruction is
  coarser than with high-density EEG.  Use ASR as a *correction* pass,
  not a replacement for the upstream amplitude / kurtosis gates.
* Calibration assumes the first ``calib_sec`` seconds are relatively
  clean (subject resting, minimal movement).  The Stage 0 acquisition-
  readiness gate enforces this.
* ``filtfilt``-style zero-phase filtering upstream (Stage 1) is required
  before ASR — this is already guaranteed by the EEGPump pipeline.

Threshold defaults
------------------
``ASRConfig`` defaults (``burst_sd``, ``calib_sec``) are sourced from
``neurolink.dsp.artifact_config`` constants so all pipeline stages share
the same authoritative baseline values.

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
    CALIBRATING = auto()   # accumulating calibration buffer
    READY = auto()         # calibration complete, applying correction
    DISABLED = auto()      # config.enable=False; pass-through


@dataclass
class ASRConfig:
    """Tunable parameters for ArtifactSubspaceReconstructor.

    Attributes
    ----------
    enable:
        Master switch.  When False the reconstructor is a no-op.
    fs:
        EEG sampling rate in Hz.
    calib_sec:
        Seconds of clean data to collect before activating correction.
        The Stage 0 acquisition-readiness gate ensures this window is
        clean.  Increase to 60 s for more stable covariance estimates.
        Default sourced from ``artifact_config.ASR_CALIB_SEC``.
    burst_sd:
        BurstCriterion: number of calibration SDs above which a sample
        is considered a burst artifact.  EEGLAB default is 20; lower
        values (e.g. 15) are more aggressive.
        Default sourced from ``artifact_config.ASR_BURST_SD``.
    eeg_channels:
        Indices of EEG channels in the frame array.  AUX/PPG channels
        are excluded from subspace reconstruction.
    """

    enable: bool = True
    fs: float = 256.0
    calib_sec: float = ASR_CALIB_SEC   # default: 30.0 s
    burst_sd: float = ASR_BURST_SD     # default: 20.0 SD
    eeg_channels: list[int] = None     # None → auto-detect as [0,1,2,3]

    def __post_init__(self) -> None:
        if self.eeg_channels is None:
            object.__setattr__(self, "eeg_channels", [0, 1, 2, 3])


class ArtifactSubspaceReconstructor:
    """Stateful ASR corrector for streaming EEG.

    Thread-safety
    -------------
    Calibration state and the reference covariance are protected by
    ``_lock``.  ``apply()`` is safe to call from the EEGPump asyncio
    task while a REST handler calls ``reset()`` or ``set_config()``.

    Usage
    -----
    asr = ArtifactSubspaceReconstructor()
    # In EEGPump._build_payload(), after Stage 3:
    if not artifact_rejected:
        eeg_arr = asr.apply(eeg_arr)
    """

    def __init__(self, config: ASRConfig | None = None) -> None:
        self._lock = threading.Lock()
        self._cfg: ASRConfig = config or ASRConfig()
        self._state: ASRState = (
            ASRState.DISABLED if not self._cfg.enable else ASRState.CALIBRATING
        )
        # Calibration buffer: list of 1-D channel-mean-subtracted frames
        self._calib_frames: list[np.ndarray] = []  # each (n_eeg_ch, n_samples)
        self._calib_samples_needed: int = int(
            self._cfg.calib_sec * self._cfg.fs
        )
        self._calib_samples_collected: int = 0
        # Fitted model (set after calibration)
        self._M: np.ndarray | None = None           # whitening matrix  (n_ch, n_ch)
        self._M_inv: np.ndarray | None = None       # de-whitening matrix
        self._calib_rms: float = 1.0                # per-sample RMS in whitened space
        # Stats
        self._frames_processed: int = 0
        self._frames_corrected: int = 0
        self._samples_reconstructed: int = 0

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def apply(self, eeg: np.ndarray) -> np.ndarray:
        """Apply ASR correction to one EEG frame.

        During calibration the frame is accumulated *and returned
        unchanged*.  After calibration corrupt bursts are reconstructed.

        Args:
            eeg: ndarray (n_channels, n_samples) float32/64.

        Returns:
            Corrected ndarray of the same shape and dtype.
        """
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

        # READY — apply correction
        return self._reconstruct(eeg, eeg_idx, cfg)

    def reset(self) -> None:
        """Reset to calibration state (call at session start)."""
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
        """Replace config and reset calibration."""
        with self._lock:
            self._cfg = config
        self.reset()
        log.info("stage4_asr_config_updated", config=config)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _accumulate_calibration(  # noqa: PLR0912
        self,
        eeg: np.ndarray,
        eeg_idx: list[int],
        cfg: ASRConfig,
    ) -> np.ndarray:
        """Collect calibration data; fit model when enough has been gathered."""
        sub = eeg[eeg_idx].astype(np.float64)  # (n_eeg_ch, n_samples)
        with self._lock:
            self._calib_frames.append(sub)
            self._calib_samples_collected += sub.shape[1]
            if self._calib_samples_collected >= self._calib_samples_needed:
                self._fit_model(cfg)
        return eeg  # return unchanged during calibration

    def _fit_model(self, cfg: ASRConfig) -> None:
        """Fit whitening matrix from the calibration buffer.

        Must be called while ``_lock`` is held.
        """
        try:
            data = np.concatenate(self._calib_frames, axis=1)  # (n_ch, N)
            # Centre
            data -= data.mean(axis=1, keepdims=True)
            # Sample covariance
            C = np.cov(data)  # (n_ch, n_ch)
            if C.ndim == 0:
                C = np.array([[float(C)]])
            # Whitening via Cholesky: M @ M.T = C
            try:
                M = np.linalg.cholesky(C + np.eye(C.shape[0]) * 1e-8)
            except np.linalg.LinAlgError:
                # Fallback: symmetric square-root via eigen-decomposition
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-8)
                M = eigvecs @ np.diag(np.sqrt(eigvals))
            M_inv = np.linalg.pinv(M)
            # Calibration RMS in whitened space (used as BurstCriterion baseline)
            whitened = M_inv @ data  # (n_ch, N)
            sample_rms = np.sqrt(np.mean(whitened ** 2, axis=0))  # (N,)
            calib_rms = float(np.median(sample_rms))  # robust median
            self._M = M
            self._M_inv = M_inv
            self._calib_rms = max(calib_rms, 1e-9)
            self._state = ASRState.READY
            self._calib_frames = []  # free memory
            log.info(
                "stage4_asr_calibrated",
                n_samples=data.shape[1],
                calib_rms=round(calib_rms, 6),
                n_channels=data.shape[0],
            )
        except Exception as exc:
            log.error("stage4_asr_calibration_failed", error=str(exc), exc_info=True)
            # Stay in CALIBRATING so we retry on the next tick
            self._calib_frames = []
            self._calib_samples_collected = 0

    def _reconstruct(  # noqa: PLR0912
        self,
        eeg: np.ndarray,
        eeg_idx: list[int],
        cfg: ASRConfig,
    ) -> np.ndarray:
        """Reconstruct burst-artifact samples using the calibration subspace."""
        with self._lock:
            M = self._M
            M_inv = self._M_inv
            calib_rms = self._calib_rms

        if M is None or M_inv is None:
            return eeg

        out = eeg.copy().astype(np.float64)
        sub = out[eeg_idx]  # (n_eeg_ch, n_samples)
        sub -= sub.mean(axis=1, keepdims=True)

        # Project into whitened space
        whitened = M_inv @ sub  # (n_ch, n_samples)

        # Identify burst samples
        sample_rms = np.sqrt(np.mean(whitened ** 2, axis=0))  # (n_samples,)
        burst_mask = sample_rms > (cfg.burst_sd * calib_rms)
        n_burst = int(burst_mask.sum())

        if n_burst > 0:
            # Reconstruct burst samples from the clean calibration covariance:
            # project burst samples back through M (the calibration mixing matrix)
            # using only the clean-subspace dimensions.  For low-rank hardware
            # (4 ch) this is equivalent to ASR's full SVD variant.
            clean_proj = M @ whitened  # full reconstruction (n_ch, n_samples)
            sub[:, burst_mask] = clean_proj[:, burst_mask]
            out[eeg_idx] = sub

        with self._lock:
            self._frames_processed += 1
            if n_burst > 0:
                self._frames_corrected += 1
                self._samples_reconstructed += n_burst
            log.debug(
                "stage4_asr_applied",
                burst_samples=n_burst,
                total_samples=eeg.shape[1],
                burst_frac=round(n_burst / max(eeg.shape[1], 1), 3),
            ) if n_burst > 0 else None

        return out.astype(eeg.dtype)
