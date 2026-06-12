"""Stage 7 — fNIRS (functional near-infrared spectroscopy) preprocessing.

Overview
--------
Muse-compatible fNIRS accessories deliver raw optical-density (OD) data at
two wavelengths (~760 nm and ~850 nm) across 4–8 source-detector pairs.
This module provides:

  apply(raw)   — preprocessing pipeline:
                   1. Spike / motion artifact clipping
                   2. Exponential-weighted baseline detrending (DC removal)
                   3. Returns float32 copy (never mutates input)

  decode(raw)  — modified Beer–Lambert Law conversion to oxygenated (HbO)
                 and deoxygenated (HbR) haemoglobin concentration changes.
                 Expects channel layout [pair0_760, pair0_850, pair1_760, …].
                 Returns (HbO, HbR) each shape (n_pairs, n_samples).

Graceful degradation
--------------------
When input is None, 1-D, or empty the module returns the input unchanged
(or None) rather than raising.  No exception propagates to the EEGPump
hot-path.

Thread safety
-------------
All mutable state is protected by a single threading.Lock.  apply() and
decode() hold the lock only for the brief state-read/write section; heavy
NumPy work runs outside the lock.
"""

from __future__ import annotations

import copy
import threading
from dataclasses import dataclass

import numpy as np
import structlog

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Beer–Lambert extinction coefficients (cm⁻¹ / mM) at 760 nm and 850 nm
# Source: Matcher et al. 1995 / Cope & Delpy 1988
# ---------------------------------------------------------------------------
_EPS_HBO_760: float = 0.328    # HbO at 760 nm
_EPS_HBO_850: float = 1.590    # HbO at 850 nm
_EPS_HBR_760: float = 3.910    # HbR at 760 nm
_EPS_HBR_850: float = 1.433    # HbR at 850 nm

# Determinant of the 2×2 extinction matrix (pre-computed for speed)
_BL_DET: float = (_EPS_HBO_760 * _EPS_HBR_850) - (_EPS_HBR_760 * _EPS_HBO_850)

# Differential path-length factor (typical adult head, unitless)
_DPF: float = 6.0


@dataclass
class FNIRSConfig:
    """Tunable parameters for the fNIRS preprocessing module.

    Attributes
    ----------
    enable:
        Master switch.  False → apply() returns input unchanged (identity).
    baseline_alpha:
        Exponential smoothing factor for the per-channel baseline tracker.
        Higher = faster adaptation.  Range (0, 1).  Default 0.01.
    spike_threshold:
        Samples whose absolute value exceeds this multiple of the per-channel
        running standard deviation are clipped to ±threshold × σ.
        Default 5.0 (5 σ clip).
    min_channels:
        Minimum number of channels required to attempt Beer–Lambert decode.
        Inputs with fewer channels return (zeros, zeros).  Default 2.
    """

    enable: bool = True
    baseline_alpha: float = 0.01
    spike_threshold: float = 5.0
    min_channels: int = 2


# ---------------------------------------------------------------------------
# Module-level singleton state
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_config: FNIRSConfig = FNIRSConfig()

# Per-channel exponential baseline (updated each frame)
# Shape: (n_channels,) — initialised lazily on first apply() call.
_baseline: np.ndarray | None = None

# Per-channel running variance for spike detection (Welford online)
_running_mean: np.ndarray | None = None
_running_m2: np.ndarray | None = None
_n_frames: int = 0


# ---------------------------------------------------------------------------
# Public API — config
# ---------------------------------------------------------------------------

def get_config() -> FNIRSConfig:
    """Return a shallow copy of the current config (thread-safe)."""
    with _lock:
        return copy.copy(_config)


def set_config(**kwargs) -> FNIRSConfig:
    """Update one or more config fields and return the new config.

    Unknown keyword arguments are silently ignored so partial updates
    are safe.

    Parameters
    ----------
    **kwargs:
        Field names and values from FNIRSConfig.
    """
    global _config
    valid = {f.name for f in _config.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    with _lock:
        current = copy.copy(_config)
        for k, v in kwargs.items():
            if k in valid:
                setattr(current, k, v)
        _config = current
        return copy.copy(_config)


def reset() -> None:
    """Reset all accumulated per-channel state.

    Call at session start, reconnect, or after a headset swap.
    Config is preserved.
    """
    global _baseline, _running_mean, _running_m2, _n_frames
    with _lock:
        _baseline = None
        _running_mean = None
        _running_m2 = None
        _n_frames = 0
    log.info("fnirs_reset")


# ---------------------------------------------------------------------------
# Public API — pipeline
# ---------------------------------------------------------------------------

def apply(raw: np.ndarray | None) -> np.ndarray | None:
    """Preprocess one fNIRS frame.

    Steps
    -----
    1. Guard: None / non-2D / zero-channel input returned unchanged.
    2. Spike clip: samples > spike_threshold × σ clipped in-place on copy.
    3. Exponential baseline detrend: subtract per-channel EMA.
    4. Update running statistics for next frame.

    Parameters
    ----------
    raw:
        (n_channels, n_samples) float array of raw optical-density data.

    Returns
    -------
    Processed float32 ndarray of the same shape, or the original object
    unchanged if processing is skipped.
    """
    if raw is None:
        return None

    with _lock:
        cfg = copy.copy(_config)

    if not cfg.enable:
        return raw

    if not isinstance(raw, np.ndarray) or raw.ndim != 2 or raw.shape[0] == 0:
        return raw

    n_ch, n_samples = raw.shape
    out = raw.astype(np.float32, copy=True)

    # ── Spike clip ────────────────────────────────────────────────────────
    with _lock:
        rm = _running_mean
        rm2 = _running_m2
        nf = _n_frames

    if rm is not None and rm2 is not None and nf > 1:
        sigma = np.sqrt(np.maximum(rm2 / nf, 1e-8)).astype(np.float32)  # (n_ch,)
        threshold = (cfg.spike_threshold * sigma).reshape(n_ch, 1)
        # Centre around current mean before clipping
        mu = rm.astype(np.float32).reshape(n_ch, 1)
        centred = out - mu
        centred = np.clip(centred, -threshold, threshold)
        out = centred + mu

    # ── Baseline detrend ──────────────────────────────────────────────────
    with _lock:
        bl = _baseline

    if bl is None:
        bl = out.mean(axis=1).copy()  # initialise from first frame
    else:
        bl = bl.astype(np.float32)

    frame_mean = out.mean(axis=1)  # (n_ch,)
    alpha = float(cfg.baseline_alpha)
    new_bl = (1.0 - alpha) * bl + alpha * frame_mean
    out -= new_bl.reshape(n_ch, 1)

    # ── Update running statistics (Welford online) ────────────────────────
    with _lock:
        _baseline = new_bl

        if _running_mean is None:
            _running_mean = frame_mean.copy()
            _running_m2 = np.zeros(n_ch, dtype=np.float64)
        else:
            _n_frames += 1
            delta = frame_mean - _running_mean
            _running_mean += delta / _n_frames
            delta2 = frame_mean - _running_mean
            _running_m2 += delta * delta2

        _n_frames = max(_n_frames, 1)

    return out


def decode(raw: np.ndarray | None) -> tuple[np.ndarray, np.ndarray] | np.ndarray | None:
    """Apply modified Beer–Lambert Law to convert OD to HbO / HbR.

    Channel layout assumed: [pair0_760nm, pair0_850nm, pair1_760nm, pair1_850nm, …]
    Pairs with only one wavelength available are skipped.

    Parameters
    ----------
    raw:
        (n_channels, n_samples) float array of optical-density data
        (pre- or post-apply()).

    Returns
    -------
    (HbO, HbR) tuple, each shape (n_pairs, n_samples), dtype float32.
    Returns (empty, empty) arrays if the channel count is below min_channels
    or the extinction matrix is singular.
    Returns None if raw is None.
    """
    if raw is None:
        return None

    with _lock:
        cfg = copy.copy(_config)

    if not isinstance(raw, np.ndarray) or raw.ndim != 2:
        return raw

    n_ch, n_samples = raw.shape

    if n_ch < cfg.min_channels or n_ch < 2:
        empty = np.zeros((0, n_samples), dtype=np.float32)
        return (empty, empty)

    if abs(_BL_DET) < 1e-12:
        log.warning("fnirs_beer_lambert_singular_matrix")
        empty = np.zeros((0, n_samples), dtype=np.float32)
        return (empty, empty)

    n_pairs = n_ch // 2
    hbo = np.zeros((n_pairs, n_samples), dtype=np.float32)
    hbr = np.zeros((n_pairs, n_samples), dtype=np.float32)

    for i in range(n_pairs):
        od_760 = raw[2 * i].astype(np.float64)
        od_850 = raw[2 * i + 1].astype(np.float64)

        # Invert 2×2 Beer–Lambert system:
        # [eps_HbO_760  eps_HbR_760] [HbO]   [OD_760 / (DPF * d)]
        # [eps_HbO_850  eps_HbR_850] [HbR] = [OD_850 / (DPF * d)]
        #
        # where d is source-detector distance (assumed 1 cm here —
        # device-specific calibration can scale post-hoc).
        a = od_760 / _DPF
        b = od_850 / _DPF

        hbo[i] = ((_EPS_HBR_850 * a - _EPS_HBR_760 * b) / _BL_DET).astype(np.float32)
        hbr[i] = ((-_EPS_HBO_850 * a + _EPS_HBO_760 * b) / _BL_DET).astype(np.float32)

    return (hbo, hbr)
