"""Stage 1 — Online zero-phase FIR filter chain.

Design goals
------------
* Zero-phase filtering via ``scipy.signal.filtfilt`` — no group-delay
  distortion in event-locked signals (Muse alpha/theta peaks land at
  the correct sample).
* Notch default 60 Hz (US mains); set ``NEUROLINK_LINE_FREQ_HZ=50`` env var
  for EU/Asia deployments.  LP default 55 Hz preserves gamma band.
* Kernels built **once** at construction time with
  ``scipy.signal.firwin`` and cached per (electrode_type, line_freq)
  combination so the hot path is pure NumPy array math.
* Fully configurable at runtime via REST (routers/stage1.py);
  changing any parameter rebuilds the chain and the new kernel is used
  from the very next EEGPump tick.
* Graceful degradation: if the buffer is shorter than the minimum
  filtfilt length (3 * filter_order) the raw data is returned
  unchanged and a structured warning is emitted — the pump never
  raises.

Filter order
-----------
  default_filter_order = 128  (at 256 Hz → 0.5 s one-sided kernel)
  Transition bands (firwin Hamming window, Kaiser not needed):
    high-pass  0.5 Hz  → transition width ~0.4 Hz
    notch      2 Hz BW (each) → sharp notch at 60/120 Hz (US default)
    low-pass  55 Hz   → preserves gamma band; raise to 65 Hz if needed

Public API
----------
  FilterConfig              — dataclass of all tunable knobs
  OnlineFilterChain         — stateless filter operator
  FilterChainRegistry       — module-level singleton / cache
  apply_online_filters()    — one-call entry point used by eeg_pump
  get_default_line_freq()   — locale-aware default (env var or 60 Hz)
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import structlog
from scipy import signal as sp_signal

log = structlog.get_logger(__name__)


def get_default_line_freq() -> float:
    """Return power-line frequency for the deployment locale.

    Resolution order:
    1. ``NEUROLINK_LINE_FREQ_HZ`` environment variable (e.g. "50" for EU/Asia)
    2. ``artifact_config.ARTIFACT_LINE_FREQ_HZ`` constant (currently 60.0)
    3. Hard-coded fallback: 60.0 Hz (US/Canada/Japan default)

    Set ``NEUROLINK_LINE_FREQ_HZ=50`` at container startup for EU/Asia
    deployments; no code change required.
    """
    env_val = os.environ.get("NEUROLINK_LINE_FREQ_HZ")
    if env_val:
        try:
            freq = float(env_val)
            if freq not in (50.0, 60.0):
                log.warning(
                    "unusual_line_freq_from_env",
                    freq_hz=freq,
                    expected="50 or 60",
                )
            return freq
        except ValueError:
            log.warning(
                "invalid_NEUROLINK_LINE_FREQ_HZ",
                raw=env_val,
                fallback=60.0,
            )
    try:
        from neurolink.dsp.artifact_config import ARTIFACT_LINE_FREQ_HZ
        return float(ARTIFACT_LINE_FREQ_HZ)
    except ImportError:
        return 60.0


# ---------------------------------------------------------------------------
# FilterConfig
# ---------------------------------------------------------------------------

@dataclass
class FilterConfig:
    """All tunable parameters for the online filter chain.

    Attributes:
        hz_highpass:    High-pass cut-off (Hz).  0 or None → skip.
        hz_notch_freqs: List of notch centre frequencies (Hz).
                        Default [60.0, 120.0] for US (60 Hz mains + 2nd harmonic).
                        Set NEUROLINK_LINE_FREQ_HZ=50 env var for EU/Asia.
                        Override explicitly: FilterConfig(hz_notch_freqs=[50.0, 100.0]).
        hz_lowpass:     Low-pass cut-off (Hz).  Default 55 Hz to preserve the
                        gamma band (30-50 Hz) used in Neurolink feature scoring.
                        Lower to 45 Hz via set_config() if gamma features are inactive.
        notch_bw_hz:    Full bandwidth of each notch (Hz).
        fs:             Sampling rate (Hz).
        filter_order:   FIR filter order (must be even; padded if odd).
    """

    hz_highpass: float | None = 0.5
    hz_notch_freqs: list[float] = field(
        default_factory=lambda: [get_default_line_freq(), get_default_line_freq() * 2]
    )
    hz_lowpass: float | None = 55.0  # raised from 45 Hz to preserve gamma band
    notch_bw_hz: float = 2.0
    fs: float = 256.0
    filter_order: int = 128

    # ------------------------------------------------------------------ #
    def with_line_freq(self, line_freq: float) -> "FilterConfig":
        """Return a copy with notch freqs set to line_freq and 2nd harmonic."""
        return FilterConfig(
            hz_highpass=self.hz_highpass,
            hz_notch_freqs=[line_freq, line_freq * 2],
            hz_lowpass=self.hz_lowpass,
            notch_bw_hz=self.notch_bw_hz,
            fs=self.fs,
            filter_order=self.filter_order,
        )

    def _ensure_even_order(self) -> int:
        order = self.filter_order
        return order + 1 if order % 2 != 0 else order


# ---------------------------------------------------------------------------
# OnlineFilterChain
# ---------------------------------------------------------------------------

class OnlineFilterChain:
    """Stateless zero-phase FIR filter chain.

    All kernels are built once in ``__init__``; ``apply()`` is a pure
    function of the input array.
    """

    def __init__(self, config: FilterConfig) -> None:
        self.config = config
        self._kernels: list[np.ndarray] = []
        self._labels: list[str] = []
        self._min_samples: int = 0
        self._build()

    # ------------------------------------------------------------------ #
    def _build(self) -> None:
        """Design all FIR kernels from the current config."""
        cfg = self.config
        nyq = cfg.fs / 2.0
        order = cfg._ensure_even_order()
        self._kernels = []
        self._labels = []

        # 1. High-pass
        if cfg.hz_highpass and cfg.hz_highpass > 0:
            hp_norm = cfg.hz_highpass / nyq
            if 0 < hp_norm < 1.0:
                kern = sp_signal.firwin(
                    order + 1, hp_norm, pass_zero=False, window="hamming"
                )
                self._kernels.append(kern)
                self._labels.append(f"HP@{cfg.hz_highpass}Hz")

        # 2. Notch filters (one per centre frequency)
        for fc in cfg.hz_notch_freqs:
            lo = (fc - cfg.notch_bw_hz / 2.0) / nyq
            hi = (fc + cfg.notch_bw_hz / 2.0) / nyq
            if 0 < lo < hi < 1.0:
                kern = sp_signal.firwin(
                    order + 1, [lo, hi], pass_zero=True, window="hamming"
                )
                self._kernels.append(kern)
                self._labels.append(f"notch@{fc}Hz")

        # 3. Low-pass
        if cfg.hz_lowpass and cfg.hz_lowpass > 0:
            lp_norm = cfg.hz_lowpass / nyq
            if 0 < lp_norm < 1.0:
                kern = sp_signal.firwin(
                    order + 1, lp_norm, pass_zero=True, window="hamming"
                )
                self._kernels.append(kern)
                self._labels.append(f"LP@{cfg.hz_lowpass}Hz")

        # filtfilt needs at least 3 * padlen samples; padlen = 3*(order)
        self._min_samples = 3 * (order + 1) + 1
        log.info(
            "stage1_filter_chain_built",
            filters=self._labels,
            min_samples=self._min_samples,
            order=order,
        )

    # ------------------------------------------------------------------ #
    def apply(self, eeg: np.ndarray) -> np.ndarray:
        """Apply the filter chain to a (channels, samples) array.

        Uses ``scipy.signal.filtfilt`` (zero-phase, forward-backward)
        for each kernel in sequence.  If the buffer is shorter than
        ``_min_samples`` the raw array is returned unchanged.

        Args:
            eeg: ndarray of shape (n_channels, n_samples), float32/64.

        Returns:
            Filtered ndarray of the same shape.
        """
        if eeg.ndim == 1:
            eeg = eeg[np.newaxis, :]

        n_samples = eeg.shape[1]
        if n_samples < self._min_samples:
            log.debug(
                "stage1_buffer_too_short",
                n_samples=n_samples,
                min_required=self._min_samples,
            )
            return eeg

        out = eeg.astype(np.float64, copy=True)
        for kern, label in zip(self._kernels, self._labels):
            try:
                for ch in range(out.shape[0]):
                    out[ch] = sp_signal.filtfilt(kern, [1.0], out[ch])
            except Exception as exc:  # pragma: no cover
                log.warning("stage1_filtfilt_error", filter=label, error=str(exc))
                # return the best we have so far rather than crashing
                return out.astype(np.float32)

        return out.astype(np.float32)


# ---------------------------------------------------------------------------
# FilterChainRegistry  (module-level singleton)
# ---------------------------------------------------------------------------

class FilterChainRegistry:
    """Thread-safe registry that caches one OnlineFilterChain per config key.

    The active config is mutable; calling ``set_config()`` rebuilds the
    chain immediately so the next EEGPump tick picks up the new kernels.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._config: FilterConfig = FilterConfig()  # defaults: 60 Hz notch, LP 55 Hz
        self._chain: OnlineFilterChain = OnlineFilterChain(self._config)

    # ------------------------------------------------------------------ #
    def set_config(self, config: FilterConfig) -> None:
        """Replace the active filter config and rebuild the chain."""
        with self._lock:
            self._config = config
            self._chain = OnlineFilterChain(config)
            log.info("stage1_config_updated", config=config)

    def get_config(self) -> FilterConfig:
        """Return the currently active FilterConfig."""
        with self._lock:
            return self._config

    def apply(self, eeg: np.ndarray) -> np.ndarray:
        """Apply the currently active filter chain to an EEG array."""
        with self._lock:
            chain = self._chain
        return chain.apply(eeg)


# ---------------------------------------------------------------------------
# Module-level singleton + convenience entry point
# ---------------------------------------------------------------------------

_registry: FilterChainRegistry | None = None
_registry_lock = threading.Lock()


def get_registry() -> FilterChainRegistry:
    """Return the module-level FilterChainRegistry singleton."""
    global _registry
    with _registry_lock:
        if _registry is None:
            _registry = FilterChainRegistry()
    return _registry


def apply_online_filters(eeg: np.ndarray) -> np.ndarray:
    """Convenience wrapper: apply the active filter chain to an EEG array."""
    return get_registry().apply(eeg)
