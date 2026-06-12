"""BaselineRecorder — 150-second eyes-closed resting baseline manager.

Purpose
-------
Every Neurolink session begins with a 150-second eyes-closed resting
baseline that serves two independent goals:

  1. **Impedance stabilisation** (dry electrodes):
     The first 30 seconds (BASELINE_DISCARD_SEC) are silently discarded.
     Dry electrodes take 20-40 s to equilibrate with the scalp via the
     sweat film; data from this period is mechanically and electrically
     unreliable regardless of signal amplitude.

  2. **ASR calibration gate**:
     This recorder no longer calls asr.apply() directly.  Instead it
     exposes the current phase via the ``phase`` property, and
     EEGPump._build_payload() guards Stage 4 (ASR) with::

         self._baseline.phase != "warmup"

     This ensures ASR only receives frames from the post-warmup
     RECORDING and COMPLETE phases, where electrode contact is stable.
     The guard lives in the pump (not here) so the control flow is
     explicit and auditable in one place.

State machine
-------------
  WARMUP    — electrode stabilisation; frames discarded, nothing forwarded
  RECORDING — phase gate lifted; ASR receives frames via main pipeline
  COMPLETE  — bell event fired once via hub; phase gate remains lifted

Bell notification
-----------------
On the first tick that crosses the COMPLETE boundary the recorder calls
hub.notify_baseline_complete().  That method pushes a special
baseline_complete SSE sentinel to all connected clients so the frontend
can play a bell sound and unlock the session UI.

Usage (EEGPump)
---------------
    # Once at startup:
    self._baseline = BaselineRecorder(asr=self._stage4, hub=hub)

    # Every clean tick (Stage 4b — runs BEFORE Stage 4 / ASR):
    eeg_arr = self._baseline.process(eeg_arr)

    # Stage 4 guard (in the same tick, after Stage 4b):
    if self._baseline.phase != "warmup":
        eeg_arr = self._stage4.apply(eeg_arr)

    payload.baseline_phase = self._baseline.phase

The recorder is a drop-in shim: process() always returns the (unchanged)
eeg_arr so the rest of the pipeline is unaffected.
"""

from __future__ import annotations

import threading
import time
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog

from neurolink.dsp.artifact_config import BASELINE_DISCARD_SEC, BASELINE_TOTAL_SEC

if TYPE_CHECKING:
    from neurolink.dsp.asr import ArtifactSubspaceReconstructor

log = structlog.get_logger(__name__)


class BaselinePhase(str, Enum):
    WARMUP = "warmup"       # electrodes stabilising — ASR gate closed
    RECORDING = "recording" # ASR gate open; baseline window accumulating
    COMPLETE = "complete"   # baseline done; bell has fired; ASR gate open


class BaselineRecorder:
    """Manages the per-session resting baseline window.

    Parameters
    ----------
    asr:
        The session's ArtifactSubspaceReconstructor instance.  Retained
        for API compatibility; no longer called directly from this class.
        The pump guards Stage 4 using ``self._baseline.phase != "warmup"``
        so ASR calibrates only on post-warmup frames.
    hub:
        The EEGHub instance.  Used only once: to fire the bell event
        when the baseline transitions to COMPLETE.
    """

    def __init__(
        self,
        asr: ArtifactSubspaceReconstructor,
        hub,  # EEGHub — avoid circular import with TYPE_CHECKING only
    ) -> None:
        self._asr = asr  # kept for API compatibility; not called here
        self._hub = hub
        self._phase: BaselinePhase = BaselinePhase.WARMUP
        self._start_ts: float = time.monotonic()
        self._bell_fired: bool = False

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def phase(self) -> str:
        """Current phase as a plain string (matches BaselinePhase enum value)."""
        return self._phase.value

    @property
    def is_complete(self) -> bool:
        return self._phase is BaselinePhase.COMPLETE

    def process(self, eeg_arr: np.ndarray) -> np.ndarray:
        """Advance the state machine and fire the bell when complete.

        Called on every clean frame (Stage 3 passed, not artifact_rejected),
        and MUST be called before Stage 4 (ASR) each tick so the phase
        gate is up-to-date when the pump evaluates it.

        Returns eeg_arr unchanged in all phases — this is a pure side-effect
        shim so the downstream pipeline requires no branching.

        Phase transitions
        -----------------
        WARMUP     → RECORDING  when elapsed >= BASELINE_DISCARD_SEC
        RECORDING  → COMPLETE   when elapsed >= BASELINE_TOTAL_SEC
        COMPLETE   → COMPLETE   (terminal state)

        Note
        ----
        This method no longer calls self._asr.apply().  ASR is driven
        exclusively by EEGPump._build_payload() (Stage 4), which is
        gated on ``self._baseline.phase != "warmup"``.
        """
        elapsed = time.monotonic() - self._start_ts

        if self._phase is BaselinePhase.WARMUP:
            if elapsed >= BASELINE_DISCARD_SEC:
                self._phase = BaselinePhase.RECORDING
                log.info(
                    "baseline_recording_started",
                    elapsed_s=round(elapsed, 1),
                    discard_s=BASELINE_DISCARD_SEC,
                )
            # WARMUP: return early — ASR gate remains closed this tick.
            return eeg_arr

        if self._phase is BaselinePhase.RECORDING:
            if elapsed >= BASELINE_TOTAL_SEC:
                self._phase = BaselinePhase.COMPLETE
                self._fire_bell(elapsed)

        # RECORDING / COMPLETE: ASR gate is open; pump handles Stage 4.
        return eeg_arr

    def reset(self) -> None:
        """Reset to WARMUP (called on reconnect or session restart)."""
        self._phase = BaselinePhase.WARMUP
        self._start_ts = time.monotonic()
        self._bell_fired = False
        log.info("baseline_recorder_reset")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _fire_bell(self, elapsed: float) -> None:
        """Fire the bell SSE event exactly once."""
        if self._bell_fired:
            return
        self._bell_fired = True
        log.info(
            "baseline_complete",
            elapsed_s=round(elapsed, 1),
            total_s=BASELINE_TOTAL_SEC,
            discard_s=BASELINE_DISCARD_SEC,
        )
        try:
            self._hub.notify_baseline_complete()
        except Exception as exc:  # pragma: no cover
            log.warning("baseline_bell_notify_failed", error=str(exc))


# =============================================================================
# ASR Covariance Cache
# =============================================================================
# Persists ASR calibration covariance across short reconnection windows so
# that a quick device disconnect / reconnect does not require a full 150 s
# baseline re-run.  Entries expire after _COV_CACHE_TTL_SEC (5 min) to
# discard stale covariance when electrode conditions have drifted.
#
# Usage (EEGPump.reset + EEGPump.__init__):
#   On disconnect:  save_asr_covariance(device_address, user_id, asr.get_calibration_cov())
#   On reconnect:   cov = load_asr_covariance(device_address, user_id)
#                   if cov is not None: asr.set_calibration_cov(cov)
# =============================================================================

_cov_cache: dict[str, dict] = {}
_cov_cache_lock = threading.Lock()
_COV_CACHE_TTL_SEC: float = 300.0  # 5 minutes


def _cov_cache_key(device_address: str, user_id: str) -> str:
    return f"{device_address}::{user_id}"


def save_asr_covariance(
    device_address: str,
    user_id: str,
    covariance: Any,
) -> None:
    """Persist an ASR calibration covariance matrix for a (device, user) pair.

    Called by EEGPump.reset() before clearing state so that a quick
    reconnect can restore the pre-calibration without a full 150 s baseline.

    Parameters
    ----------
    device_address:
        BLE MAC or LSL stream ID that uniquely identifies the hardware.
    user_id:
        Session or user identifier.  Use settings.user_id or a session UUID.
    covariance:
        np.ndarray returned by ArtifactSubspaceReconstructor.get_calibration_cov().
    """
    key = _cov_cache_key(device_address, user_id)
    with _cov_cache_lock:
        _cov_cache[key] = {"cov": covariance, "ts": time.monotonic()}
    log.info("asr_covariance_cached", key=key)


def load_asr_covariance(
    device_address: str,
    user_id: str,
) -> Any | None:
    """Retrieve a cached ASR covariance matrix if still within TTL.

    Returns None if:
    - No entry exists for the key.
    - The entry is older than _COV_CACHE_TTL_SEC (5 min).
      After a long disconnect electrode conditions will have drifted
      enough to warrant fresh calibration.

    Parameters
    ----------
    device_address / user_id : same keys passed to save_asr_covariance().
    """
    key = _cov_cache_key(device_address, user_id)
    with _cov_cache_lock:
        entry = _cov_cache.get(key)
    if entry is None:
        log.debug("asr_covariance_cache_miss", key=key)
        return None
    age = time.monotonic() - entry["ts"]
    if age > _COV_CACHE_TTL_SEC:
        log.info("asr_covariance_cache_expired", key=key, age_s=round(age, 1))
        with _cov_cache_lock:
            _cov_cache.pop(key, None)
        return None
    log.info("asr_covariance_cache_hit", key=key, age_s=round(age, 1))
    return entry["cov"]
