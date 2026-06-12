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

  2. **ASR calibration window**:
     Frames accepted during the RECORDING phase (seconds 30-150) are
     forwarded to the ASR instance for covariance-model fitting.  This
     120-second clean window replaces the previous 30-second default and
     produces substantially more stable burst-reconstruction statistics.

State machine
-------------
  WARMUP    — electrode stabilisation; frames discarded, nothing forwarded
  RECORDING — clean frames forwarded to ASR; counter advancing
  COMPLETE  — bell event fired once via hub; all subsequent frames pass
              directly to ASR as normal (baseline has no further effect)

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

    # Every clean tick (after Stage 3 passes):
    eeg_arr = self._baseline.process(eeg_arr)
    payload.baseline_phase = self._baseline.phase

The recorder is a drop-in shim: process() always returns the (unchanged)
eeg_arr so the rest of the pipeline is unaffected.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
import structlog

from neurolink.dsp.artifact_config import BASELINE_DISCARD_SEC, BASELINE_TOTAL_SEC

if TYPE_CHECKING:
    from neurolink.dsp.asr import ArtifactSubspaceReconstructor

log = structlog.get_logger(__name__)


class BaselinePhase(str, Enum):
    WARMUP = "warmup"       # electrodes stabilising — frames discarded
    RECORDING = "recording" # accumulating ASR calibration data
    COMPLETE = "complete"   # baseline done; bell has fired


class BaselineRecorder:
    """Manages the per-session resting baseline window.

    Parameters
    ----------
    asr:
        The session's ArtifactSubspaceReconstructor instance.  Clean
        frames during the RECORDING phase are forwarded to asr.apply()
        so that ASR calibrates on genuinely rested, stabilised data.
    hub:
        The EEGHub instance.  Used only once: to fire the bell event
        when the baseline transitions to COMPLETE.
    """

    def __init__(
        self,
        asr: ArtifactSubspaceReconstructor,
        hub,  # EEGHub — avoid circular import with TYPE_CHECKING only
    ) -> None:
        self._asr = asr
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
        """Advance the state machine and (conditionally) feed ASR.

        Called on every clean frame (Stage 3 passed, not artifact_rejected).
        Returns eeg_arr unchanged in all phases — this is a pure side-effect
        shim so the downstream pipeline requires no branching.

        Phase transitions
        -----------------
        WARMUP     → RECORDING  when elapsed >= BASELINE_DISCARD_SEC
        RECORDING  → COMPLETE   when elapsed >= BASELINE_TOTAL_SEC
        COMPLETE   → COMPLETE   (terminal state)
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
            # Frames during WARMUP are discarded — do NOT feed ASR.
            return eeg_arr

        if self._phase is BaselinePhase.RECORDING:
            if elapsed >= BASELINE_TOTAL_SEC:
                self._phase = BaselinePhase.COMPLETE
                self._fire_bell(elapsed)
            else:
                # Feed clean frame to ASR for covariance model fitting.
                # asr.apply() is idempotent once calibrated so calling it
                # here is safe regardless of whether ASR is still in its
                # own calib window.
                self._asr.apply(eeg_arr)

        # COMPLETE phase: baseline has no further effect.
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
