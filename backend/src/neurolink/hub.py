"""EEGHub — in-memory EEG state store.

Process-global, thread-safe singleton hub.
Dual classifier: v2 always runs; v0.1 runs only when source == 'muse_ble'.
Ported from Rigpa-v2 hub.py + Rigpa-v3 hub.py.

SSE event types
---------------
The hub pushes three kinds of objects onto registered SSE queues:

  NeurolinkState        — normal per-frame data tick (every ~250 ms at 4 Hz)
  _BASELINE_COMPLETE_EVENT  — fired once when the 150 s baseline finishes;
                              clients play a bell and unlock the session UI
  _SETTLING_EVENT       — fired on every frame held by Stage 0 acquisition
                          guard; carries a structured reason code so the
                          frontend can show a contextual waiting message:
                            'impedance_unstable' — electrode contact not stable
                            'motion_settling'    — movement; waiting for rest
                            'env_not_ready'      — environment not ready
                            'settling'           — generic fallback

SSE consumers distinguish the three types by checking for 'event' key:
  if isinstance(item, dict) and item.get('event') == 'baseline_complete': ...
  if isinstance(item, dict) and item.get('event') == 'settling': ...
  else:  # NeurolinkState
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import TYPE_CHECKING

import structlog

from neurolink.dsp.classifiers import classify_v01, classify_v2, compute_s_space
from neurolink.ea1_scorer import score as ea1_score
from neurolink.fatigue import FatigueDetector
from neurolink.focus_state import classify_focus, compute_focus_score, set_current_focus_score
from neurolink.models.eeg import (
    EA1Result,
    IngestPayload,
    NeurolinkState,
)

if TYPE_CHECKING:
    from neurolink.hardware.base import EEGSample

log = structlog.get_logger(__name__)

_DEFAULT_BASELINE_ALPHA: float = 0.30

# Sentinel pushed to SSE queues when the 150 s baseline completes.
# Clients receive event type "baseline_complete" carrying this dict.
_BASELINE_COMPLETE_EVENT: dict = {"event": "baseline_complete"}

# Settling sentinel template — cloned per call with the reason injected.
# Clients receive event type "settling" while Stage 0 holds frames.
_SETTLING_EVENT_TYPE: str = "settling"


class EEGHub:
    """Central in-memory EEG state store.

    Single writer (EEGPump) via update(); multiple readers via get_state().
    All writes are protected by a threading.Lock.
    SSE fan-out uses per-client asyncio.Queue populated in update().
    Redis write-through (Task 7.2): hub.update() calls cache.push_state() after
    every state update when NEUROLINK_REDIS_ENABLED=true.

    SSE queue item types
    --------------------
    Queues carry three distinct item types (see module docstring for details):
      NeurolinkState            — normal data tick
      _BASELINE_COMPLETE_EVENT  — one-shot bell event
      settling dict             — emitted by emit_settling() on held frames
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = NeurolinkState()
        self._ea1 = EA1Result()
        self._latest_sample: EEGSample | None = None
        self._fatigue = FatigueDetector()
        self.baseline_alpha: float = _DEFAULT_BASELINE_ALPHA
        # SSE fan-out queues: one per connected SSE client
        self._sse_queues: list[asyncio.Queue] = []
        self._sse_lock = threading.Lock()
        # Diagnostic counters
        self._settling_events_emitted: int = 0
        self._frames_processed: int = 0

    # ── Core update ────────────────────────────────────────────────────────────

    def update(self, payload: IngestPayload) -> NeurolinkState:
        """Ingest a new payload, run classifiers, update state, fan-out to SSE."""
        bands = payload.bands

        # ── v2 classifier (always runs) ─────────────────────────────────────
        region_v2, stage_v2 = classify_v2(bands)
        s_space = compute_s_space(bands)

        payload.region = region_v2
        payload.alchemical_stage = stage_v2
        payload.s_space = s_space
        payload.integration_coverage = s_space.y
        payload.engagement_index = s_space.x

        # ── v0.1 classifier (muse_ble only) ──────────────────────────────
        region_v01 = "A"
        stage_v01 = "Nigredo"
        if payload.source == "muse_ble":
            region_v01, stage_v01 = classify_v01(
                alpha=bands.alpha,
                theta=bands.theta,
                beta=bands.beta,
                delta=bands.delta,
                gamma=bands.gamma,
                faa=payload.faa,
                fmt=payload.fmt,
            )

        # ── EA-1 scoring ───────────────────────────────────────────────
        ea1_result = ea1_score(payload)

        # ── Focus + Fatigue ──────────────────────────────────────────────
        fatigue_score = self._fatigue.update(bands.theta, bands.alpha)
        focus_score = compute_focus_score(bands.alpha, bands.beta, self.baseline_alpha)
        focus_state = classify_focus(focus_score)

        set_current_focus_score(focus_score)

        # ── Build NeurolinkState ─────────────────────────────────────────
        with self._lock:
            prev_count = self._state.frame_count
            new_state = NeurolinkState(
                connected=True,
                source=payload.source,
                region=region_v2,
                alchemical_stage=stage_v2,
                integration_coverage=s_space.y,
                engagement_index=s_space.x,
                bands=bands,
                s_space=s_space,
                ea1=ea1_result,
                last_ts=payload.timestamp or time.time(),
                frame_count=prev_count + 1,
                poor_contact=payload.poor_contact,
                region_v01=region_v01,
                alchemical_stage_v01=stage_v01,
                faa=payload.faa,
                fmt=payload.fmt,
                hr_bpm=payload.ppg.hr_bpm if payload.ppg else None,
                hrv_rmssd=payload.ppg.hrv_rmssd if payload.ppg else None,
                rr_bpm=payload.breathing.rr_bpm if payload.breathing else None,
                pitch_deg=payload.imu.pitch_deg if payload.imu else None,
                roll_deg=payload.imu.roll_deg if payload.imu else None,
                motion_rms=payload.imu.motion_rms if payload.imu else None,
                contact_quality=payload.contact_quality,
                focus_state=focus_state.value,
                focus_score=focus_score,
                fatigue_score=fatigue_score,
                fnirs_oxy=payload.fnirs_oxy,
                fnirs_deoxy=payload.fnirs_deoxy,
                eeg_samples=payload.eeg_samples,
                bad_channels=payload.bad_channels,
                artifact_rejected=payload.artifact_rejected,
                artifact_reasons=payload.artifact_reasons,
                artifact_annotations=payload.artifact_annotations,
                artifact_correction_plan=payload.artifact_correction_plan,
                channel_impedances=payload.channel_impedances,
                baseline_phase=payload.baseline_phase,
            )
            self._state = new_state
            self._ea1 = ea1_result
            self._frames_processed += 1

        self._fanout(new_state)
        self._schedule_redis_push(new_state)

        return new_state

    # ── SSE sentinel events ────────────────────────────────────────────────────

    def notify_baseline_complete(self) -> None:
        """Push a baseline_complete sentinel to all SSE queues.

        Called exactly once per session by BaselineRecorder._fire_bell().
        The sentinel is a plain dict — not a NeurolinkState — so SSE
        consumers must check the type before deserialising.
        Clients that receive this event should play a bell sound and
        unlock the session UI.
        """
        log.info("hub_baseline_complete_notified")
        with self._sse_lock:
            queues = list(self._sse_queues)
        for q in queues:
            try:
                q.put_nowait(_BASELINE_COMPLETE_EVENT)
            except asyncio.QueueFull:
                log.warning("sse_queue_full_dropping_baseline_event")

    def emit_settling(self, reason: str = "settling") -> None:
        """Push a settling sentinel to all SSE queues.

        Called by EEGPump._tick() on every frame held by the Stage 0
        acquisition guard (impedance not stable, motion detected, or
        environment not ready).  The frontend uses the reason code to
        display a contextual waiting indicator:

          'impedance_unstable'  — "Adjusting headband / electrode contact"
          'motion_settling'     — "Keep still"
          'env_not_ready'       — "Calibrating sensors"
          'settling'            — generic spinner

        Emitting a settling event does NOT update NeurolinkState — the
        frame was dropped before any DSP processing.  The SSE stream
        therefore carries both normal ticks (NeurolinkState) and
        out-of-band sentinel dicts; consumers must branch on type.

        Parameters
        ----------
        reason:
            One of the four reason codes above.  Unknown values are
            forwarded as-is so future codes are forwards-compatible.
        """
        event = {"event": _SETTLING_EVENT_TYPE, "reason": reason}
        with self._sse_lock:
            queues = list(self._sse_queues)
        for q in queues:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                log.warning(
                    "sse_queue_full_dropping_settling_event",
                    reason=reason,
                )
        with self._lock:
            self._settling_events_emitted += 1
        log.debug("hub_settling_emitted", reason=reason)

    # ── Diagnostics ────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Return lightweight diagnostic counters.

        Safe to call from any thread at any time.  Values are best-effort
        snapshots — they may be slightly stale under concurrent writes.

        Returns
        -------
        dict with keys:
          frames_processed        — total update() calls since construction/reset
          settling_events_emitted — total emit_settling() calls
          sse_client_count        — number of currently registered SSE queues
          frame_count             — NeurolinkState.frame_count of current state
          baseline_phase          — current baseline phase string
        """
        with self._lock:
            frame_count = self._state.frame_count
            baseline_phase = self._state.baseline_phase
            frames_processed = self._frames_processed
            settling_events_emitted = self._settling_events_emitted
        with self._sse_lock:
            sse_client_count = len(self._sse_queues)
        return {
            "frames_processed": frames_processed,
            "settling_events_emitted": settling_events_emitted,
            "sse_client_count": sse_client_count,
            "frame_count": frame_count,
            "baseline_phase": baseline_phase,
        }

    # ── Redis write-through ────────────────────────────────────────────────────

    def _schedule_redis_push(self, state: NeurolinkState) -> None:
        """Schedule a non-blocking Redis write-through for the new state."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                _task = asyncio.ensure_future(  # noqa: RUF006
                    _push_state_to_redis(state.model_dump()),
                    loop=loop,
                )
        except RuntimeError:
            pass

    # ── State accessors ────────────────────────────────────────────────────────

    def get_state(self) -> NeurolinkState:
        """Return the current NeurolinkState snapshot."""
        with self._lock:
            return self._state

    def get_ea1(self) -> EA1Result:
        """Return the latest EA1Result."""
        with self._lock:
            return self._ea1

    def snapshot(self) -> dict:
        """Return current state as a dict (for Redis caching)."""
        return self.get_state().model_dump()

    def get_latest(self) -> EEGSample | None:
        """Return the latest raw EEGSample (may be None before first frame)."""
        with self._lock:
            return self._latest_sample

    def set_latest_sample(self, sample: EEGSample) -> None:
        """Store the latest raw EEGSample."""
        with self._lock:
            self._latest_sample = sample

    def reset(self) -> None:
        """Reset hub to initial state (used in tests and on disconnect)."""
        with self._lock:
            self._state = NeurolinkState()
            self._ea1 = EA1Result()
            self._latest_sample = None
            self._fatigue.reset()
            self.baseline_alpha = _DEFAULT_BASELINE_ALPHA
            self._frames_processed = 0
            self._settling_events_emitted = 0

    # ── SSE queue management ───────────────────────────────────────────────────

    def register_sse_queue(self, q: asyncio.Queue) -> None:
        """Register a per-client SSE asyncio queue for fan-out."""
        with self._sse_lock:
            self._sse_queues.append(q)

    def unregister_sse_queue(self, q: asyncio.Queue) -> None:
        """Unregister a client SSE queue."""
        with self._sse_lock:
            try:
                self._sse_queues.remove(q)
            except ValueError:
                pass

    def _fanout(self, state: NeurolinkState) -> None:
        """Push state to all registered SSE queues (non-blocking put_nowait)."""
        with self._sse_lock:
            queues = list(self._sse_queues)
        for q in queues:
            try:
                q.put_nowait(state)
            except asyncio.QueueFull:
                log.warning("sse_queue_full_dropping_frame")


async def _push_state_to_redis(state_dict: dict) -> None:
    """Coroutine: push state dict to Redis (Task 7.2 write-through)."""
    from neurolink.cache.redis_client import push_state

    await push_state(state_dict)


# ── Module-level singleton ──────────────────────────────────────────────────

_hub: EEGHub = EEGHub()


def get_hub() -> EEGHub:
    """Return the global EEGHub singleton."""
    return _hub


def update(payload: IngestPayload) -> NeurolinkState:
    """Module-level update delegate."""
    return _hub.update(payload)


def get_state() -> NeurolinkState:
    """Module-level get_state delegate."""
    return _hub.get_state()


def get_ea1() -> EA1Result:
    """Module-level get_ea1 delegate."""
    return _hub.get_ea1()


def snapshot() -> dict:
    """Module-level snapshot delegate."""
    return _hub.snapshot()


def reset() -> None:
    """Module-level reset delegate."""
    _hub.reset()


def emit_settling(reason: str = "settling") -> None:
    """Module-level emit_settling delegate."""
    _hub.emit_settling(reason=reason)


def get_stats() -> dict:
    """Module-level get_stats delegate."""
    return _hub.get_stats()
