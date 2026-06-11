"""EEG Hub — in-memory state store with dual-classifier enrichment.

Ported from Rigpa-v2 hub.py + Rigpa-v3 hub.py.
Thread-safe via threading.Lock. Singleton per process.
"""
from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from typing import Any

import structlog

from neurolink.dsp.classifiers import classify_v01, classify_v2, compute_s_space
from neurolink.ea1_scorer import score as ea1_score
from neurolink.fatigue import FatigueDetector
from neurolink.focus_state import FocusState, classify_focus
from neurolink.models.eeg import (
    BandPowers,
    EA1Result,
    IngestPayload,
    NeurolinkState,
)

log = structlog.get_logger(__name__)

_SSE_QUEUE_MAX = 64


class EEGHub:
    """In-memory EEG state hub.

    Dual classifier tracks:
    - v2 (8-region alchemical): always runs
    - v0.1 (6-region S-space): only when source == 'muse_ble'

    Public methods are thread-safe.
    SSE fan-out via asyncio.Queue (per-client).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = NeurolinkState()
        self._latest_sample: IngestPayload | None = None
        self._ea1: EA1Result = EA1Result()
        self._fatigue = FatigueDetector()
        self.baseline_alpha: float | None = None
        self._sse_queues: list[asyncio.Queue[NeurolinkState]] = []
        self._sse_lock = threading.Lock()

    def update(self, payload: IngestPayload) -> NeurolinkState:
        """Ingest a frame, enrich with classifiers, update state.

        Returns the updated NeurolinkState.
        """
        with self._lock:
            bands = payload.bands

            # v2 classifier (always)
            region_v2, stage_v2 = classify_v2(bands)
            s_space = compute_s_space(bands)

            # v0.1 classifier (muse_ble only)
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
            else:
                region_v01 = "A"
                stage_v01 = "Nigredo"

            # Update payload with classified region for EA-1 scoring
            payload_for_ea1 = payload.model_copy(
                update={
                    "region": region_v2,
                    "alchemical_stage": stage_v2,
                    "s_space": s_space,
                    "integration_coverage": s_space.y,
                    "engagement_index": s_space.x,
                }
            )

            # EA-1 scoring
            ea1_result = ea1_score(payload_for_ea1)
            self._ea1 = ea1_result

            # Fatigue
            fatigue_score = self._fatigue.update(bands.theta, bands.alpha)

            # Focus score from EA-1 score (calibration-normalised when available)
            raw_alpha = bands.alpha
            if self.baseline_alpha is not None and self.baseline_alpha > 0:
                focus_score = min(1.0, raw_alpha / self.baseline_alpha)
            else:
                focus_score = min(1.0, raw_alpha * 3.33)  # rough normalise
            focus_state = classify_focus(focus_score)

            # Build state
            self._state = NeurolinkState(
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
                frame_count=self._state.frame_count + 1,
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
            )
            self._latest_sample = payload
            state_copy = self._state.model_copy()

        # Fan-out to SSE queues (non-blocking)
        self._publish_to_sse(state_copy)
        return state_copy

    def _publish_to_sse(self, state: NeurolinkState) -> None:
        """Push state to all registered SSE queues (best-effort)."""
        with self._sse_lock:
            dead: list[int] = []
            for i, q in enumerate(self._sse_queues):
                try:
                    q.put_nowait(state)
                except asyncio.QueueFull:
                    dead.append(i)
            # Remove stale queues
            for i in reversed(dead):
                self._sse_queues.pop(i)

    def register_sse_queue(self) -> asyncio.Queue[NeurolinkState]:
        """Register a new SSE client queue and return it."""
        q: asyncio.Queue[NeurolinkState] = asyncio.Queue(maxsize=_SSE_QUEUE_MAX)
        with self._sse_lock:
            self._sse_queues.append(q)
        return q

    def deregister_sse_queue(self, q: asyncio.Queue[NeurolinkState]) -> None:
        """Remove an SSE client queue."""
        with self._sse_lock:
            try:
                self._sse_queues.remove(q)
            except ValueError:
                pass

    def get_state(self) -> NeurolinkState:
        """Return a copy of the current NeurolinkState."""
        with self._lock:
            return self._state.model_copy()

    def get_ea1(self) -> EA1Result:
        """Return the latest EA-1 result."""
        with self._lock:
            return self._ea1

    def get_latest(self) -> IngestPayload | None:
        """Return the latest ingested payload."""
        with self._lock:
            return self._latest_sample

    def snapshot(self) -> dict[str, Any]:
        """Return hub state as a plain dict for external consumers."""
        with self._lock:
            return self._state.model_dump()

    def reset(self) -> None:
        """Reset hub to initial state."""
        with self._lock:
            self._state = NeurolinkState()
            self._latest_sample = None
            self._ea1 = EA1Result()
            self._fatigue.reset()
            self.baseline_alpha = None
        with self._sse_lock:
            self._sse_queues.clear()


# Process-global singleton
_hub: EEGHub = EEGHub()


def get_hub() -> EEGHub:
    """Return the process-global EEGHub singleton."""
    return _hub


def reset() -> None:
    """Reset the global hub (used in tests)."""
    _hub.reset()
