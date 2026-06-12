"""Branch-coverage tests targeting previously-uncovered lines.

Covers:
- hub._schedule_redis_push (loop.is_running() and RuntimeError paths)
- hub._push_state_to_redis
- service._create_db_session / _close_db_session with real session_id
- service.get_sessions with a mock factory
- eeg_pump._build_payload edge branches (None buffers, fnirs extra, accel no gyro)
- eeg_pump._pump_loop watchdog log
- dsp.classifiers v01 Region B (high beta), Citrinitas (theta flow), faa gate
- dsp.classifiers v2 Citrinitas branch
"""
from __future__ import annotations

import asyncio
import datetime
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from neurolink.dsp.classifiers import classify_v01, classify_v2
from neurolink.hub import EEGHub
from neurolink.models.eeg import BandPowers, IngestPayload, NeurolinkState
from neurolink.service import NeuroLinkService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _payload(**kw) -> IngestPayload:
    defaults = dict(
        source="mock",
        bands=BandPowers(alpha=0.3, theta=0.2, beta=0.15, delta=0.1, gamma=0.05),
    )
    defaults.update(kw)
    return IngestPayload(**defaults)


def _service() -> NeuroLinkService:
    return NeuroLinkService(hub=EEGHub())


# Patch target for SessionLogRepository.
# Each service method does a LAZY import:
#   from neurolink.db.repository import SessionLogRepository
# so the class is never bound on the service module. We must patch it
# at its definition site so the lazy import picks up the mock.
_REPO_PATH = "neurolink.db.repository.SessionLogRepository"


# ===========================================================================
# hub._schedule_redis_push — loop.is_running() == True path
# ===========================================================================

async def test_schedule_redis_push_running_loop():
    """When called from a running event loop, ensure_future is called."""
    hub = EEGHub()
    state = NeurolinkState()
    with patch("neurolink.hub._push_state_to_redis", new=AsyncMock()):
        hub._schedule_redis_push(state)
        await asyncio.sleep(0)  # give the event loop a tick


async def test_schedule_redis_push_runtime_error_suppressed():
    """RuntimeError from get_event_loop() must be silently swallowed."""
    hub = EEGHub()
    state = NeurolinkState()
    with patch("neurolink.hub.asyncio.get_event_loop", side_effect=RuntimeError("no loop")):
        hub._schedule_redis_push(state)  # must not raise


# ===========================================================================
# hub._push_state_to_redis — coroutine path
# ===========================================================================

async def test_push_state_to_redis_calls_cache():
    """_push_state_to_redis must call cache.push_state with the state dict."""
    from neurolink import hub as hub_module
    with patch("neurolink.cache.redis_client.push_state", new=AsyncMock()) as mock_push:
        await hub_module._push_state_to_redis({"frame_count": 1})
        mock_push.assert_called_once_with({"frame_count": 1})


# ===========================================================================
# service._create_db_session — success path
# ===========================================================================

async def test_create_db_session_success():
    """_create_db_session stores the returned entry.id in _db_session_id."""
    svc = _service()

    mock_entry = MagicMock()
    mock_entry.id = 42

    mock_repo = AsyncMock()
    mock_repo.create_session.return_value = mock_entry

    @asynccontextmanager
    async def _factory():
        yield MagicMock()  # fake AsyncSession

    svc.set_db_session_factory(_factory)

    with patch(_REPO_PATH, return_value=mock_repo):
        await svc._create_db_session("mock", "muse_s_gen1", "AA:BB:CC")

    assert svc._db_session_id == 42


async def test_create_db_session_exception_is_swallowed():
    """DB errors in _create_db_session must be caught and logged, not raised."""
    svc = _service()

    @asynccontextmanager
    async def _bad_factory():
        raise RuntimeError("db offline")
        yield  # pragma: no cover

    svc.set_db_session_factory(_bad_factory)
    await svc._create_db_session("mock", "muse_s_gen1", None)  # must not raise
    assert svc._db_session_id is None


# ===========================================================================
# service._close_db_session — success path with real session_id
# ===========================================================================

async def test_close_db_session_with_session_id():
    """_close_db_session calls repo.end_session when factory + id are set."""
    svc = _service()
    svc._db_session_id = 7

    mock_repo = AsyncMock()

    @asynccontextmanager
    async def _factory():
        yield MagicMock()  # fake AsyncSession

    svc.set_db_session_factory(_factory)

    with patch(_REPO_PATH, return_value=mock_repo):
        await svc._close_db_session()

    mock_repo.end_session.assert_called_once()
    assert svc._db_session_id is None


async def test_close_db_session_exception_is_swallowed():
    svc = _service()
    svc._db_session_id = 99

    @asynccontextmanager
    async def _bad_factory():
        raise RuntimeError("db offline")
        yield  # pragma: no cover

    svc.set_db_session_factory(_bad_factory)
    await svc._close_db_session()  # must not raise


# ===========================================================================
# service.get_sessions — factory path
# ===========================================================================

async def test_get_sessions_with_factory():
    """get_sessions returns a SessionSummary list built from repo rows."""
    svc = _service()

    mock_row = MagicMock()
    mock_row.id = 1
    mock_row.started_at = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
    mock_row.ended_at = None
    mock_row.device_model = "muse_s_gen1"
    mock_row.adapter_type = "mock"
    mock_row.frame_count = 100
    mock_row.final_ea1_eligible = False

    mock_repo = AsyncMock()
    mock_repo.list_recent.return_value = [mock_row]

    @asynccontextmanager
    async def _factory():
        yield MagicMock()  # fake AsyncSession

    svc.set_db_session_factory(_factory)

    with patch(_REPO_PATH, return_value=mock_repo):
        sessions = await svc.get_sessions(limit=5)

    assert len(sessions) == 1
    assert sessions[0].id == 1
    assert sessions[0].frame_count == 100


# ===========================================================================
# eeg_pump._build_payload — None eeg_buffer branch
# ===========================================================================

async def test_build_payload_no_eeg_buffer():
    """_build_payload with no eeg_buffer should produce zero BandPowers."""
    from neurolink.eeg_pump import EEGPump
    from neurolink.hardware.base import EEGSample

    hub = EEGHub()
    pump = EEGPump(adapter=AsyncMock(), hub=hub)

    sample = EEGSample(
        source="mock",
        address="",
        timestamp=0.0,
        eeg_buffer=None,
        ppg_buffer=None,
        accel_buffer=None,
        gyro_buffer=None,
        poor_contact=False,
        extra={},
    )
    payload = await pump._build_payload(sample)
    assert payload.bands.alpha == 0.0
    assert payload.faa is None
    assert payload.ppg is None
    assert payload.imu is None


# ===========================================================================
# eeg_pump._build_payload — fnirs_oxy/deoxy from extra dict
# ===========================================================================

async def test_build_payload_fnirs_extra():
    """fnirs_oxy and fnirs_deoxy should be pulled from sample.extra."""
    from neurolink.eeg_pump import EEGPump
    from neurolink.hardware.base import EEGSample

    hub = EEGHub()
    pump = EEGPump(adapter=AsyncMock(), hub=hub)

    sample = EEGSample(
        source="mock",
        address="",
        timestamp=0.0,
        eeg_buffer=None,
        ppg_buffer=None,
        accel_buffer=None,
        gyro_buffer=None,
        poor_contact=False,
        extra={"fnirs_oxy": 1.23, "fnirs_deoxy": 0.45},
    )
    payload = await pump._build_payload(sample)
    assert payload.fnirs_oxy == pytest.approx(1.23)
    assert payload.fnirs_deoxy == pytest.approx(0.45)


# ===========================================================================
# eeg_pump._build_payload — accel_buffer present but no gyro_buffer
# ===========================================================================

async def test_build_payload_accel_no_gyro_no_imu():
    """accel_buffer (3,N) present but gyro_buffer=None → imu_payload is None."""
    from neurolink.eeg_pump import EEGPump
    from neurolink.hardware.base import EEGSample
    import numpy as np

    hub = EEGHub()
    pump = EEGPump(adapter=AsyncMock(), hub=hub)

    accel_buf = np.zeros((3, 10), dtype=np.float32).tolist()  # correct (3,N) shape

    sample = EEGSample(
        source="mock",
        address="",
        timestamp=0.0,
        eeg_buffer=None,
        ppg_buffer=None,
        accel_buffer=accel_buf,
        gyro_buffer=None,  # no gyro → imu branch skipped
        poor_contact=False,
        extra={},
    )
    payload = await pump._build_payload(sample)
    assert payload.imu is None


# ===========================================================================
# eeg_pump._pump_loop — watchdog warning branch
# ===========================================================================

async def test_pump_loop_watchdog_fires():
    """If last_frame_ts is old, the watchdog log.warning branch executes."""
    import time

    from neurolink.eeg_pump import EEGPump

    hub = EEGHub()
    adapter = AsyncMock()
    adapter.read_sample.return_value = None

    pump = EEGPump(adapter=adapter, hub=hub, publish_hz=100.0)
    pump._last_frame_ts = time.time() - 30.0  # stale → triggers watchdog
    pump._running = True

    original_sleep = asyncio.sleep

    async def _one_shot_sleep(secs):
        pump._running = False
        await original_sleep(0)

    with patch("neurolink.eeg_pump.log") as mock_log:
        with patch("neurolink.eeg_pump.asyncio.sleep", side_effect=_one_shot_sleep):
            await pump._pump_loop()

    mock_log.warning.assert_called()


# ===========================================================================
# dsp.classifiers.classify_v01 — Region B (high beta)
# ===========================================================================

def test_v01_region_b_high_beta():
    region, stage = classify_v01(
        alpha=0.10, theta=0.10, beta=0.35, delta=0.10, gamma=0.05
    )
    assert region == "B"
    assert stage == "Albedo"


# ===========================================================================
# dsp.classifiers.classify_v01 — Region D Citrinitas (theta-dominant flow)
# ===========================================================================

def test_v01_region_d_citrinitas():
    region, stage = classify_v01(
        alpha=0.10, theta=0.25, beta=0.10, delta=0.05, gamma=0.05
    )
    assert region == "D"
    assert stage == "Citrinitas"


# ===========================================================================
# dsp.classifiers.classify_v01 — faa gate blocks Multiplicatio → Rubedo
# ===========================================================================

def test_v01_multiplicatio_faa_gate_blocks():
    region, stage = classify_v01(
        alpha=0.40, theta=0.20, beta=0.10, delta=0.05, gamma=0.05,
        faa=-0.10,  # below _V01_MULTIPLICATIO_FAA threshold of -0.05
    )
    assert region == "E"
    assert stage == "Rubedo"


# ===========================================================================
# dsp.classifiers.classify_v2 — Citrinitas branch
# ===========================================================================

def test_v2_citrinitas():
    region, stage = classify_v2(
        BandPowers(alpha=0.22, theta=0.12, beta=0.10, delta=0.10, gamma=0.05)
    )
    assert region == "D"
    assert stage == "Citrinitas"
