"""Coverage gap-filling tests for NeuroLinkService."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

from neurolink.hub import EEGHub
from neurolink.models.eeg import BandPowers, IngestPayload
from neurolink.service import NeuroLinkService


def _svc() -> NeuroLinkService:
    return NeuroLinkService(EEGHub())


# ---------------------------------------------------------------------------
# _create_db_session — guard: no factory
# ---------------------------------------------------------------------------

async def test_create_db_session_no_factory_is_noop():
    svc = _svc()
    assert svc._db_session_id is None
    await svc._create_db_session("mock", "mock", None)  # factory=None, returns immediately
    assert svc._db_session_id is None


# ---------------------------------------------------------------------------
# _create_db_session — exception is swallowed
# ---------------------------------------------------------------------------

async def test_create_db_session_exception_is_swallowed():
    svc = _svc()

    @asynccontextmanager
    async def bad_factory():
        raise RuntimeError("db boom")
        yield  # makes it an async generator

    svc._db_session_factory = bad_factory
    await svc._create_db_session("mock", "mock", None)  # must not raise
    assert svc._db_session_id is None


# ---------------------------------------------------------------------------
# _close_db_session — guard branches
# ---------------------------------------------------------------------------

async def test_close_db_session_no_factory_is_noop():
    svc = _svc()
    svc._db_session_id = None
    await svc._close_db_session()


async def test_close_db_session_no_session_id_is_noop():
    svc = _svc()
    svc._db_session_factory = object()  # non-None factory
    svc._db_session_id = None
    await svc._close_db_session()  # guard: session_id is None


async def test_close_db_session_exception_is_swallowed():
    svc = _svc()
    svc._db_session_id = 99

    @asynccontextmanager
    async def bad_factory():
        raise RuntimeError("close boom")
        yield

    svc._db_session_factory = bad_factory
    await svc._close_db_session()  # must not raise


# ---------------------------------------------------------------------------
# start_calibration idempotency
# ---------------------------------------------------------------------------

async def test_start_calibration_idempotent_when_running():
    svc = _svc()
    await svc.connect(adapter_type="mock", device_model="mock")
    try:
        resp1 = await svc.start_calibration()
        assert resp1.status == "started"
        # Replace task with a mock that reports not-done
        fake_task = MagicMock()
        fake_task.done.return_value = False
        svc._calibration_task = fake_task
        resp2 = await svc.start_calibration()
        assert resp2.status == "started"
    finally:
        if svc._calibration_task and not isinstance(svc._calibration_task, MagicMock):
            svc._calibration_task.cancel()
        await svc.disconnect()


# ---------------------------------------------------------------------------
# stream_state — yields an actual pushed value
# ---------------------------------------------------------------------------

async def test_stream_state_yields_pushed_value():
    svc = _svc()
    payload = IngestPayload(
        source="mock",
        bands=BandPowers(alpha=0.5, theta=0.1, beta=0.2, delta=0.1, gamma=0.1),
    )

    yielded = []

    async def consume():
        async for state in svc.stream_state():
            yielded.append(state)
            break

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.01)
    svc._hub.update(payload)
    await asyncio.wait_for(task, timeout=2.0)

    assert len(yielded) == 1
    assert yielded[0].frame_count == 1


# ---------------------------------------------------------------------------
# disconnect when adapter.disconnect() raises
# ---------------------------------------------------------------------------

async def test_disconnect_swallows_adapter_error():
    svc = _svc()
    await svc.connect(adapter_type="mock", device_model="mock")
    svc._adapter.disconnect = AsyncMock(side_effect=RuntimeError("ble gone"))
    result = await svc.disconnect()  # must not raise
    assert result.ok is True
    assert svc._adapter is None


# ---------------------------------------------------------------------------
# get_sessions with a mock DB factory
# ---------------------------------------------------------------------------

async def test_get_sessions_with_factory_returns_summaries():
    from datetime import datetime, timezone

    svc = _svc()

    mock_row = MagicMock()
    mock_row.id = 1
    mock_row.started_at = datetime.now(timezone.utc)
    mock_row.ended_at = None
    mock_row.device_model = "muse_s_gen1"
    mock_row.adapter_type = "mock"
    mock_row.frame_count = 10
    mock_row.final_ea1_eligible = False

    mock_repo = AsyncMock()
    mock_repo.list_recent.return_value = [mock_row]

    @asynccontextmanager
    async def factory():
        yield mock_repo

    svc._db_session_factory = factory

    # Patch the class where it is actually imported at call-time
    with patch("neurolink.db.repository.SessionLogRepository", return_value=mock_repo):
        sessions = await svc.get_sessions(limit=5)

    assert len(sessions) == 1
    assert sessions[0].id == 1
    assert sessions[0].device_model == "muse_s_gen1"
