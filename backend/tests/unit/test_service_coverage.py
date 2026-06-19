"""Branch-coverage tests for service.py (NeuroLinkService)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from neurolink.exceptions import AdapterAlreadyConnectedError, AdapterNotConnectedError
from neurolink.hub import EEGHub
from neurolink.service import NeuroLinkService


def _service() -> NeuroLinkService:
    return NeuroLinkService(hub=EEGHub())


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------


def test_service_initial_state():
    svc = _service()
    assert svc.is_connected is False
    assert svc.adapter_type == "mock"


# ---------------------------------------------------------------------------
# connect() + disconnect() round-trip
# ---------------------------------------------------------------------------


async def test_connect_mock_and_disconnect():
    svc = _service()
    resp = await svc.connect(adapter_type="mock", device_model="mock")
    assert resp.ok is True
    assert svc.is_connected is True

    resp2 = await svc.disconnect()
    assert resp2.ok is True
    assert svc.is_connected is False


# ---------------------------------------------------------------------------
# connect() twice raises AdapterAlreadyConnectedError
# ---------------------------------------------------------------------------


async def test_connect_twice_raises():
    svc = _service()
    await svc.connect(adapter_type="mock", device_model="mock")
    with pytest.raises(AdapterAlreadyConnectedError):
        await svc.connect(adapter_type="mock", device_model="mock")
    await svc.disconnect()


# ---------------------------------------------------------------------------
# disconnect() with no adapter (never connected)
# ---------------------------------------------------------------------------


async def test_disconnect_when_not_connected_is_safe():
    svc = _service()
    resp = await svc.disconnect()
    assert resp.ok is True


# ---------------------------------------------------------------------------
# disconnect() — adapter.disconnect() raises, still cleans up
# ---------------------------------------------------------------------------


async def test_disconnect_adapter_error_still_cleans_up():
    svc = _service()
    await svc.connect(adapter_type="mock", device_model="mock")
    svc._adapter.disconnect = AsyncMock(side_effect=RuntimeError("hw crash"))
    resp = await svc.disconnect()
    assert resp.ok is True
    assert svc._adapter is None


# ---------------------------------------------------------------------------
# get_current_state
# ---------------------------------------------------------------------------


async def test_get_current_state():
    svc = _service()
    state = await svc.get_current_state()
    assert hasattr(state, "frame_count")


# ---------------------------------------------------------------------------
# get_band_powers
# ---------------------------------------------------------------------------


async def test_get_band_powers_returns_response():
    svc = _service()
    resp = await svc.get_band_powers(channel="mean")
    assert resp.channel == "mean"
    assert isinstance(resp.alpha, float)


# ---------------------------------------------------------------------------
# get_ea1
# ---------------------------------------------------------------------------


async def test_get_ea1():
    from neurolink.models.eeg import EA1Result

    svc = _service()
    result = await svc.get_ea1()
    assert isinstance(result, EA1Result)


# ---------------------------------------------------------------------------
# start_calibration — no adapter raises
# ---------------------------------------------------------------------------


async def test_start_calibration_no_adapter_raises():
    svc = _service()
    with pytest.raises(AdapterNotConnectedError):
        await svc.start_calibration()


# ---------------------------------------------------------------------------
# start_calibration — starts task
# ---------------------------------------------------------------------------


async def test_start_calibration_starts_task():
    svc = _service()
    await svc.connect(adapter_type="mock", device_model="mock")
    resp = await svc.start_calibration()
    assert resp.status == "started"
    assert svc._calibration_task is not None
    # cleanup — CancelledError is expected; any other exception is also swallowed
    # because this is test teardown, not application logic.  noqa: S110 is
    # intentional: we cannot log here without a logger fixture.
    svc._calibration_task.cancel()
    try:
        await svc._calibration_task
    except (asyncio.CancelledError, Exception):  # noqa: S110
        pass
    await svc.disconnect()


# ---------------------------------------------------------------------------
# start_calibration — idempotent while already running
# ---------------------------------------------------------------------------


async def test_start_calibration_idempotent():
    svc = _service()
    await svc.connect(adapter_type="mock", device_model="mock")
    resp1 = await svc.start_calibration()
    resp2 = await svc.start_calibration()  # task still pending
    assert resp1.status == "started"
    assert resp2.status == "started"
    svc._calibration_task.cancel()
    try:
        await svc._calibration_task
    except (asyncio.CancelledError, Exception):  # noqa: S110
        pass
    await svc.disconnect()


# ---------------------------------------------------------------------------
# get_sessions — no factory returns empty list
# ---------------------------------------------------------------------------


async def test_get_sessions_no_factory():
    svc = _service()
    sessions = await svc.get_sessions()
    assert sessions == []


# ---------------------------------------------------------------------------
# set_db_session_factory
# ---------------------------------------------------------------------------


def test_set_db_session_factory():
    svc = _service()
    factory = MagicMock()
    svc.set_db_session_factory(factory)
    assert svc._db_session_factory is factory


# ---------------------------------------------------------------------------
# _close_db_session — no factory, no session_id (both early-exit branches)
# ---------------------------------------------------------------------------


async def test_close_db_session_no_factory():
    svc = _service()
    await svc._close_db_session()  # must not raise


async def test_close_db_session_no_session_id():
    svc = _service()
    svc._db_session_factory = MagicMock()  # factory set but no session_id
    await svc._close_db_session()  # must not raise


# ---------------------------------------------------------------------------
# stream_state — timeout branch yields current state
# ---------------------------------------------------------------------------


async def test_stream_state_timeout_yields_current_state():
    svc = _service()
    # Patch wait_for to immediately raise TimeoutError
    with patch("neurolink.service.asyncio.wait_for", side_effect=TimeoutError):
        gen = svc.stream_state()
        state = await gen.__anext__()
        assert hasattr(state, "frame_count")
        # StopAsyncIteration raised on next call since timeout only yields once
        with pytest.raises(StopAsyncIteration):
            await gen.__anext__()


# ---------------------------------------------------------------------------
# stream_state — CancelledError exits cleanly
# ---------------------------------------------------------------------------


async def test_stream_state_cancelled_unregisters_queue():
    svc = _service()

    async def _run():
        async for _ in svc.stream_state():
            break  # consume one state then stop

    # Push a real state so the queue yields once
    from neurolink.models.eeg import BandPowers, IngestPayload

    p = IngestPayload(
        source="mock",
        address="mock",
        timestamp=0.0,
        bands=BandPowers(alpha=0.3, theta=0.2, beta=0.15, delta=0.1, gamma=0.05),
        poor_contact=False,
    )

    task = asyncio.create_task(_run())
    await asyncio.sleep(0.01)
    svc._hub.update(p)  # triggers fanout → queue fills → generator yields
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    # After cancel the SSE queue should be unregistered
    assert len(svc._hub._sse_queues) == 0
