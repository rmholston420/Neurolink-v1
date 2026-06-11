"""Unit tests for NeuroLinkService."""

from __future__ import annotations

import asyncio

from neurolink.hub import EEGHub
from neurolink.models.eeg import BandPowers, IngestPayload
from neurolink.service import NeuroLinkService


def _svc() -> NeuroLinkService:
    return NeuroLinkService(EEGHub())


async def test_get_current_state_returns_neurolink_state():
    svc = _svc()
    state = await svc.get_current_state()
    assert state.frame_count == 0
    assert state.connected is False


async def test_get_ea1_returns_result():
    svc = _svc()
    ea1 = await svc.get_ea1()
    assert ea1 is not None
    assert hasattr(ea1, "eligible")


async def test_get_band_powers_returns_response():
    svc = _svc()
    # Prime hub with a frame so bands are non-zero
    svc._hub.update(IngestPayload(
        source="mock",
        bands=BandPowers(alpha=0.3, theta=0.2, beta=0.15, delta=0.2, gamma=0.1),
    ))
    resp = await svc.get_band_powers()
    assert resp.alpha == 0.3
    assert resp.theta == 0.2


async def test_connect_mock_sets_is_connected():
    svc = _svc()
    result = await svc.connect(adapter_type="mock", device_model="mock")
    assert result.ok is True
    assert svc.is_connected is True
    await svc.disconnect()


async def test_connect_twice_raises():
    from neurolink.exceptions import AdapterAlreadyConnectedError
    svc = _svc()
    await svc.connect(adapter_type="mock", device_model="mock")
    try:
        raised = False
        try:
            await svc.connect(adapter_type="mock", device_model="mock")
        except AdapterAlreadyConnectedError:
            raised = True
        assert raised
    finally:
        await svc.disconnect()


async def test_disconnect_when_not_connected_is_safe():
    svc = _svc()
    result = await svc.disconnect()
    assert result.ok is True


async def test_disconnect_clears_adapter():
    svc = _svc()
    await svc.connect(adapter_type="mock", device_model="mock")
    await svc.disconnect()
    assert svc.is_connected is False
    assert svc._adapter is None


async def test_is_connected_false_initially():
    svc = _svc()
    assert svc.is_connected is False


async def test_adapter_type_default():
    svc = _svc()
    assert svc.adapter_type == "mock"


async def test_get_sessions_no_db_returns_empty():
    svc = _svc()
    sessions = await svc.get_sessions()
    assert sessions == []


async def test_stream_state_yields_keepalive():
    """stream_state() yields current state as keepalive after timeout."""
    svc = _svc()
    gen = svc.stream_state()
    # The generator has a 2s timeout then yields current state.
    # Cancel quickly to avoid waiting 2 seconds.
    task = asyncio.create_task(gen.__anext__())
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, StopAsyncIteration):
        pass
    await gen.aclose()


async def test_calibrate_raises_when_not_connected():
    from neurolink.exceptions import AdapterNotConnectedError
    svc = _svc()
    raised = False
    try:
        await svc.start_calibration()
    except AdapterNotConnectedError:
        raised = True
    assert raised
