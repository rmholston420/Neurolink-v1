"""Coverage for service.py, hub.py, adapter_factory.py, and hardware/mock.py.

Targets the branches that remain uncovered after the existing test suite:
  - service disconnect/calibration/stream/sessions edge cases
  - hub muse_ble v01 path, QueueFull, module-level delegates
  - adapter_factory all non-mock branches (imports mocked)
  - mock.py read_sample when not connected
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from neurolink.hub import EEGHub
from neurolink.models.eeg import BandPowers, IngestPayload, NeurolinkState
from neurolink.service import NeuroLinkService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _payload(source: str = "mock", **kw) -> IngestPayload:
    defaults = dict(
        source=source,
        bands=BandPowers(alpha=0.30, theta=0.15, beta=0.15, delta=0.20, gamma=0.10),
    )
    defaults.update(kw)
    return IngestPayload(**defaults)


def _service() -> NeuroLinkService:
    return NeuroLinkService(hub=EEGHub())


# ===========================================================================
# service.py
# ===========================================================================

# -- disconnect with no pump / no adapter -----------------------------------

async def test_service_disconnect_no_pump_no_adapter():
    """disconnect() with nothing connected should succeed silently."""
    svc = _service()
    result = await svc.disconnect()
    assert result.ok is True


async def test_service_disconnect_adapter_raises_exception():
    """Exception from adapter.disconnect() is logged and swallowed."""
    svc = _service()
    adapter = AsyncMock()
    adapter.is_connected = True
    adapter.disconnect = AsyncMock(side_effect=RuntimeError("BLE gone"))
    svc._adapter = adapter
    result = await svc.disconnect()
    assert result.ok is True
    assert svc._adapter is None


# -- start_calibration: no adapter -> AdapterNotConnectedError --------------

async def test_service_calibration_no_adapter_raises():
    from neurolink.exceptions import AdapterNotConnectedError
    svc = _service()
    with pytest.raises(AdapterNotConnectedError):
        await svc.start_calibration()


async def test_service_calibration_adapter_not_connected_raises():
    from neurolink.exceptions import AdapterNotConnectedError
    svc = _service()
    adapter = MagicMock()
    adapter.is_connected = False
    svc._adapter = adapter
    with pytest.raises(AdapterNotConnectedError):
        await svc.start_calibration()


# -- start_calibration: already running -> idempotent -----------------------

async def test_service_calibration_already_running_idempotent():
    svc = _service()
    adapter = AsyncMock()
    adapter.is_connected = True
    svc._adapter = adapter

    # Simulate a task that is not done
    future = asyncio.get_event_loop().create_future()
    mock_task = MagicMock()
    mock_task.done.return_value = False
    svc._calibration_task = mock_task

    result = await svc.start_calibration()
    assert result.status == "started"
    assert result.baseline_alpha is None


# -- get_sessions: no factory -> empty list ---------------------------------

async def test_service_get_sessions_no_factory():
    svc = _service()
    sessions = await svc.get_sessions()
    assert sessions == []


# -- is_connected property --------------------------------------------------

def test_service_is_connected_false_when_no_adapter():
    svc = _service()
    assert svc.is_connected is False


def test_service_is_connected_true_when_connected():
    svc = _service()
    adapter = MagicMock()
    adapter.is_connected = True
    svc._adapter = adapter
    assert svc.is_connected is True


# -- adapter_type property --------------------------------------------------

def test_service_adapter_type_default():
    svc = _service()
    assert svc.adapter_type == "mock"


# -- stream_state: TimeoutError yields current state -----------------------

async def test_service_stream_state_timeout_yields_state():
    svc = _service()
    # Don't register any SSE queue writers — the internal wait_for will timeout
    gen = svc.stream_state()
    state = await gen.__anext__()
    assert isinstance(state, NeurolinkState)
    await gen.aclose()


# -- stream_state: CancelledError exits cleanly ----------------------------

async def test_service_stream_state_cancelled_exits():
    svc = _service()
    gen = svc.stream_state()

    async def _cancel_after_first():
        state = await gen.__anext__()  # gets the timeout yield
        await gen.aclose()             # triggers CancelledError cleanup
        return state

    state = await asyncio.wait_for(_cancel_after_first(), timeout=5.0)
    assert isinstance(state, NeurolinkState)


# ===========================================================================
# hub.py — muse_ble v01 path
# ===========================================================================

def test_hub_update_muse_ble_source_runs_v01():
    """update() with source='muse_ble' must run the v01 classifier."""
    hub = EEGHub()
    p = _payload(source="muse_ble")
    state = hub.update(p)
    # v01 classifier sets region_v01 to a real region letter
    assert state.region_v01 in {"A", "B", "C", "D", "E"}
    assert state.alchemical_stage_v01 is not None


# -- _fanout QueueFull branch -----------------------------------------------

def test_hub_fanout_queue_full_drops_frame():
    """QueueFull from a full SSE queue must be swallowed (not raised)."""
    hub = EEGHub()
    q = asyncio.Queue(maxsize=1)
    hub.register_sse_queue(q)
    # Fill the queue
    q.put_nowait(NeurolinkState())
    # Now update — _fanout tries put_nowait, gets QueueFull, logs warning
    hub.update(_payload())
    hub.unregister_sse_queue(q)


# -- get_latest / set_latest_sample ----------------------------------------

def test_hub_get_latest_initially_none():
    hub = EEGHub()
    assert hub.get_latest() is None


def test_hub_set_and_get_latest_sample():
    from neurolink.hardware.base import EEGSample
    hub = EEGHub()
    sample = EEGSample(
        channels=[0.0] * 5,
        timestamp=0.0,
        source="mock",
        address="",
        poor_contact=False,
    )
    hub.set_latest_sample(sample)
    assert hub.get_latest() is sample


# -- snapshot ---------------------------------------------------------------

def test_hub_snapshot_returns_dict():
    hub = EEGHub()
    snap = hub.snapshot()
    assert isinstance(snap, dict)
    assert "frame_count" in snap


# -- unregister queue not present (ValueError swallowed) -------------------

def test_hub_unregister_queue_not_present_is_noop():
    hub = EEGHub()
    q = asyncio.Queue()
    hub.unregister_sse_queue(q)  # never registered — must not raise


# -- module-level delegates ------------------------------------------------

def test_hub_module_level_update():
    import neurolink.hub as hub_mod
    state = hub_mod.update(_payload())
    assert isinstance(state, NeurolinkState)


def test_hub_module_level_get_state():
    import neurolink.hub as hub_mod
    state = hub_mod.get_state()
    assert isinstance(state, NeurolinkState)


def test_hub_module_level_get_ea1():
    import neurolink.hub as hub_mod
    from neurolink.models.eeg import EA1Result
    result = hub_mod.get_ea1()
    assert isinstance(result, EA1Result)


def test_hub_module_level_snapshot():
    import neurolink.hub as hub_mod
    snap = hub_mod.snapshot()
    assert isinstance(snap, dict)


def test_hub_module_level_reset():
    import neurolink.hub as hub_mod
    hub_mod.update(_payload())  # make frame_count > 0
    hub_mod.reset()
    assert hub_mod.get_state().frame_count == 0


# ===========================================================================
# adapter_factory.py — non-mock branches (imports mocked)
# ===========================================================================

def test_adapter_factory_ble_muse_s_gen1():
    """ble + muse_s_gen1 -> MuseSBleAdapter instantiated."""
    mock_adapter = MagicMock()
    mock_cls = MagicMock(return_value=mock_adapter)
    with patch.dict("sys.modules", {
        "neurolink.hardware.muse_s.ble_adapter": MagicMock(MuseSBleAdapter=mock_cls)
    }):
        from neurolink.adapter_factory import create_adapter
        result = create_adapter(adapter_type="ble", device_model="muse_s_gen1", address="AA:BB")
    mock_cls.assert_called_once_with(address="AA:BB")


def test_adapter_factory_ble_muse_s_athena():
    """ble + muse_s_athena -> AthenaBlueAdapter instantiated."""
    mock_adapter = MagicMock()
    mock_cls = MagicMock(return_value=mock_adapter)
    with patch.dict("sys.modules", {
        "neurolink.hardware.muse_athena.ble_adapter": MagicMock(AthenaBlueAdapter=mock_cls)
    }):
        from neurolink.adapter_factory import create_adapter
        result = create_adapter(adapter_type="ble", device_model="muse_s_athena")
    mock_cls.assert_called_once()


def test_adapter_factory_ble_unknown_model_raises():
    from neurolink.adapter_factory import create_adapter
    with pytest.raises(ValueError, match="Unknown device_model for BLE"):
        create_adapter(adapter_type="ble", device_model="unknown_headset")


def test_adapter_factory_lsl_muse_s_athena():
    """lsl + muse_s_athena -> AthenaBlueAdapter instantiated."""
    mock_cls = MagicMock(return_value=MagicMock())
    with patch.dict("sys.modules", {
        "neurolink.hardware.muse_athena.ble_adapter": MagicMock(AthenaBlueAdapter=mock_cls)
    }):
        from neurolink.adapter_factory import create_adapter
        create_adapter(adapter_type="lsl", device_model="muse_s_athena")
    mock_cls.assert_called_once()


def test_adapter_factory_lsl_default_model():
    """lsl + any other model -> MuseSLslAdapter instantiated."""
    mock_cls = MagicMock(return_value=MagicMock())
    with patch.dict("sys.modules", {
        "neurolink.hardware.muse_s.lsl_adapter": MagicMock(MuseSLslAdapter=mock_cls)
    }):
        from neurolink.adapter_factory import create_adapter
        create_adapter(adapter_type="lsl", device_model="muse_s_gen1")
    mock_cls.assert_called_once()


def test_adapter_factory_unknown_type_raises():
    from neurolink.adapter_factory import create_adapter
    with pytest.raises(ValueError, match="Unknown adapter_type"):
        create_adapter(adapter_type="zigbee", device_model="muse_s_gen1")


# ===========================================================================
# hardware/mock.py — read_sample when not connected
# ===========================================================================

async def test_mock_adapter_read_sample_not_connected_returns_none():
    from neurolink.hardware.mock import MockAdapter
    adapter = MockAdapter()
    # Never called connect() so _connected=False
    result = await adapter.read_sample()
    assert result is None
