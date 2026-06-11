"""Coverage tests for adapter_factory.py and hardware/mock.py."""
from __future__ import annotations

import pytest

from neurolink.adapter_factory import build_adapter
from neurolink.hardware.mock import MockAdapter


# ===========================================================================
# MockAdapter
# ===========================================================================

async def test_mock_adapter_connect_disconnect():
    adapter = MockAdapter()
    assert adapter.is_connected is False
    await adapter.connect()
    assert adapter.is_connected is True
    assert adapter.source_name == "mock"
    await adapter.disconnect()
    assert adapter.is_connected is False


async def test_mock_adapter_read_sample_while_connected():
    adapter = MockAdapter()
    await adapter.connect()
    sample = await adapter.read_sample()
    assert sample is not None
    assert sample.source == "mock"
    assert len(sample.channels) == 5
    assert sample.eeg_buffer is not None
    assert sample.ppg_buffer is not None
    assert sample.accel_buffer is not None
    assert sample.gyro_buffer is not None
    await adapter.disconnect()


async def test_mock_adapter_read_sample_while_disconnected():
    adapter = MockAdapter()
    # Not connected — should return None immediately
    sample = await adapter.read_sample()
    assert sample is None


# ===========================================================================
# adapter_factory.build_adapter
# ===========================================================================

def test_build_adapter_mock():
    adapter = build_adapter(adapter_type="mock", device_model="mock", address=None)
    assert isinstance(adapter, MockAdapter)


def test_build_adapter_unknown_raises():
    with pytest.raises((ValueError, KeyError, NotImplementedError)):
        build_adapter(adapter_type="unknown_xyz", device_model="mock", address=None)
