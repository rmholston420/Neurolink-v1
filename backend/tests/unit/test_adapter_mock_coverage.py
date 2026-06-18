"""Coverage tests for adapter_factory.py and hardware/mock.py."""

from __future__ import annotations

import pytest

from neurolink.adapter_factory import create_adapter
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
    sample = await adapter.read_sample()
    assert sample is None


# ===========================================================================
# adapter_factory.create_adapter
# ===========================================================================


def test_create_adapter_mock():
    adapter = create_adapter(adapter_type="mock", device_model="mock", address=None)
    assert isinstance(adapter, MockAdapter)


def test_create_adapter_unknown_raises():
    with pytest.raises(ValueError, match="Unknown adapter_type"):
        create_adapter(adapter_type="unknown_xyz", device_model="mock", address=None)
