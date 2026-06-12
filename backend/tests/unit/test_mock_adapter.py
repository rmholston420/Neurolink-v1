"""Unit tests for hardware.mock.MockAdapter."""

from __future__ import annotations

import pytest

from neurolink.hardware.mock import MockAdapter


class TestMockAdapterContract:
    @pytest.mark.asyncio
    async def test_connect_sets_is_connected(self):
        adapter = MockAdapter()
        assert adapter.is_connected is False
        await adapter.connect()
        assert adapter.is_connected is True
        await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_disconnect_clears_is_connected(self):
        adapter = MockAdapter()
        await adapter.connect()
        await adapter.disconnect()
        assert adapter.is_connected is False

    @pytest.mark.asyncio
    async def test_stream_yields_eeg_samples(self):
        """MockAdapter.stream() should yield at least one EEGSample."""
        adapter = MockAdapter()
        await adapter.connect()
        samples = []
        async for sample in adapter.stream():
            samples.append(sample)
            if len(samples) >= 3:
                break
        await adapter.disconnect()
        assert len(samples) == 3

    @pytest.mark.asyncio
    async def test_sample_has_expected_fields(self):
        adapter = MockAdapter()
        await adapter.connect()
        async for sample in adapter.stream():
            assert hasattr(sample, "eeg")
            assert hasattr(sample, "timestamp")
            break
        await adapter.disconnect()

    def test_source_name_is_string(self):
        adapter = MockAdapter()
        assert isinstance(adapter.source_name, str)
