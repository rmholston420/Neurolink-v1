"""Integration tests: MockAdapter produces valid EEGSamples with expected structure."""

from __future__ import annotations

import asyncio

from neurolink.hardware.mock import MockAdapter


class TestMockAdapter:
    def test_read_sample_returns_sample(self):
        adapter = MockAdapter()
        asyncio.run(self._connect_and_read(adapter))

    async def _connect_and_read(self, adapter):
        await adapter.connect()
        sample = await adapter.read_sample()
        assert sample is not None
        await adapter.disconnect()

    def test_sample_has_five_channels(self):
        """MockAdapter models a Muse S (TP9/AF7/AF8/TP10/AUX) — 5 channels."""

        async def _run():
            adapter = MockAdapter()
            await adapter.connect()
            sample = await adapter.read_sample()
            assert len(sample.eeg_buffer) == 5
            await adapter.disconnect()

        asyncio.run(_run())

    def test_sample_channels_are_nonempty(self):
        async def _run():
            adapter = MockAdapter()
            await adapter.connect()
            sample = await adapter.read_sample()
            for ch in sample.eeg_buffer:
                assert len(ch) > 0
            await adapter.disconnect()

        asyncio.run(_run())

    def test_sample_source_is_mock(self):
        async def _run():
            adapter = MockAdapter()
            await adapter.connect()
            sample = await adapter.read_sample()
            assert "mock" in sample.source.lower()
            await adapter.disconnect()

        asyncio.run(_run())

    def test_disconnect_does_not_raise(self):
        async def _run():
            adapter = MockAdapter()
            await adapter.connect()
            await adapter.disconnect()

        asyncio.run(_run())

    def test_poor_contact_is_bool(self):
        async def _run():
            adapter = MockAdapter()
            await adapter.connect()
            sample = await adapter.read_sample()
            assert isinstance(sample.poor_contact, bool)
            await adapter.disconnect()

        asyncio.run(_run())
