"""Integration tests: MockAdapter produces valid EEGSamples with expected structure."""

from __future__ import annotations

import asyncio

import pytest

from neurolink.hardware.mock import MockAdapter


class TestMockAdapter:
    def test_read_sample_returns_sample(self):
        adapter = MockAdapter()
        asyncio.get_event_loop().run_until_complete(adapter.connect())
        sample = asyncio.get_event_loop().run_until_complete(adapter.read_sample())
        assert sample is not None

    def test_sample_has_four_channels(self):
        adapter = MockAdapter()
        asyncio.get_event_loop().run_until_complete(adapter.connect())
        sample = asyncio.get_event_loop().run_until_complete(adapter.read_sample())
        assert len(sample.eeg_buffer) == 4

    def test_sample_channels_are_nonempty(self):
        adapter = MockAdapter()
        asyncio.get_event_loop().run_until_complete(adapter.connect())
        sample = asyncio.get_event_loop().run_until_complete(adapter.read_sample())
        for ch in sample.eeg_buffer:
            assert len(ch) > 0

    def test_sample_source_is_mock(self):
        adapter = MockAdapter()
        asyncio.get_event_loop().run_until_complete(adapter.connect())
        sample = asyncio.get_event_loop().run_until_complete(adapter.read_sample())
        assert "mock" in sample.source.lower()

    def test_disconnect_does_not_raise(self):
        adapter = MockAdapter()
        asyncio.get_event_loop().run_until_complete(adapter.connect())
        asyncio.get_event_loop().run_until_complete(adapter.disconnect())

    def test_poor_contact_is_bool(self):
        adapter = MockAdapter()
        asyncio.get_event_loop().run_until_complete(adapter.connect())
        sample = asyncio.get_event_loop().run_until_complete(adapter.read_sample())
        assert isinstance(sample.poor_contact, bool)
