"""Unit tests for config.Settings."""

from __future__ import annotations

import os

import pytest

from neurolink.config import Settings, get_settings


class TestSettings:
    def test_defaults_are_valid(self):
        s = Settings()
        assert isinstance(s.publish_hz, (int, float))
        assert s.publish_hz > 0

    def test_publish_hz_positive(self):
        s = Settings()
        assert s.publish_hz > 0

    def test_get_settings_returns_settings(self):
        s = get_settings()
        assert isinstance(s, Settings)

    def test_env_override_publish_hz(self, monkeypatch):
        monkeypatch.setenv("NEUROLINK_PUBLISH_HZ", "8")
        # Re-instantiate to pick up env
        s = Settings()
        assert s.publish_hz == pytest.approx(8.0, abs=0.01)

    def test_redis_enabled_default_bool(self):
        s = Settings()
        assert isinstance(s.redis_enabled, bool)
