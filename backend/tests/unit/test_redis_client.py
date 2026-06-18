"""Unit tests for cache.redis_client (Redis-disabled path)."""

from __future__ import annotations

import os


async def test_push_state_noop_when_disabled():
    """push_state is a no-op when NEUROLINK_REDIS_ENABLED=false."""
    os.environ["NEUROLINK_REDIS_ENABLED"] = "false"
    import neurolink.config as config_module

    config_module._settings = None

    from neurolink.cache.redis_client import push_state

    # Should return None without raising
    result = await push_state({"connected": False, "frame_count": 0})
    assert result is None


async def test_get_state_returns_none_when_disabled():
    """get_state returns None when NEUROLINK_REDIS_ENABLED=false."""
    os.environ["NEUROLINK_REDIS_ENABLED"] = "false"
    import neurolink.config as config_module

    config_module._settings = None

    from neurolink.cache.redis_client import get_state

    result = await get_state()
    assert result is None


async def test_push_state_handles_bad_url_gracefully():
    """push_state swallows exceptions from a bad Redis URL."""
    os.environ["NEUROLINK_REDIS_ENABLED"] = "true"
    os.environ["NEUROLINK_REDIS_URL"] = "redis://localhost:1"  # nothing listening
    import neurolink.config as config_module

    config_module._settings = None

    from neurolink.cache.redis_client import push_state

    # Should not raise — errors are caught and logged
    await push_state({"connected": False})


async def test_get_state_handles_bad_url_gracefully():
    """get_state returns None when Redis URL is unreachable."""
    os.environ["NEUROLINK_REDIS_ENABLED"] = "true"
    os.environ["NEUROLINK_REDIS_URL"] = "redis://localhost:1"  # nothing listening
    import neurolink.config as config_module

    config_module._settings = None

    from neurolink.cache.redis_client import get_state

    result = await get_state()
    assert result is None
