"""Shared pytest fixtures for Neurolink tests."""
from __future__ import annotations

import os
import pytest
from httpx import AsyncClient, ASGITransport

# Force mock adapter before any import
os.environ.setdefault("NEUROLINK_ADAPTER_TYPE", "mock")
os.environ.setdefault("NEUROLINK_DB_PATH", ":memory:")
os.environ.setdefault("NEUROLINK_REDIS_URL", "")


@pytest.fixture(autouse=True)
def reset_hub():
    """Clear hub state before every test."""
    import neurolink.hub as hub_module
    hub_module.reset()
    yield
    hub_module.reset()


@pytest.fixture
def app():
    os.environ["NEUROLINK_ADAPTER_TYPE"] = "mock"
    os.environ["NEUROLINK_DB_PATH"] = ":memory:"
    # Reset config singleton to pick up env changes
    import neurolink.config as cfg_module
    cfg_module._settings = None
    # Reset service singleton
    import neurolink.dependencies as dep_module
    dep_module._service = None
    from neurolink.main import create_app
    return create_app()


@pytest.fixture
async def client(app):
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac
