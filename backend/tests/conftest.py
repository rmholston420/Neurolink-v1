"""Shared pytest fixtures for Neurolink tests."""

from __future__ import annotations

import os

import pytest

# Force mock adapter and in-memory DB for all tests
os.environ.setdefault("NEUROLINK_ADAPTER_TYPE", "mock")
os.environ.setdefault("NEUROLINK_DB_PATH", ":memory:")
os.environ.setdefault("NEUROLINK_REDIS_ENABLED", "false")
os.environ.setdefault("NEUROLINK_REDIS_URL", "")


@pytest.fixture(autouse=True)
def reset_hub():
    """Clear hub state before and after every test."""
    import neurolink.hub as hub_module

    hub_module.reset()
    # Reset service singleton
    import neurolink.dependencies as deps_module

    deps_module._service = None
    # Reset settings singleton
    import neurolink.config as config_module

    config_module._settings = None
    yield
    hub_module.reset()
    # Reset engine between tests to avoid connection pool issues
    import neurolink.db.engine as engine_module

    engine_module._engine = None
    engine_module._session_factory = None


@pytest.fixture
def app():
    """Create a fresh FastAPI app instance for testing."""
    import os

    os.environ["NEUROLINK_ADAPTER_TYPE"] = "mock"
    os.environ["NEUROLINK_DB_PATH"] = ":memory:"
    from neurolink.main import create_app

    return create_app()


@pytest.fixture
async def client(app):
    """Async HTTP test client."""
    from httpx import ASGITransport, AsyncClient

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac
