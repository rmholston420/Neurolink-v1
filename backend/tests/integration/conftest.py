"""Integration-test fixtures (app + async HTTP client)."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient


@pytest.fixture()
def app():
    """Fresh FastAPI app for each integration test."""
    from neurolink.main import create_app
    return create_app()


@pytest.fixture()
async def client(app):
    """Async httpx client wired to the ASGI app."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as c:
        yield c
