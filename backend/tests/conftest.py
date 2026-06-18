"""Shared pytest fixtures for Neurolink backend tests."""

from __future__ import annotations

import numpy as np
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from neurolink.hub import EEGHub
from neurolink.models.eeg import BandPowers, IngestPayload

# ---------------------------------------------------------------------------
# Domain fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def hub() -> EEGHub:
    """Fresh EEGHub instance for each test."""
    h = EEGHub()
    yield h
    h.reset()


@pytest.fixture()
def flat_bands() -> BandPowers:
    """Equal-power band powers (0.2 each)."""
    return BandPowers(alpha=0.2, theta=0.2, beta=0.2, delta=0.2, gamma=0.2)


@pytest.fixture()
def alpha_dominant_bands() -> BandPowers:
    """Alpha-dominant band powers -- typical relaxed-focus state."""
    return BandPowers(alpha=0.55, theta=0.15, beta=0.15, delta=0.10, gamma=0.05)


@pytest.fixture()
def base_payload(flat_bands) -> IngestPayload:
    """Minimal valid IngestPayload with flat bands."""
    return IngestPayload(source="mock", bands=flat_bands)


@pytest.fixture()
def eeg_buffer_256hz() -> list[list[float]]:
    """4-channel x 256-sample synthetic EEG buffer (1 second at 256 Hz)."""
    rng = np.random.default_rng(42)
    t = np.linspace(0, 1, 256)
    channels = [
        np.sin(2 * np.pi * 10 * t) + 0.1 * rng.standard_normal(256),
        np.sin(2 * np.pi * 6 * t) + 0.1 * rng.standard_normal(256),
        np.sin(2 * np.pi * 20 * t) + 0.1 * rng.standard_normal(256),
        np.sin(2 * np.pi * 2 * t) + 0.1 * rng.standard_normal(256),
    ]
    return [ch.tolist() for ch in channels]


# ---------------------------------------------------------------------------
# FastAPI / httpx fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def app():
    """Create a fresh FastAPI application instance with reset singletons."""
    import neurolink.dependencies as deps
    import neurolink.hub as hub_mod

    deps._service = None
    hub_mod._hub = hub_mod.EEGHub()

    from neurolink.main import create_app

    return create_app()


@pytest.fixture()
async def client(app):
    """Async httpx client bound to the FastAPI app via ASGI transport."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as c:
        yield c


@pytest_asyncio.fixture()
async def async_client(app):
    """Async httpx client for SSE / streaming tests.

    Uses pytest_asyncio.fixture so it is recognised by pytest-asyncio
    in auto or strict mode.  Identical transport setup to `client`.
    """
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as c:
        yield c
