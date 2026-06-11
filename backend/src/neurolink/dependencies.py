"""FastAPI dependency providers.

All Depends() factories are defined here. Routers import from this module.
"""
from __future__ import annotations

from functools import lru_cache
from typing import AsyncGenerator

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from neurolink.db.engine import get_session_factory
from neurolink.hub import EEGHub, get_hub
from neurolink.service import NeuroLinkService


def get_eeg_hub() -> EEGHub:
    """Return the process-global EEGHub."""
    return get_hub()


# Process-global service instance
_service: NeuroLinkService | None = None


def get_neurolink_service(
    hub: EEGHub = Depends(get_eeg_hub),
) -> NeuroLinkService:
    """Return the process-global NeuroLinkService."""
    global _service
    if _service is None or _service._hub is not hub:
        _service = NeuroLinkService(hub=hub)
    return _service


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield an async SQLAlchemy session."""
    factory = get_session_factory()
    async with factory() as session:
        yield session
