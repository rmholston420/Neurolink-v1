"""Unit tests for DB repository layer.

Each test uses its own isolated in-memory SQLite engine so that rows
written by other tests (or by the lifespan startup) cannot leak in.
"""

from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from neurolink.db.models import Base
from neurolink.db.repository import SessionLogRepository


# ---------------------------------------------------------------------------
# Isolated engine fixture — prevents cross-test pollution
# ---------------------------------------------------------------------------

@pytest.fixture()
async def isolated_session() -> AsyncSession:
    """Fresh in-memory SQLite engine + session scoped to one test."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        yield session
    await engine.dispose()


# ---------------------------------------------------------------------------
# Repository tests
# ---------------------------------------------------------------------------

async def test_create_session_log(isolated_session):
    repo = SessionLogRepository(isolated_session)
    log = await repo.create(device_model="muse-s", adapter_type="ble")
    assert log.id is not None
    assert log.device_model == "muse-s"


async def test_update_session_log(isolated_session):
    repo = SessionLogRepository(isolated_session)
    log = await repo.create(device_model="muse-s", adapter_type="ble")
    updated = await repo.update(log.id, duration_sec=120.0, notes="test note")
    assert updated is not None
    assert updated.duration_sec == 120.0
    assert updated.notes == "test note"


async def test_get_session_log(isolated_session):
    repo = SessionLogRepository(isolated_session)
    log = await repo.create(device_model="muse-s", adapter_type="ble")
    fetched = await repo.get(log.id)
    assert fetched is not None
    assert fetched.id == log.id


async def test_list_recent_sessions(isolated_session):
    repo = SessionLogRepository(isolated_session)
    for i in range(3):
        await repo.create(device_model=f"device-{i}", adapter_type="mock")
    sessions = await repo.list_recent(limit=10)
    assert len(sessions) == 3


async def test_list_recent_empty_db_returns_empty(isolated_session):
    """An empty DB returns an empty list."""
    repo = SessionLogRepository(isolated_session)
    sessions = await repo.list_recent(limit=10)
    assert sessions == []
