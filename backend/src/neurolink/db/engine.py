"""SQLAlchemy async engine and session factory.

Ported from Rigpa-v3 db/engine.py.
Uses aiosqlite for SQLite backend.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from neurolink.config import get_settings

_engine = None
_session_factory = None


def get_engine():
    """Return the global async SQLAlchemy engine (lazy init)."""
    global _engine
    if _engine is None:
        settings = get_settings()
        db_path = settings.db_path
        if db_path == ":memory:":
            url = "sqlite+aiosqlite:///:memory:"
        else:
            url = f"sqlite+aiosqlite:///{db_path}"
        _engine = create_async_engine(
            url,
            echo=False,
            connect_args={"check_same_thread": False},
        )
    return _engine


async def create_tables() -> None:
    """Create all DB tables if not present."""
    from neurolink.db.models import Base

    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


# Alias used by tests and app startup
init_db = create_tables


async def dispose_engine() -> None:
    """Dispose the engine connection pool."""
    global _engine, _session_factory
    if _engine:
        await _engine.dispose()
        _engine = None
        _session_factory = None


def get_session_factory():
    """Return a session factory context manager callable."""
    engine = get_engine()
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    @asynccontextmanager
    async def session_cm() -> AsyncGenerator[AsyncSession]:
        async with factory() as session:
            yield session

    return session_cm
