"""Unit tests for db.engine — create engine, session factory, reset."""

from __future__ import annotations

import os


async def test_get_engine_creates_engine():
    os.environ["NEUROLINK_DB_PATH"] = ":memory:"
    import neurolink.db.engine as engine_module

    engine_module._engine = None
    engine_module._session_factory = None
    engine = engine_module.get_engine()
    assert engine is not None


async def test_get_session_factory_returns_callable():
    os.environ["NEUROLINK_DB_PATH"] = ":memory:"
    import neurolink.db.engine as engine_module

    engine_module._engine = None
    engine_module._session_factory = None
    factory = engine_module.get_session_factory()
    assert callable(factory)


async def test_init_db_creates_tables():
    os.environ["NEUROLINK_DB_PATH"] = ":memory:"
    import neurolink.db.engine as engine_module

    engine_module._engine = None
    engine_module._session_factory = None
    await engine_module.init_db()
    # Should not raise; tables created successfully


async def test_engine_reuse():
    """get_engine() returns same object on second call (cached)."""
    import neurolink.db.engine as engine_module

    engine_module._engine = None
    engine_module._session_factory = None
    e1 = engine_module.get_engine()
    e2 = engine_module.get_engine()
    assert e1 is e2
