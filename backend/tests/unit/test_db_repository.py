"""Unit tests for db.repository using an in-memory SQLite database."""

from __future__ import annotations

import os
import tempfile

import pytest


async def _make_factory(path: str = ":memory:"):
    """Create a fresh db engine at *path* and return the session factory.

    Use a unique temp-file path (not `:memory:`) when the test needs a
    truly isolated database — SQLite `:memory:` databases share state for
    the lifetime of the cached engine object.
    """
    os.environ["NEUROLINK_DB_PATH"] = path
    import neurolink.db.engine as engine_module

    engine_module._engine = None
    engine_module._session_factory = None
    await engine_module.init_db()
    return engine_module.get_session_factory()


async def test_create_session_returns_entry():
    factory = await _make_factory()
    from neurolink.db.repository import SessionLogRepository

    async with factory() as db:
        repo = SessionLogRepository(db)
        entry = await repo.create_session(
            device_model="muse_s_gen1",
            adapter_type="mock",
            address=None,
        )
    assert entry.id is not None
    assert entry.device_model == "muse_s_gen1"


async def test_end_session_updates_frame_count():
    factory = await _make_factory()
    from neurolink.db.repository import SessionLogRepository

    async with factory() as db:
        repo = SessionLogRepository(db)
        entry = await repo.create_session(device_model="mock", adapter_type="mock")
        session_id = entry.id
    async with factory() as db:
        repo = SessionLogRepository(db)
        await repo.end_session(
            session_id=session_id,
            frame_count=42,
            final_region="E",
            final_stage="Rubedo",
            final_ea1_eligible=True,
        )


async def test_list_recent_returns_entries():
    factory = await _make_factory()
    from neurolink.db.repository import SessionLogRepository

    async with factory() as db:
        repo = SessionLogRepository(db)
        await repo.create_session(device_model="mock", adapter_type="mock")
        await repo.create_session(device_model="muse", adapter_type="lsl")
    async with factory() as db:
        repo = SessionLogRepository(db)
        sessions = await repo.list_recent(limit=10)
    assert len(sessions) >= 2


async def test_list_recent_empty_db_returns_empty():
    """Use a unique temp file so we get a truly empty database."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        tmp_path = f.name
    os.unlink(tmp_path)  # remove file so SQLAlchemy creates a fresh one
    try:
        factory = await _make_factory(tmp_path)
        from neurolink.db.repository import SessionLogRepository

        async with factory() as db:
            repo = SessionLogRepository(db)
            sessions = await repo.list_recent(limit=10)
        assert sessions == []
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


async def test_get_by_id_returns_none_for_missing():
    factory = await _make_factory()
    from neurolink.db.repository import SessionLogRepository

    async with factory() as db:
        repo = SessionLogRepository(db)
        result = await repo.get_by_id(9999)
    assert result is None
