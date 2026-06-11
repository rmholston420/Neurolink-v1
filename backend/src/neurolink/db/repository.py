"""Repository layer for SessionLog DB operations.

All DB access goes through these methods.
"""

from __future__ import annotations

import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from neurolink.db.models import SessionLog


class SessionLogRepository:
    """Data access layer for SessionLog table."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create_session(
        self,
        device_model: str,
        adapter_type: str,
        address: str | None = None,
    ) -> SessionLog:
        """Create and persist a new session log entry."""
        entry = SessionLog(
            device_model=device_model,
            adapter_type=adapter_type,
            address=address,
            started_at=datetime.datetime.utcnow(),
        )
        self._session.add(entry)
        await self._session.commit()
        await self._session.refresh(entry)
        return entry

    async def end_session(
        self,
        session_id: int,
        frame_count: int,
        final_region: str | None = None,
        final_stage: str | None = None,
        final_ea1_eligible: bool | None = None,
    ) -> SessionLog | None:
        """Mark a session as ended and update final state."""
        result = await self._session.execute(select(SessionLog).where(SessionLog.id == session_id))
        entry = result.scalar_one_or_none()
        if entry:
            entry.ended_at = datetime.datetime.utcnow()
            entry.frame_count = frame_count
            entry.final_region = final_region
            entry.final_stage = final_stage
            entry.final_ea1_eligible = final_ea1_eligible
            await self._session.commit()
            await self._session.refresh(entry)
        return entry

    async def list_recent(self, limit: int = 20) -> list[SessionLog]:
        """Return the most recent session log entries."""
        result = await self._session.execute(
            select(SessionLog).order_by(SessionLog.id.desc()).limit(limit)
        )
        return list(result.scalars().all())

    async def get_by_id(self, session_id: int) -> SessionLog | None:
        """Return a session log entry by ID."""
        result = await self._session.execute(select(SessionLog).where(SessionLog.id == session_id))
        return result.scalar_one_or_none()
