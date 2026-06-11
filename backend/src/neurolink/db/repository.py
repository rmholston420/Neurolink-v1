"""Session log repository."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Sequence

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from neurolink.models.session import SessionLog

log = structlog.get_logger(__name__)


class SessionLogRepository:
    """Async repository for SessionLog ORM model."""

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
            started_at=datetime.now(tz=timezone.utc),
            device_model=device_model,
            adapter_type=adapter_type,
            address=address,
        )
        self._session.add(entry)
        await self._session.commit()
        await self._session.refresh(entry)
        log.info("session_created", session_id=entry.id, device=device_model)
        return entry

    async def end_session(
        self,
        session_id: int,
        frame_count: int = 0,
        final_region: str | None = None,
        final_stage: str | None = None,
        final_ea1_eligible: bool = False,
    ) -> SessionLog | None:
        """Mark a session as ended with final state."""
        result = await self._session.execute(
            select(SessionLog).where(SessionLog.id == session_id)
        )
        entry = result.scalar_one_or_none()
        if entry is None:
            log.warning("session_not_found", session_id=session_id)
            return None
        entry.ended_at = datetime.now(tz=timezone.utc)
        entry.frame_count = frame_count
        entry.final_region = final_region
        entry.final_stage = final_stage
        entry.final_ea1_eligible = final_ea1_eligible
        await self._session.commit()
        await self._session.refresh(entry)
        log.info("session_ended", session_id=session_id, frames=frame_count)
        return entry

    async def list_recent(
        self, limit: int = 20
    ) -> Sequence[SessionLog]:
        """Return the most recent session log entries."""
        result = await self._session.execute(
            select(
                SessionLog.id,
                SessionLog.started_at,
                SessionLog.ended_at,
                SessionLog.device_model,
                SessionLog.adapter_type,
                SessionLog.address,
                SessionLog.frame_count,
                SessionLog.final_region,
                SessionLog.final_stage,
                SessionLog.final_ea1_eligible,
            )
            .order_by(SessionLog.started_at.desc())
            .limit(limit)
        )
        return result.all()  # type: ignore[return-value]

    async def count(self) -> int:
        """Return total number of session log entries."""
        result = await self._session.execute(
            select(SessionLog.id)
        )
        return len(result.all())
