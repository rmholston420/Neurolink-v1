"""SQLAlchemy ORM model for EEG session log."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class SessionLog(Base):
    """Records every EEG connection session."""

    __tablename__ = "session_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    device_model: Mapped[str] = mapped_column(String(64), nullable=False, default="mock")
    adapter_type: Mapped[str] = mapped_column(String(32), nullable=False, default="mock")
    address: Mapped[str | None] = mapped_column(String(64), nullable=True)
    frame_count: Mapped[int] = mapped_column(Integer, default=0)
    final_region: Mapped[str | None] = mapped_column(String(8), nullable=True)
    final_stage: Mapped[str | None] = mapped_column(String(64), nullable=True)
    final_ea1_eligible: Mapped[bool] = mapped_column(Boolean, default=False)
