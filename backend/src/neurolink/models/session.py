"""SQLAlchemy ORM model for session log."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class SessionLog(Base):
    """Records every EEG session: start/end time, device, adapter, frame count."""

    __tablename__ = "session_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    device_model: Mapped[str] = mapped_column(String, nullable=False)
    adapter_type: Mapped[str] = mapped_column(String, nullable=False)
    address: Mapped[str | None] = mapped_column(String, nullable=True)
    frame_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    final_region: Mapped[str | None] = mapped_column(String, nullable=True)
    final_stage: Mapped[str | None] = mapped_column(String, nullable=True)
    final_ea1_eligible: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
