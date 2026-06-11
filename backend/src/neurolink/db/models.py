"""SQLAlchemy ORM models for Neurolink.

Tables:
    session_log: EEG session log with start/end time, frame count, EA1 status.
"""
from __future__ import annotations

import datetime

from sqlalchemy import Boolean, DateTime, Float, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class SessionLog(Base):
    """Session log entry for each EEG connection."""

    __tablename__ = "session_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    started_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow
    )
    ended_at: Mapped[datetime.datetime | None] = mapped_column(DateTime, nullable=True)
    device_model: Mapped[str] = mapped_column(String(64), default="")
    adapter_type: Mapped[str] = mapped_column(String(64), default="")
    address: Mapped[str | None] = mapped_column(String(128), nullable=True)
    frame_count: Mapped[int] = mapped_column(Integer, default=0)
    final_region: Mapped[str | None] = mapped_column(String(8), nullable=True)
    final_stage: Mapped[str | None] = mapped_column(String(64), nullable=True)
    final_ea1_eligible: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    baseline_alpha: Mapped[float | None] = mapped_column(Float, nullable=True)
