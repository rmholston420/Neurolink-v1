"""models/session.py — re-export alias for the SessionLog ORM model.

The spec references neurolink.models.session.SessionLog.  The canonical
ORM definition lives in neurolink.db.models so that SQLAlchemy's
DeclarativeBase is co-located with the engine.  This module re-exports
it so that both import paths resolve to the same class.
"""

from __future__ import annotations

from neurolink.db.models import SessionLog as SessionLog  # noqa: F401

__all__ = ["SessionLog"]
