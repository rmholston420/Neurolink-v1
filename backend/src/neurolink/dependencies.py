"""FastAPI dependency providers for Neurolink.

All route handlers should use Depends() from here.
Never instantiate services or hub directly in route handlers.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends

from neurolink.hub import EEGHub, get_hub
from neurolink.service import NeuroLinkService

# Module-level singleton service instance
_service: NeuroLinkService | None = None


def get_neurolink_service() -> NeuroLinkService:
    """Return the global NeuroLinkService instance.

    Created lazily on first call.
    """
    global _service
    if _service is None:
        hub = get_hub()
        _service = NeuroLinkService(hub)
    return _service


# Alias used by integration tests
get_service = get_neurolink_service


def get_eeg_hub() -> EEGHub:
    """Return the global EEGHub instance."""
    return get_hub()


ServiceDep = Annotated[NeuroLinkService, Depends(get_neurolink_service)]
HubDep = Annotated[EEGHub, Depends(get_eeg_hub)]
