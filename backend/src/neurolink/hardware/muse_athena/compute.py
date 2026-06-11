"""Band power computation for Muse S Athena (Gen 2).

Ported from Rigpa-v3 hardware/muse_athena/compute.py.
Delegates to muse_s compute for EEG; adds fNIRS support.
"""

from __future__ import annotations

from neurolink.hardware.muse_s.compute import compute_all_bands

# Re-export for Athena consumers
__all__ = ["compute_all_bands"]
