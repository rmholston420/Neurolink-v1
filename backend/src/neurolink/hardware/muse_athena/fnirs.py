"""fNIRS decoder for Muse S Athena (Gen 2).

Ported from Rigpa-v3 hardware/muse_athena/fnirs.py.
Decodes interleaved oxy/deoxy channels from OpenMuse LSL outlet.
"""
from __future__ import annotations

import numpy as np


class FNIRSDecoder:
    """Decodes interleaved fNIRS samples from Muse Athena.

    The OpenMuse LSL outlet interleaves oxy and deoxy channels:
        [oxy_ch0, deoxy_ch0, oxy_ch1, deoxy_ch1, ...]

    Average of all oxy channels -> fnirs_oxy
    Average of all deoxy channels -> fnirs_deoxy
    """

    def decode(self, raw_sample: list[float] | np.ndarray) -> dict[str, float]:
        """Decode a single fNIRS sample to oxy/deoxy averages.

        Args:
            raw_sample: interleaved list [oxy0, deoxy0, oxy1, deoxy1, ...]
                        Length must be even; odd index = oxy, even index+1 = deoxy.

        Returns:
            dict with 'fnirs_oxy' and 'fnirs_deoxy' keys.
        """
        if len(raw_sample) < 2:
            return {"fnirs_oxy": 0.0, "fnirs_deoxy": 0.0}

        sample = np.array(raw_sample, dtype=float)
        # Even indices (0, 2, 4, ...) = oxy channels
        oxy_vals = sample[0::2]
        # Odd indices (1, 3, 5, ...) = deoxy channels
        deoxy_vals = sample[1::2]

        fnirs_oxy = float(np.mean(oxy_vals)) if len(oxy_vals) > 0 else 0.0
        fnirs_deoxy = float(np.mean(deoxy_vals)) if len(deoxy_vals) > 0 else 0.0

        return {"fnirs_oxy": fnirs_oxy, "fnirs_deoxy": fnirs_deoxy}
