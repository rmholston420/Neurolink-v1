"""fNIRS decoder for Muse S Athena (Gen 2).

Ported from Rigpa-v3 hardware/muse_athena/fnirs.py.
Decodes interleaved oxy/deoxy fNIRS samples from OpenMuse LSL.
"""
from __future__ import annotations


class FNIRSDecoder:
    """Decodes Athena fNIRS samples.

    OpenMuse LSL exports fNIRS as interleaved [oxy0, deoxy0, oxy1, deoxy1, ...].
    Even indices (0, 2, 4, ...) are oxygenated channels.
    Odd indices (1, 3, 5, ...) are deoxygenated channels.
    """

    def decode(self, raw_sample: list[float]) -> dict[str, float]:
        """Decode a raw fNIRS LSL sample to oxy/deoxy averages.

        Args:
            raw_sample: list of float values from OpenMuse fNIRS LSL outlet
                        [oxy0, deoxy0, oxy1, deoxy1, ...]

        Returns:
            Dict with keys:
            - "fnirs_oxy": mean oxygenated hemoglobin (even-indexed channels)
            - "fnirs_deoxy": mean deoxygenated hemoglobin (odd-indexed channels)
        """
        if not raw_sample:
            return {"fnirs_oxy": 0.0, "fnirs_deoxy": 0.0}

        oxy_vals = [raw_sample[i] for i in range(0, len(raw_sample), 2)]
        deoxy_vals = [raw_sample[i] for i in range(1, len(raw_sample), 2)]

        fnirs_oxy = float(sum(oxy_vals) / len(oxy_vals)) if oxy_vals else 0.0
        fnirs_deoxy = float(sum(deoxy_vals) / len(deoxy_vals)) if deoxy_vals else 0.0

        return {"fnirs_oxy": fnirs_oxy, "fnirs_deoxy": fnirs_deoxy}
