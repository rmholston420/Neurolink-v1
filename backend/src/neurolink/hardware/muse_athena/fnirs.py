"""fNIRS decoder for Muse S Athena via OpenMuse LSL outlet.

The Athena headset exposes alternating oxy/deoxy channels in its fNIRS stream.
This decoder averages across channels to produce scalar oxy/deoxy values.
"""

from __future__ import annotations


class FNIRSDecoder:
    """Decode raw fNIRS LSL sample into oxy/deoxy dict."""

    def decode(self, raw: list[float]) -> dict[str, float]:
        """Decode a raw fNIRS LSL sample.

        Alternating layout: [oxy0, deoxy0, oxy1, deoxy1, ...]

        Args:
            raw: Raw fNIRS sample values.

        Returns:
            Dict with 'fnirs_oxy' and 'fnirs_deoxy' float values.
        """
        if not raw:
            return {"fnirs_oxy": 0.0, "fnirs_deoxy": 0.0}

        oxy = [raw[i] for i in range(0, len(raw), 2)]
        deoxy = [raw[i] for i in range(1, len(raw), 2)]

        return {
            "fnirs_oxy": float(sum(oxy) / len(oxy)) if oxy else 0.0,
            "fnirs_deoxy": float(sum(deoxy) / len(deoxy)) if deoxy else 0.0,
        }
