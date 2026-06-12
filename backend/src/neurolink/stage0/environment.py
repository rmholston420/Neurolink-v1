"""Stage 0 — Environment Checklist & Stabilisation Countdown.

Guides the user through setup steps that prevent non-physiological artifacts:

  1. Move away from switching power supplies and USB hubs (EMI)
  2. Remove / power-off nearby cell phones (GSM burst noise)
  3. Ensure stable electrode seating (impedance, cable movement)
  4. Wait 60 s after placement for dry electrodes to stabilise
     (skin-electrode interface polarisation)

The checklist exposes a `is_ready` gate that Stage0Guard checks before
allowing the EEGPump to feed data to hub.update().
"""

from __future__ import annotations

import time

# ------------------------------------------------------------------
# Ordered environment prompts shown to the user
# ------------------------------------------------------------------

ENVIRONMENT_PROMPTS: list[dict] = [
    {
        "id": "emi_distance",
        "title": "Move away from interference sources",
        "body": (
            "Step at least 1 metre away from switching power supplies, "
            "laptop chargers, USB hubs, fluorescent lights, and monitors. "
            "These emit EMI that appears as power-line harmonics and broadband "
            "noise in frontal EEG channels."
        ),
        "icon": "zap-off",
    },
    {
        "id": "phone_distance",
        "title": "Remove or silence nearby cell phones",
        "body": (
            "Place your phone in aeroplane mode or move it at least 2 metres "
            "away. GSM/LTE burst transmissions (217 Hz pulse envelope) inject "
            "artifact transients into temporal EEG channels."
        ),
        "icon": "smartphone-off",
    },
    {
        "id": "electrode_seating",
        "title": "Verify electrode seating",
        "body": (
            "Check that all electrodes are firmly seated against the scalp. "
            "For behind-the-ear contacts (TP9/TP10), press gently and fold "
            "the ear forward if needed. Loose contacts cause step-function "
            "artifacts and high impedance readings."
        ),
        "icon": "activity",
    },
    {
        "id": "stabilise",
        "title": "Wait 60 s for dry electrodes to stabilise",
        "body": (
            "Dry electrodes need time for the skin-electrode interface to "
            "polarise and impedance to fall. Sit still for 60 seconds after "
            "placing the headset before starting your session. The countdown "
            "below tracks this automatically."
        ),
        "icon": "timer",
    },
]

# Stabilisation duration for dry electrodes (seconds)
_DRY_STABILISE_SEC: float = 60.0
_SEMI_WET_STABILISE_SEC: float = 10.0


class EnvironmentChecklist:
    """Tracks user acknowledgement of environment prompts and stabilisation.

    Lifecycle
    ---------
    1. Instantiated when the adapter connects (service.connect()).
    2. Frontend polls GET /api/v1/stage0/status to render the checklist.
    3. User acknowledges each step via POST /api/v1/stage0/environment/ack.
    4. Stabilisation countdown starts when the adapter connects.
    5. is_ready -> True when all steps acked AND stabilise_remaining_s == 0.
    """

    def __init__(self, electrode_type: str = "dry") -> None:
        self._start_ts: float = time.time()
        self._stabilise_duration: float = (
            _DRY_STABILISE_SEC if electrode_type == "dry" else _SEMI_WET_STABILISE_SEC
        )
        self._acked: set[str] = set()
        self._all_prompt_ids: list[str] = [p["id"] for p in ENVIRONMENT_PROMPTS]

    # ------------------------------------------------------------------
    # Stabilisation countdown
    # ------------------------------------------------------------------

    @property
    def stabilise_remaining_s(self) -> float:
        """Seconds remaining in the stabilisation countdown (≥ 0)."""
        elapsed = time.time() - self._start_ts
        return max(0.0, self._stabilise_duration - elapsed)

    @property
    def stabilise_complete(self) -> bool:
        return self.stabilise_remaining_s == 0.0

    # ------------------------------------------------------------------
    # Step acknowledgement
    # ------------------------------------------------------------------

    def acknowledge(self, step_id: str) -> bool:
        """Mark a checklist step as acknowledged by the user.

        Returns True if the step_id is valid, False otherwise.
        """
        if step_id not in self._all_prompt_ids:
            return False
        self._acked.add(step_id)
        return True

    def acknowledge_all(self) -> None:
        """Acknowledge all steps at once (convenience for testing / mock mode)."""
        self._acked = set(self._all_prompt_ids)

    @property
    def all_steps_acked(self) -> bool:
        return self._acked >= set(self._all_prompt_ids)

    # ------------------------------------------------------------------
    # Readiness gate
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        """True when both the countdown has expired and all steps are acked."""
        return self.stabilise_complete and self.all_steps_acked

    # ------------------------------------------------------------------
    # Reset (called on disconnect)
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._start_ts = time.time()
        self._acked.clear()

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def status_dict(self) -> dict:
        remaining = round(self.stabilise_remaining_s, 1)
        return {
            "is_ready": self.is_ready,
            "stabilise_remaining_s": remaining,
            "stabilise_complete": self.stabilise_complete,
            "all_steps_acked": self.all_steps_acked,
            "acked_steps": sorted(self._acked),
            "prompts": [
                {
                    **p,
                    "acked": p["id"] in self._acked,
                }
                for p in ENVIRONMENT_PROMPTS
            ],
        }
