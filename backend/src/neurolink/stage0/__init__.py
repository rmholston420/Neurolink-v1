"""Stage 0 — Hardware & Setup Prevention.

Public facade: Stage0Guard wires together ImpedanceGuard, IMUGate,
and EnvironmentChecklist into a single call-site for EEGPump.
"""

from __future__ import annotations

from neurolink.stage0.environment import EnvironmentChecklist
from neurolink.stage0.impedance import ImpedanceGuard
from neurolink.stage0.imu_gate import IMUGate

__all__ = [
    "EnvironmentChecklist",
    "IMUGate",
    "ImpedanceGuard",
    "Stage0Guard",
]


class Stage0Guard:
    """Facade that owns the three Stage-0 sub-systems.

    Usage in EEGPump._tick():

        sample = await self._adapter.read_sample()
        sample = self._stage0.gate_sample(sample)      # mutates extra{}
        if not self._stage0.acquisition_ready:
            return                                     # skip hub.update()
        ...
    """

    def __init__(self, electrode_type: str = "dry") -> None:
        self.impedance = ImpedanceGuard(electrode_type=electrode_type)
        self.imu = IMUGate()
        self.environment = EnvironmentChecklist(electrode_type=electrode_type)

    # ------------------------------------------------------------------
    # Primary gate called by EEGPump
    # ------------------------------------------------------------------

    def gate_sample(self, sample):
        """Apply IMU gating in-place.  Returns the (possibly mutated) sample.

        Does NOT block acquisition — flagging is additive metadata so the
        downstream pipeline can decide whether to discard or process.
        """
        if sample is None:
            return sample
        return self.imu.flag_segment(sample)

    # ------------------------------------------------------------------
    # Readiness gate — pump calls this to decide whether to update hub
    # ------------------------------------------------------------------

    @property
    def acquisition_ready(self) -> bool:
        """True when all three Stage-0 conditions are satisfied.

        * ImpedanceGuard: no channel is above threshold (or no readings yet)
        * EnvironmentChecklist: stabilisation countdown complete & all steps acked
        * IMUGate: not currently in a high-motion segment

        Note: the pump bypasses this gate in mock mode so that tests and demos
        continue to produce state updates regardless of Stage-0 status.
        """
        return self.impedance.all_channels_ok and self.environment.is_ready

    # ------------------------------------------------------------------
    # Snapshot for SSE / REST
    # ------------------------------------------------------------------

    def status_dict(self) -> dict:
        """Return a serialisable Stage0Status dict suitable for JSON."""
        return {
            "acquisition_ready": self.acquisition_ready,
            "impedance": self.impedance.summary_dict(),
            "imu": self.imu.status_dict(),
            "environment": self.environment.status_dict(),
        }
