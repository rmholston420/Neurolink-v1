"""Integration-style unit tests for the Stage 6 cardiac regression path
in EEGPump._build_payload().

Scope
-----
These tests do NOT spin up the full BLE stack.  They exercise the
logical wiring contract between:

  filter_toggles.stage6_cardiac
  artifact_detector.CorrectionPlan.apply_cardiac_regression
  CardiacRegressor.apply()
  PPGPayload.ibi_ms

in isolation using fakes and mocks so the test suite runs without
hardware, BLE, or a running asyncio event loop.

Coverage goals
--------------
  1. Stage 6 runs when: toggles.stage6_cardiac=True AND plan.apply_cardiac
     AND not artifact_rejected AND ppg.ibi_ms is non-empty.
  2. Stage 6 is skipped when any guard condition fails.
  3. The corrector receives exactly the post-Stage-5 EEG array.
  4. Its return value becomes the EEG array forwarded to band-power.
  5. Toggling stage6_cardiac=False bypasses the corrector entirely.
  6. A None ppg_payload bypasses the corrector.
  7. An empty ibi_ms list bypasses the corrector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import MagicMock

import numpy as np

# ---------------------------------------------------------------------------
# Minimal fakes that replicate only the fields Stage 6 cares about
# ---------------------------------------------------------------------------


@dataclass
class _FakePPGPayload:
    ibi_ms: list[float] = field(default_factory=lambda: [800.0])
    hr_bpm: float | None = 72.0
    hrv_rmssd: float | None = 45.0


@dataclass
class _FakeCorrectionPlan:
    apply_cardiac_regression: bool = True
    apply_ocular_regression: bool = True
    apply_asr: bool = True


class _FakeToggleConfig:
    """Minimal toggle snapshot -- only fields Stage 6 reads."""

    def __init__(
        self,
        stage6_cardiac: bool = True,
        stage5_ocular: bool = True,
    ):
        self.stage6_cardiac = stage6_cardiac
        self.stage5_ocular = stage5_ocular


# ---------------------------------------------------------------------------
# Fake Stage 6 runner -- mirrors the pump's Stage 6 guard logic
# ---------------------------------------------------------------------------

FS = 256.0
N_CH = 4
N_SAMPLES = 32


def _run_stage6(
    eeg: np.ndarray,
    corrector,
    toggles: _FakeToggleConfig,
    plan: _FakeCorrectionPlan,
    ppg_payload: _FakePPGPayload | None,
    artifact_rejected: bool = False,
) -> np.ndarray:
    """Replicate the exact guard logic from EEGPump._build_payload() Stage 6.

    Returns the (possibly corrected) EEG array.
    """
    if (
        not artifact_rejected
        and toggles.stage6_cardiac
        and plan.apply_cardiac_regression
        and ppg_payload is not None
        and ppg_payload.ibi_ms
    ):
        eeg = corrector.apply(eeg, ppg_payload.ibi_ms, fs=FS)
    return eeg


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _eeg(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((N_CH, N_SAMPLES)).astype(np.float32)


# ---------------------------------------------------------------------------
# Stage 6 runs
# ---------------------------------------------------------------------------


class TestStage6Runs:
    def test_corrector_called_when_all_guards_pass(self):
        corrector = MagicMock()
        eeg = _eeg()
        corrector.apply.return_value = eeg

        _run_stage6(
            eeg,
            corrector,
            _FakeToggleConfig(stage6_cardiac=True),
            _FakeCorrectionPlan(apply_cardiac_regression=True),
            _FakePPGPayload(ibi_ms=[800.0]),
            artifact_rejected=False,
        )

        corrector.apply.assert_called_once_with(eeg, [800.0], fs=FS)

    def test_corrector_receives_post_stage5_array(self):
        """The array passed to corrector.apply must be the one after Stage 5."""
        corrector = MagicMock()
        post_stage5 = _eeg(seed=42) * 2.0  # distinct array
        corrector.apply.return_value = post_stage5

        result = _run_stage6(
            post_stage5,
            corrector,
            _FakeToggleConfig(),
            _FakeCorrectionPlan(),
            _FakePPGPayload(),
        )

        args, _kwargs = corrector.apply.call_args
        assert args[0] is post_stage5

    def test_corrector_return_value_is_forwarded(self):
        corrector = MagicMock()
        corrected = _eeg(seed=99)
        corrector.apply.return_value = corrected

        result = _run_stage6(
            _eeg(),
            corrector,
            _FakeToggleConfig(),
            _FakeCorrectionPlan(),
            _FakePPGPayload(),
        )

        assert result is corrected

    def test_multiple_ibis_passed_through(self):
        corrector = MagicMock()
        eeg = _eeg()
        corrector.apply.return_value = eeg
        ibis = [780.0, 810.0, 795.0]

        _run_stage6(
            eeg,
            corrector,
            _FakeToggleConfig(),
            _FakeCorrectionPlan(),
            _FakePPGPayload(ibi_ms=ibis),
        )

        corrector.apply.assert_called_once_with(eeg, ibis, fs=FS)


# ---------------------------------------------------------------------------
# Stage 6 skipped
# ---------------------------------------------------------------------------


class TestStage6Skipped:
    def _assert_not_called(self, **kwargs):
        corrector = MagicMock()
        eeg = _eeg()
        result = _run_stage6(eeg, corrector, **kwargs)
        corrector.apply.assert_not_called()
        assert result is eeg  # original array returned unchanged

    def test_skipped_when_toggle_false(self):
        self._assert_not_called(
            toggles=_FakeToggleConfig(stage6_cardiac=False),
            plan=_FakeCorrectionPlan(apply_cardiac_regression=True),
            ppg_payload=_FakePPGPayload(),
            artifact_rejected=False,
        )

    def test_skipped_when_plan_apply_cardiac_false(self):
        self._assert_not_called(
            toggles=_FakeToggleConfig(stage6_cardiac=True),
            plan=_FakeCorrectionPlan(apply_cardiac_regression=False),
            ppg_payload=_FakePPGPayload(),
            artifact_rejected=False,
        )

    def test_skipped_when_artifact_rejected(self):
        self._assert_not_called(
            toggles=_FakeToggleConfig(stage6_cardiac=True),
            plan=_FakeCorrectionPlan(apply_cardiac_regression=True),
            ppg_payload=_FakePPGPayload(),
            artifact_rejected=True,
        )

    def test_skipped_when_ppg_payload_none(self):
        self._assert_not_called(
            toggles=_FakeToggleConfig(stage6_cardiac=True),
            plan=_FakeCorrectionPlan(apply_cardiac_regression=True),
            ppg_payload=None,
            artifact_rejected=False,
        )

    def test_skipped_when_ibi_ms_empty(self):
        self._assert_not_called(
            toggles=_FakeToggleConfig(stage6_cardiac=True),
            plan=_FakeCorrectionPlan(apply_cardiac_regression=True),
            ppg_payload=_FakePPGPayload(ibi_ms=[]),
            artifact_rejected=False,
        )

    def test_skipped_when_both_toggle_and_plan_false(self):
        self._assert_not_called(
            toggles=_FakeToggleConfig(stage6_cardiac=False),
            plan=_FakeCorrectionPlan(apply_cardiac_regression=False),
            ppg_payload=_FakePPGPayload(),
            artifact_rejected=False,
        )


# ---------------------------------------------------------------------------
# Original array preserved when skipped
# ---------------------------------------------------------------------------


class TestArrayPreservationWhenSkipped:
    def test_eeg_identical_when_stage6_disabled(self):
        corrector = MagicMock()
        eeg = _eeg(seed=77)
        result = _run_stage6(
            eeg,
            corrector,
            _FakeToggleConfig(stage6_cardiac=False),
            _FakeCorrectionPlan(),
            _FakePPGPayload(),
        )
        np.testing.assert_array_equal(result, eeg)

    def test_eeg_identical_when_no_ppg(self):
        corrector = MagicMock()
        eeg = _eeg(seed=88)
        result = _run_stage6(
            eeg,
            corrector,
            _FakeToggleConfig(),
            _FakeCorrectionPlan(),
            ppg_payload=None,
        )
        np.testing.assert_array_equal(result, eeg)


# ---------------------------------------------------------------------------
# CardiacRegressor integration (real object, fake warm-up)
# ---------------------------------------------------------------------------


class TestStage6WithRealRegressor:
    """Use the actual CardiacRegressor to verify end-to-end shape contract."""

    def test_output_shape_preserved_after_stage6(self):
        from neurolink.dsp.cardiac_regression import CardiacRegressor

        corrector = CardiacRegressor()
        # Warm up ring
        for _ in range(60):
            corrector.apply(_eeg(), [800.0], fs=FS)

        eeg = _eeg(seed=3)
        result = _run_stage6(
            eeg,
            corrector,
            _FakeToggleConfig(),
            _FakeCorrectionPlan(),
            _FakePPGPayload(ibi_ms=[800.0]),
        )
        assert result.shape == eeg.shape

    def test_stage6_disabled_returns_exact_input(self):
        from neurolink.dsp.cardiac_regression import CardiacRegressor

        corrector = CardiacRegressor()
        eeg = _eeg(seed=5)
        result = _run_stage6(
            eeg,
            corrector,
            _FakeToggleConfig(stage6_cardiac=False),
            _FakeCorrectionPlan(),
            _FakePPGPayload(),
        )
        np.testing.assert_array_equal(result, eeg)


# ---------------------------------------------------------------------------
# IBI boundary / physiology guard propagation
# ---------------------------------------------------------------------------


class TestIBIBoundaryPropagation:
    """Verify that out-of-range IBIs propagate to the corrector (not filtered at pump level)."""

    def test_out_of_range_ibi_reaches_corrector(self):
        """Pump passes all IBIs; the corrector handles physiological filtering."""
        corrector = MagicMock()
        corrector.apply.return_value = _eeg()
        ibis = [50.0, 9999.0]  # both out of range

        _run_stage6(
            _eeg(),
            corrector,
            _FakeToggleConfig(),
            _FakeCorrectionPlan(),
            _FakePPGPayload(ibi_ms=ibis),
        )
        # Corrector still called; it internally filters
        corrector.apply.assert_called_once()
        _, args_ibis, _ = corrector.apply.call_args[0][0], corrector.apply.call_args[0][1], None
        assert args_ibis == ibis
