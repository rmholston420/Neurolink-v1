"""Unit tests for EA-1 eligibility scorer."""

from __future__ import annotations

import pytest

from neurolink.ea1_scorer import score
from neurolink.models.eeg import BandPowers, IngestPayload, PPGPayload, SSpaceCoords


class TestEA1Score:
    def test_returns_ea1_result(self, base_payload):
        from neurolink.models.eeg import EA1Result
        result = score(base_payload)
        assert isinstance(result, EA1Result)

    def test_score_in_unit_interval(self, base_payload):
        result = score(base_payload)
        assert 0.0 <= result.score <= 1.0

    def test_criteria_total_positive(self, base_payload):
        result = score(base_payload)
        assert result.criteria_total > 0

    def test_criteria_met_lte_criteria_total(self, base_payload):
        result = score(base_payload)
        assert result.criteria_met <= result.criteria_total

    def test_label_is_string(self, base_payload):
        result = score(base_payload)
        assert isinstance(result.label, str) and len(result.label) > 0

    def test_high_alpha_resting_state_increments_criteria(self, alpha_dominant_bands):
        # Alpha-dominant state with calm HR should score higher than flat bands
        flat_result = score(IngestPayload(source="mock", bands=BandPowers()))
        alpha_result = score(
            IngestPayload(
                source="mock",
                bands=alpha_dominant_bands,
                ppg=PPGPayload(hr_bpm=62, hrv_rmssd=55),
            )
        )
        assert alpha_result.score >= flat_result.score
